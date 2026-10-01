"""大语言模型（如 Gemma 4 12B）的零样本 / LoRA 推理器，实现与 mmBERT 相同的 ``Predictor`` 接口。

    from ajev.lm.predictor import LMPredictor
    probs = LMPredictor("google/gemma-4-12B-it").predict(decisions)

或者通过评测命令：

    python -m ajev.eval.evaluate --data data/external/jevbench_public.jsonl \\
        --predictor lm --lm-model google/gemma-4-12B-it

原理（详见 ajev/lm/prompt.py）：把题目写成带字母选项的对话，用模型的对话模板包装好，
做一次前向，只取**最后一个位置**的 logits，挑出 A、B、C… 这几个字母对应的 token，在它们之间做 softmax。

几个工程细节（初学者可以先跳过）：

1. **只算最后一个位置的 logits。** Gemma 的词表有 26.2 万个 token，如果对序列里每个位置都算完整的 logits，
   1,000 个 token 的序列就要输出 2.6 亿个数，显存会被撑爆。``logits_to_keep=1`` 让模型只算最后一个位置。

2. **左侧补齐（left padding）。** 一批里的序列长短不一，需要补齐。我们要读的是“每条序列的最后一个位置”，
   如果在右边补 pad，各条序列的最后一个真实 token 位置就不一样了；补在左边，所有序列的最后一个位置都对齐在末尾。

3. **字母 token 的两种写法。** 分词器里 "A" 和 " A"（前面带空格）往往是两个不同的 token，模型回答时可能用任意一种。
   我们把两种写法的打分合并（logsumexp，相当于把两者的概率相加），作为这个字母的总打分。

4. **思考模式。** Gemma 4 的对话模板在默认（关闭思考）时，会在模型回合开头放一个空的思考块，
   紧接着的下一个 token 就是正式回答，所以可以直接在这里读字母。

5. **材料太长时截短。** 材料按 token 数截到 ``max_state_tokens``（保留开头），避免个别超长样本拖慢整批。
"""

from __future__ import annotations

import sys

import torch

from ajev.lm.prompt import LETTERS, MAX_OPTIONS, build_user_message
from ajev.schema import Decision


class LMPredictor:
    def __init__(self, model_id: str, device: str | None = None, batch_tokens: int = 24000,
                 max_state_tokens: int = 3000, dtype: torch.dtype = torch.bfloat16,
                 temperatures: dict[str, float] | None = None) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(model_id, dtype).to(self.device).eval()
        self.batch_tokens = batch_tokens
        self.max_state_tokens = max_state_tokens
        self.temperatures = temperatures or {}
        # 每个字母可能对应的 token id（"A" 和 " A" 两种写法，只保留恰好是单个 token 的）。
        self.letter_ids: list[list[int]] = []
        for letter in LETTERS:
            ids = []
            for form in (letter, " " + letter):
                t = self.tok.encode(form, add_special_tokens=False)
                if len(t) == 1 and t[0] not in ids:
                    ids.append(t[0])
            if not ids:
                raise ValueError(f"letter {letter!r} is not a single token for {model_id}")
            self.letter_ids.append(ids)
        self.skipped = 0  # 选项超过 26 个、无法用字母表示的题数

    @staticmethod
    def _load_model(model_id: str, dtype: torch.dtype):
        """加载模型。Gemma 4 是图文统一模型，纯文本的 AutoModelForCausalLM 可能不认它，
        这时退回到图文模型类 AutoModelForImageTextToText（我们只喂文本，不影响结果）。"""
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        try:
            return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except (ValueError, KeyError):
            return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)

    def _prompt_ids(self, d: Decision) -> list[int]:
        """一道题 → 完整的输入 token id（材料先截短，再套对话模板，末尾就是模型开始回答的位置）。"""
        state_ids = self.tok.encode(d.state, add_special_tokens=False)
        state = d.state if len(state_ids) <= self.max_state_tokens else \
            self.tok.decode(state_ids[: self.max_state_tokens]) + " …[truncated]"
        messages = [{"role": "user", "content": build_user_message(d, state)}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        # 不同版本的 transformers 返回值不同：有的直接是 token 列表，有的是带 "input_ids" 的字典。
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)

    @torch.no_grad()
    def predict_logits(self, decisions: list[Decision]) -> list[list[float]]:
        """每道题在各选项字母上的打分（logits）。选项超过 26 个的题无法处理，返回全 0（等于均匀分布）。"""
        out: list[list[float]] = [[0.0] * len(d.options) for d in decisions]
        todo = []
        for i, d in enumerate(decisions):
            if len(d.options) > MAX_OPTIONS:
                self.skipped += 1
                continue
            todo.append((i, self._prompt_ids(d)))
        # 按长度排序后按 token 预算分批：长度相近的放一起，补齐浪费最少。
        todo.sort(key=lambda x: len(x[1]))
        batch: list[tuple[int, list[int]]] = []

        def flush() -> None:
            if not batch:
                return
            width = len(batch[-1][1])  # 已按长度升序，最后一条最长
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, (_, seq) in enumerate(batch):
                ids[r, width - len(seq):] = torch.tensor(seq)  # 左侧补齐
                mask[r, width - len(seq):] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            try:
                logits = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
            except TypeError:  # 旧版本不支持 logits_to_keep
                logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
            for r, (i, _) in enumerate(batch):
                k = len(decisions[i].options)
                # 每个字母：把 "A" 和 " A" 两种写法的打分用 logsumexp 合并。
                out[i] = [torch.logsumexp(logits[r, self.letter_ids[j]], dim=0).item() for j in range(k)]
            batch.clear()

        for item in todo:
            if batch and (len(batch) + 1) * len(item[1]) > self.batch_tokens:
                flush()
            batch.append(item)
        flush()
        if self.skipped:
            print(f"[lm] {self.skipped} decisions have more than {MAX_OPTIONS} options; scored as uniform",
                  file=sys.stderr)
        return out

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        """每道题的选项概率：字母打分按题型温度缩放后做 softmax（未校准时温度为 1）。"""
        probs = []
        for d, lg in zip(decisions, self.predict_logits(decisions)):
            t = torch.tensor(lg) / self.temperatures.get(d.type, 1.0)
            probs.append(torch.softmax(t, dim=-1).tolist())
        return probs
