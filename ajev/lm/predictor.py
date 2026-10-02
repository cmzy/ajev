"""大语言模型（如 Gemma 4 12B）的推理器（零样本或加载 LoRA 适配器），实现与 mmBERT 相同的 ``Predictor`` 接口。

    from ajev.lm.predictor import LMPredictor
    probs = LMPredictor("google/gemma-4-12B-it").predict(decisions)                         # 零样本
    probs = LMPredictor("google/gemma-4-12B-it", adapter="runs/gemma_lora1/best").predict(decisions)  # LoRA

或者通过评测命令：

    python -m ajev.eval.evaluate --data data/external/jevbench_public.jsonl \\
        --predictor lm --lm-model google/gemma-4-12B-it [--lm-adapter runs/gemma_lora1/best]

原理（详见 ajev/lm/prompt.py）：把题目写成带字母选项的对话，用模型的对话模板包装好，
做一次前向，只取**最后一个位置**的 logits，挑出 A、B、C… 这几个字母对应的 token，在它们之间做 softmax。

本模块里的几个函数（``prompt_ids``、``left_pad``、``letter_logits`` 等）**训练和推理共用**
（ajev/lm/train.py 直接 import 它们），保证训练时算的和推理时读的是同一个东西。

几个工程细节（初学者可以先跳过）：

1. **只算最后一个位置的 logits。** Gemma 的词表有 26.2 万个 token，如果对序列里每个位置都算完整的 logits，
   1,000 个 token 的序列就要输出 2.6 亿个数，显存会被撑爆。``logits_to_keep=1`` 让模型只算最后一个位置。

2. **左侧补齐（left padding）。** 一批里的序列长短不一，需要补齐。我们要读的是“每条序列的最后一个位置”，
   如果在右边补 pad，各条序列的最后一个真实 token 位置就不一样了；补在左边，所有序列的最后一个位置都对齐在末尾。

3. **字母 token 的两种写法。** 分词器里 "A" 和 " A"（前面带空格）往往是两个不同的 token，模型回答时可能用任意一种。
   我们把两种写法的打分合并（logsumexp，相当于把两者的概率相加），作为这个字母的总打分。

4. **思考模式。** Gemma 4 的对话模板在默认（关闭思考）时，会在模型回合开头放一个空的思考块，
   紧接着的下一个 token 就是正式回答，所以可以直接在这里读字母。

5. **材料太长时截短。** 材料按 token 数截到 ``max_state_tokens``（保留开头）。注意**训练和推理的截断长度是两回事**：
   训练时为了省显存截得比较短（例如 1,500），只作记录存在适配器配置的 ``train_max_state_tokens`` 里；
   推理时默认放宽到 ``DEFAULT_INFER_STATE_TOKENS``（16,384，与训练上限一致），尽量让模型看到完整材料。
   我们吃过这个亏：最初推理沿用了训练的 1,500，长政策 / 多跳推理题的关键证据被截掉，
   JevBench 上材料超过 1,500 token 的 37 道题准确率只有 0.35，被误判成“LoRA 损害了推理能力”。
"""

from __future__ import annotations

import json
import os
import sys

import torch

from ajev.lm.prompt import LETTERS, MAX_OPTIONS, build_user_message
from ajev.schema import Decision

# LoRA 适配器目录里的 AJev 配置文件：基座模型名、训练时的材料截断长度、训练步数、校准温度。
LM_CONFIG = "ajev_lm_config.json"
# 推理时材料的默认截断长度（token）。Gemma 4 支持 256K 上下文；16K 与 gemma_lora2 的训练上限一致，足以覆盖我们用到的评测集。
DEFAULT_INFER_STATE_TOKENS = 16384


def load_base_model(model_id: str, dtype: torch.dtype = torch.bfloat16):
    """加载基座模型。Gemma 4 是图文统一模型，纯文本的 AutoModelForCausalLM 可能不认它，
    这时退回到图文模型类 AutoModelForImageTextToText（我们只喂文本，不影响结果）。"""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except (ValueError, KeyError):
        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    return tok


def letter_token_table(tok) -> tuple[torch.Tensor, torch.Tensor]:
    """每个字母可能对应的 token id 表：返回 ``ids`` [26, 2] 和 ``valid`` [26, 2]。

    每个字母最多两种写法（"A" 和 " A"），只保留恰好是单个 token 的写法；只有一种写法时，
    第二格用第一格填充，并在 ``valid`` 里标记为无效（合并打分时会被排除，不会被重复计算）。
    """
    ids, valid = [], []
    for letter in LETTERS:
        forms = []
        for form in (letter, " " + letter):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1 and t[0] not in forms:
                forms.append(t[0])
        if not forms:
            raise ValueError(f"letter {letter!r} is not a single token for this tokenizer")
        ids.append(forms + [forms[0]] * (2 - len(forms)))
        valid.append([True] * len(forms) + [False] * (2 - len(forms)))
    return torch.tensor(ids), torch.tensor(valid)


def prompt_ids(tok, d: Decision, max_state_tokens: int) -> list[int]:
    """一道题 → 完整的输入 token id（材料先截短，再套对话模板，末尾就是模型开始回答的位置）。"""
    state_ids = tok.encode(d.state, add_special_tokens=False)
    state = d.state if len(state_ids) <= max_state_tokens else \
        tok.decode(state_ids[:max_state_tokens]) + " …[truncated]"
    messages = [{"role": "user", "content": build_user_message(d, state)}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # 不同版本的 transformers 返回值不同：有的直接是 token 列表，有的是带 "input_ids" 的字典。
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把长短不一的序列在**左侧**补齐，返回 input_ids 和 attention_mask（真实 token 为 1）。

    例如 [[5, 6, 7], [8]]、pad_id=0 → input_ids [[5, 6, 7], [0, 0, 8]]，mask [[1, 1, 1], [0, 0, 1]]。
    """
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, width - len(s):] = torch.tensor(s)
        mask[r, width - len(s):] = 1
    return ids, mask


def letter_logits(model, ids: torch.Tensor, mask: torch.Tensor, table: torch.Tensor, valid: torch.Tensor,
                  k: int) -> torch.Tensor:
    """一批题在前 k 个字母上的打分，形状 [B, k]（float32）。训练时带梯度，推理时在 no_grad 下调用。

    步骤：
        第 1 步：前向，只取最后一个位置的完整词表 logits：[B, V]；
        第 2 步：按字母表取出每个字母两种写法的打分：[B, 26, 2]，无效的写法填 -inf；
        第 3 步：两种写法 logsumexp 合并 → [B, 26]，再取前 k 个字母。
    """
    try:
        out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
    except TypeError:  # 旧版本不支持 logits_to_keep
        out = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    table, valid = table.to(out.device), valid.to(out.device)
    per_form = out[:, table].masked_fill(~valid, float("-inf"))  # [B, 26, 2]
    return torch.logsumexp(per_form, dim=-1)[:, :k]


def read_lm_config(adapter: str | None) -> dict:
    p = os.path.join(adapter, LM_CONFIG) if adapter else ""
    if not adapter or not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


class LMPredictor:
    """零样本或 LoRA 的大模型推理器。

    Args:
        model_id: 基座模型（HF id 或本地目录），例如 "google/gemma-4-12B-it"。
        adapter: LoRA 适配器目录（训练产出）；None 表示零样本。适配器目录里的配置会提供
            材料截断长度和校准温度。
        temperatures: 每种题型的温度；None 时用适配器配置里的，传 {} 表示强制不用（拟合温度时这样用）。
    """

    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None,
                 batch_tokens: int = 24000, max_state_tokens: int | None = None,
                 dtype: torch.dtype = torch.bfloat16, temperatures: dict[str, float] | None = None,
                 model=None, tok=None) -> None:
        cfg = read_lm_config(adapter)
        self.tok = tok or load_tokenizer(model_id)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if model is None:
            model = load_base_model(model_id, dtype)
            if adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter)
            model = model.to(self.device)
        self.model = model.eval()
        self.batch_tokens = batch_tokens
        # 推理截断长度：显式传入的优先，否则用推理默认值；不再沿用训练时为省显存设的较短截断。
        self.max_state_tokens = max_state_tokens or DEFAULT_INFER_STATE_TOKENS
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        self.table, self.valid = letter_token_table(self.tok)
        self.skipped = 0  # 选项超过 26 个、无法用字母表示的题数

    @torch.no_grad()
    def predict_logits(self, decisions: list[Decision]) -> list[list[float]]:
        """每道题在各选项字母上的打分（logits）。选项超过 26 个的题无法处理，返回全 0（等于均匀分布）。"""
        was_training = self.model.training
        self.model.eval()
        out: list[list[float]] = [[0.0] * len(d.options) for d in decisions]
        todo = []
        for i, d in enumerate(decisions):
            if len(d.options) > MAX_OPTIONS:
                self.skipped += 1
                continue
            todo.append((i, prompt_ids(self.tok, d, self.max_state_tokens)))
        # 按长度排序后按 token 预算分批：长度相近的放一起，补齐浪费最少。
        todo.sort(key=lambda x: len(x[1]))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        batch: list[tuple[int, list[int]]] = []

        def flush() -> None:
            if not batch:
                return
            ids, mask = left_pad([s for _, s in batch], pad)
            k = max(len(decisions[i].options) for i, _ in batch)
            scores = letter_logits(self.model, ids.to(self.device), mask.to(self.device), self.table, self.valid, k)
            for r, (i, _) in enumerate(batch):
                out[i] = scores[r, : len(decisions[i].options)].tolist()
            batch.clear()

        for item in todo:
            if batch and (len(batch) + 1) * len(item[1]) > self.batch_tokens:
                flush()
            batch.append(item)
        flush()
        if was_training:
            self.model.train()
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
