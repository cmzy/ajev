"""用 vLLM 推理的预测器：提示词、标签、读出方式和校准温度都与 ``LMPredictor`` 相同，只是换了推理引擎。

    from ajev.lm.vllm_predictor import VLLMPredictor
    p = VLLMPredictor("google/gemma-4-12B-it", adapter="runs/gemma_lora5/best")
    probs = p.predict(decisions)

为什么用 vLLM：连续批处理（多个请求随到随算）、分页 KV cache、跨请求的自动前缀缓存、CUDA graph，
吞吐和单请求延迟都比 HuggingFace transformers 好得多（lora4 实测：600 道题 33 秒对 82 秒，单请求 39 ms）。

和 LMPredictor 保持一致的做法：
    1. 提示词由我们自己生成 token id（``prompt_ids``，同一个对话模板），直接交给 vLLM，不让它再分词；
    2. 只生成 1 个 token，并且只允许在本题的标签 token 里选（``allowed_token_ids``）；
       让 vLLM 返回这些 token 的 log 概率（屏蔽其他 token 后的 log-softmax，和全词表 logits 只差一个常数）；
    3. 每个标签两种写法（"A" / " A"）logsumexp 合并，除以题型温度后做 softmax。
LoRA 适配器**直接加载**（LoRARequest），不要合并后保存再加载：Gemma 4 合并后保存的权重重新加载会出错
（22% 的题答案改变，原因未查清），而直接加载与 transformers 结果一致（答案一致率 98%，平均概率差 0.014）。

不截断材料：提示词超过 ``max_model_len`` 的题抛出 ``ContextTooLong``，由调用方决定怎么处理。
"""

from __future__ import annotations

import math
import uuid

from ajev.lm.labels import option_labels
from ajev.lm.predictor import load_tokenizer, prompt_ids, read_lm_config
from ajev.lm.prompt import MAX_OPTIONS
from ajev.schema import Decision

NO_TRUNCATION = 10 ** 9


class ContextTooLong(ValueError):
    pass


def engine_kwargs(model_id: str, adapter: str | None, max_model_len: int, gpu_memory_utilization: float) -> dict:
    """LLM / AsyncEngineArgs 共用的参数。max_logprobs 要容纳 255 个标签 × 2 种写法。"""
    kw = dict(model=model_id, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True, max_logprobs=2 * MAX_OPTIONS + 2, logprobs_mode="processed_logprobs")
    if adapter:
        kw.update(enable_lora=True, max_lora_rank=64, max_loras=1)
    return kw


class LabelReader:
    """提示词 token、每题的采样参数、从 vLLM 输出读回选项概率——同步和异步两种引擎共用。"""

    def __init__(self, model_id: str, adapter: str | None, max_model_len: int,
                 temperatures: dict[str, float] | None = None) -> None:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.tok = load_tokenizer(model_id)
        self.max_model_len = max_model_len
        cfg = read_lm_config(adapter)
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        self.lora = LoRARequest("ajev", 1, adapter) if adapter else None
        _, ids, valid = option_labels(self.tok)
        self.forms = [[ids[i][j] for j in range(2) if valid[i][j]] for i in range(len(ids))]

    def prompt(self, d: Decision) -> list[int]:
        ids = prompt_ids(self.tok, d, NO_TRUNCATION)
        if len(ids) + 1 > self.max_model_len:
            raise ContextTooLong(f"prompt of {len(ids)} tokens exceeds the maximum context length {self.max_model_len}")
        return ids

    def params(self, d: Decision):
        if len(d.options) > len(self.forms):
            raise ValueError(f"{len(d.options)} options per choice > {len(self.forms)} labels")
        allowed = sorted({t for f in self.forms[: len(d.options)] for t in f})
        return self.SamplingParams(max_tokens=1, temperature=0, logprobs=len(allowed), allowed_token_ids=allowed)

    def probs(self, d: Decision, output) -> list[float]:
        lp = output.outputs[0].logprobs[0]  # {token_id: Logprob}，只含允许的标签 token
        scores = []
        for f in self.forms[: len(d.options)]:
            vals = [lp[t].logprob for t in f if t in lp] or [-1e4]
            m = max(vals)
            scores.append(m + math.log(sum(math.exp(v - m) for v in vals)))
        t = self.temperatures.get(d.type, 1.0)
        m = max(scores)
        e = [math.exp((s - m) / t) for s in scores]
        z = sum(e)
        return [x / z for x in e]


class VLLMPredictor:
    """离线批量推理（评测、校准用）。接口与 LMPredictor 相同：``predict(decisions) -> 概率列表``。"""

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88, temperatures: dict[str, float] | None = None) -> None:
        from vllm import LLM

        self.reader = LabelReader(model_id, adapter, max_model_len, temperatures)
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.llm = LLM(**engine_kwargs(model_id, adapter, max_model_len, gpu_memory_utilization))

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        from vllm.inputs import TokensPrompt

        r = self.reader
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=r.prompt(d)) for d in decisions],
                                 [r.params(d) for d in decisions], lora_request=r.lora, use_tqdm=False)
        return [r.probs(d, o) for d, o in zip(decisions, outs)]


class AsyncVLLMPredictor:
    """异步推理（HTTP 服务用）：同时到达的请求由 vLLM 自动合并成批次。"""

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.reader = LabelReader(model_id, adapter, max_model_len)
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs(model_id, adapter, max_model_len,
                                                                                 gpu_memory_utilization)))

    async def _one(self, d: Decision, ids: list[int]) -> list[float]:
        from vllm.inputs import TokensPrompt

        final = None
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=ids), self.reader.params(d),
                                              request_id=uuid.uuid4().hex, lora_request=self.reader.lora):
            final = out
        return self.reader.probs(d, final)

    async def predict(self, decisions: list[Decision]) -> tuple[list[list[float]], int]:
        """返回 (每题概率, 输入 token 总数)。一个请求里的各题同时提交，共享的材料开头由前缀缓存复用。"""
        import asyncio

        ids = [self.reader.prompt(d) for d in decisions]
        probs = await asyncio.gather(*(self._one(d, i) for d, i in zip(decisions, ids)))
        return list(probs), sum(len(i) for i in ids)
