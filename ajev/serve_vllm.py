"""AJev 的 vLLM 推理服务：Jev / Decision Index 兼容的 ``POST /v1/systemone`` 接口，支持并发。

    python -m ajev.serve_vllm --adapter runs/gemma_lora5/best --port 8000
    curl -s localhost:8000/v1/systemone -d '{"state": "...", "questions": {...}}'

排行榜工具包可以直接连上来（多个运行器同时连同一个服务，见 scripts/jdi_parallel.py）::

    python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=ajev

和 Mac 部署包（deploy/mac/server.py）的区别：
    1. 推理用 vLLM（只能在 NVIDIA GPU 上运行），不加锁：同时到达的请求由 vLLM 合并成批次一起算，
       一个请求里的各个问题也同时提交，共享的材料开头由前缀缓存复用；
    2. 不截断材料：提示词超过 --max-model-len 时返回 HTTP 422，内容含 "maximum context length"
       （排行榜按规则记为 unsupported），绝不截短；
    3. 每个答案带 "type" 字段（排行榜校验要求），概率不做四舍五入。

响应格式::

    {"model": "ajev", "answers": {"category": {"type": "choice", "choice": "delivery", "probabilities": {...}},
                                  "urgent": {"type": "noul", "noul": 0.31},
                                  "anger": {"type": "score", "score": 1.2, "legend": [...], "probabilities": [...]}},
     "usage": {"input_tokens": 812, "output_tokens": 0}, "latency_ms": 41.3}
"""

from __future__ import annotations

import argparse
import time

from ajev.jev_request import detect_lang, plain_questions
from ajev.lm.vllm_predictor import AsyncVLLMPredictor, ContextTooLong
from ajev.schema import NOUL_TRUE, decisions_from_jev, jev_answer

MAX_QUESTIONS = 512


def answer(d, probs: list[float]) -> dict:
    if d.type == "noul":
        return {"type": "noul", "noul": dict(zip(d.option_names, probs))[NOUL_TRUE]}
    if d.type == "choice":
        pmap = dict(zip(d.option_names, probs))
        return {"type": "choice", "choice": max(pmap, key=pmap.get), "probabilities": pmap}
    return {"type": "score", **jev_answer(d, probs)}


def create_app(predictor: AsyncVLLMPredictor, model_name: str):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="AJev (vLLM)")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name, "temperatures": predictor.temperatures,
                "max_model_len": predictor.reader.max_model_len}

    @app.post("/v1/systemone")
    async def systemone(body: dict) -> dict:
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "request needs 'state' and a non-empty 'questions' object")
        if len(questions) > MAX_QUESTIONS:
            raise HTTPException(422, f"too many questions: {len(questions)} > {MAX_QUESTIONS}")
        try:
            qs = plain_questions(questions)
            decisions = decisions_from_jev(body["state"], qs, group="request", lang=detect_lang(body["state"], qs))
            for d in decisions:
                d.validate()
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            msg = str(e)
            raise HTTPException(422 if "options" in msg else 400, f"invalid question: {msg}") from e
        t = time.perf_counter()
        try:
            probs, input_tokens = await predictor.predict(decisions)
        except ContextTooLong as e:
            raise HTTPException(422, str(e)) from e
        except ValueError as e:  # 选项多于可用标签
            raise HTTPException(422, str(e)) from e
        return {"model": model_name, "answers": {d.meta["question_id"]: answer(d, p) for d, p in zip(decisions, probs)},
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t) * 1000, 1)}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="google/gemma-4-12B-it")
    ap.add_argument("--adapter", help="LoRA adapter directory (loaded directly, never merged)")
    ap.add_argument("--name", default="ajev", help="model name reported in responses")
    ap.add_argument("--max-model-len", type=int, default=131072, help="longer prompts are refused, not truncated")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    predictor = AsyncVLLMPredictor(a.base_model, a.adapter, a.max_model_len, a.gpu_memory_utilization)
    uvicorn.run(create_app(predictor, a.name), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
