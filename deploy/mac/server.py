"""AJev（Gemma 4 12B + LoRA）在 Mac 上的推理服务：Jev 兼容接口 + 网页 Playground。

    .venv/bin/python server.py                       # 默认：models/ajev-lora2，http://127.0.0.1:8000
    .venv/bin/python server.py --port 9000 --host 0.0.0.0

启动后：
    浏览器打开 http://127.0.0.1:8000/          → Playground（填写材料和问题，查看各选项概率）
    GET  /health                              → 模型、设备、校准温度、加载耗时
    POST /v1/systemone                        → Jev 兼容的决策接口（格式见下）

请求格式（与 Jev 相同）::

    {"state": "材料，字符串或 JSON 对象",
     "questions": {
        "category": {"type": "choice", "instructions": "这是什么问题？",
                     "criteria": {"delivery": "物流配送", "refund": "退款"}},
        "urgent":   {"type": "noul", "instructions": "需要马上人工处理。"},
        "anger":    {"type": "score", "instructions": "客户有多生气？", "criteria": ["平静", "不满", "很生气"]}}}

响应格式::

    {"model": "ajev-lora2", "answers": {"category": {"choice": "delivery", "probabilities": {...}, "confidence": 0.8},
                                       "urgent": {"noul": 0.31}, "anger": {"score": 1.2, ...}},
     "usage": {"input_tokens": 812, "output_tokens": 0}, "latency_ms": 2310.5}

工作原理（和训练、评测用的是同一套代码，见 ajev/lm/predictor.py）：
    每个问题写成“材料 + 问题 + 字母选项”的提示词，模型做一次前向，只读“下一个 token 是哪个字母”的打分，
    按题型除以校准温度后做 softmax，得到各选项的概率。不生成文字，所以 output_tokens 永远是 0。

为什么一次只处理一个请求（加锁）：Mac 上只有一块 GPU，几个请求同时算不会更快，反而可能爆内存。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # 让 Python 找到同目录下打包进来的 ajev 代码

_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def detect_lang(state: Any, questions: dict) -> str:
    """请求里出现中日韩文字就按中文处理（只影响 noul 题没写 criteria 时的默认“是/否”描述）。"""
    text = str(state) + " ".join(str(q.get("instructions", "")) for q in questions.values())
    return "zh" if _CJK.search(text) else "en"


def create_app(base_model: str, adapter: str, device: str | None, max_state_tokens: int, batch_tokens: int,
               merge: bool):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    from ajev.lm.predictor import LMPredictor, prompt_ids
    from ajev.schema import decisions_from_jev, jev_answer

    import torch

    from ajev.lm.predictor import default_device

    device = device or default_device()
    if device == "mps":
        # Apple 芯片 + macOS 14 以上的 MPS 支持 bf16；更老的系统或 Intel Mac 的独立显卡不支持，提前给出明确提示。
        try:
            torch.ones(1, dtype=torch.bfloat16, device="mps")
        except (TypeError, RuntimeError):
            raise SystemExit("这台 Mac 的 GPU（MPS）不支持 bf16：需要 Apple 芯片（M1–M4）和 macOS 14 以上。"
                             "可以用 --device cpu 运行，但会非常慢。")
    t0 = time.perf_counter()
    print(f"[ajev] loading {base_model} + {adapter} on {device} ...", flush=True)
    predictor = LMPredictor(base_model, adapter=adapter, device=device, max_state_tokens=max_state_tokens,
                            batch_tokens=batch_tokens, merge=merge)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[ajev] ready on {predictor.device} in {load_s}s", flush=True)
    model_name = os.path.basename(os.path.normpath(adapter))
    lock = threading.Lock()
    app = FastAPI(title="AJev", description="Jev-compatible typed decision server (Gemma 4 12B + LoRA)")

    @app.get("/")
    def playground():
        return FileResponse(os.path.join(HERE, "playground.html"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name, "base_model": base_model, "device": str(predictor.device),
                "temperatures": predictor.temperatures, "max_state_tokens": predictor.max_state_tokens,
                "merged_lora": merge, "load_seconds": load_s}

    @app.post("/v1/systemone")
    def systemone(body: dict) -> dict:
        # 第 1 步：检查请求格式。
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "request needs 'state' and a non-empty 'questions' object")
        if len(questions) > 64:
            raise HTTPException(400, "at most 64 questions per request")
        # 第 2 步：Jev 请求 → 每个问题一道 Decision（共享同一份材料）。
        try:
            decisions = decisions_from_jev(body["state"], questions, group="request",
                                           lang=detect_lang(body["state"], questions))
            for d in decisions:
                d.validate()
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise HTTPException(400, f"invalid question: {e}") from e
        # 第 3 步：推理（自动使用适配器里校准好的温度）。
        t1 = time.perf_counter()
        with lock:
            probs = predictor.predict(decisions)
            input_tokens = sum(len(prompt_ids(predictor.tok, d, predictor.max_state_tokens)) for d in decisions)
        # 第 4 步：概率 → Jev 格式的答案。
        answers = {d.meta["question_id"]: jev_answer(d, p) for d, p in zip(decisions, probs)}
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t1) * 1000, 1)}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=os.environ.get("AJEV_BASE_MODEL", "google/gemma-4-12B-it"),
                    help="HF id or local directory of the base model")
    ap.add_argument("--adapter", default=os.path.join(HERE, "models", "ajev-lora2"), help="LoRA adapter directory")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
    ap.add_argument("--max-state-tokens", type=int, default=16384, help="truncate longer states (tokens)")
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="padded tokens per forward pass; lower it if memory runs out")
    ap.add_argument("--no-merge", action="store_true", help="keep LoRA separate instead of merging (slower)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept requests from other machines")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    app = create_app(a.base_model, a.adapter, a.device, a.max_state_tokens, a.batch_tokens, merge=not a.no_merge)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
