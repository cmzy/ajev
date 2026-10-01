"""AJev 推理服务：接口格式与 TypeSafe Jev 兼容的 HTTP 服务。

    python -m ajev.server --checkpoint runs/sft4/best --port 8000

    curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
      "state": {"ticket": "我的订单三天了还没发货"},
      "questions": {
        "category": {"type": "choice", "instructions": "这是什么问题？",
                     "criteria": {"delivery": "物流配送", "refund": "退款", "account": "账户"}},
        "urgent":   {"type": "noul", "instructions": "需要马上人工处理。"},
        "anger":    {"type": "score", "instructions": "客户有多生气？",
                     "criteria": ["平静", "有点不满", "很生气"]}
      }
    }'

在 AJev 流程中的位置：训练 → 校准 → 评测 → **部署（本模块）**。它只是把已有的几块拼起来：
``schema.decisions_from_jev``（Jev 请求 → Decision）、``EncoderPredictor``（Decision → 概率，
自动使用 checkpoint 里校准好的温度）、``schema.jev_answer``（概率 → Jev 响应）。

请求格式（与 Jev 相同）：
- ``state``：材料，可以是字符串、JSON 对象或数组；
- ``questions``：{问题名: 问题}，问题有三种：
  - ``noul``（是/否）：``instructions`` 写成一个陈述句，返回它成立的概率；
    可选 ``criteria: {"true": "...", "false": "..."}`` 说明两边的判断标准；
  - ``choice``（单选）：``criteria: {选项名: 选项描述}``，2–255 个选项；
  - ``score``（打分）：``criteria: [第 0 级描述, 第 1 级描述, ...]``，2–10 级。

响应格式（与 Jev 相同）::

    {"model": "...",
     "answers": {"category": {"choice": "delivery", "probabilities": {...}, "confidence": 0.82},
                 "urgent":   {"noul": 0.31},
                 "anger":    {"score": 0.7, "legend": [...], "probabilities": [...], "confidence": 0.4}},
     "usage": {"input_tokens": 312, "output_tokens": 0}}

``confidence`` = 1 − H(p)/ln K：1 表示很确定，0 表示完全拿不准。决策模型不生成文本，所以 output_tokens 永远是 0。

给初学者的几点说明：
- **FastAPI** 是一个 Python Web 框架：用 ``@app.post("/路径")`` 装饰一个函数，这个函数就成了一个 HTTP 接口，
  请求体里的 JSON 会自动解析成 Python 字典传进来，返回的字典会自动转成 JSON。
- **uvicorn** 是运行 FastAPI 应用的服务器程序，负责监听端口、接收请求。
- 一个请求里的所有问题会拼成一个 batch，**一次前向全部算完**，比逐个问题调用快得多。
- 模型同一时间只处理一个请求（用锁保护）：PyTorch 模型不适合多个线程同时调用，
  而这个模型单次推理只要几十毫秒，排队等待的开销很小。
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from typing import Any

from ajev.schema import decisions_from_jev, jev_answer

# 中日韩文字的 Unicode 范围，用来判断请求是不是中文（决定 noul 题默认的“是 / 否”描述用中文还是英文）。
_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def _detect_lang(state: Any, questions: dict) -> str:
    """请求里出现中日韩文字就按中文处理，否则按英文。只影响 noul 题没写 criteria 时的默认选项描述。"""
    text = str(state) + " ".join(str(q.get("instructions", "")) for q in questions.values())
    return "zh" if _CJK.search(text) else "en"


def create_app(checkpoint: str, device: str | None = None, batch_size: int = 32):
    """构造 FastAPI 应用并加载模型（模型只在启动时加载一次，之后所有请求共用）。

    单独写成函数（而不是在模块顶层直接创建 app），是为了让测试可以用任意 checkpoint 创建应用。
    """
    from fastapi import FastAPI, HTTPException

    from ajev.model.predictor import EncoderPredictor

    predictor = EncoderPredictor(checkpoint, device=device, batch_size=batch_size)
    model_name = os.path.basename(os.path.normpath(checkpoint))
    lock = threading.Lock()  # 同一时间只让一个请求使用模型
    app = FastAPI(title="AJev", description="Jev-compatible typed decision server")

    @app.get("/health")
    def health() -> dict:
        """健康检查：返回模型名、运行设备和各题型的校准温度，方便确认服务加载的是哪个模型。"""
        return {"status": "ok", "model": model_name, "device": str(predictor.device),
                "temperatures": predictor.temperatures}

    @app.post("/v1/systemone")
    def systemone(body: dict) -> dict:
        """Jev 兼容的决策接口。

        处理步骤：
            第 1 步：检查请求格式；
            第 2 步：把 Jev 请求展开成 Decision（每个问题一道题，共享同一份 state）；
            第 3 步：一次前向算出所有题的概率（自动使用校准温度）；
            第 4 步：把概率转换成 Jev 格式的答案，并统计输入 token 数。
        格式不对（缺字段、题型未知、选项数超出范围、选项太多放不下等）时返回 HTTP 400 和原因。
        """
        # 第 1 步：检查请求格式。
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "request needs 'state' and a non-empty 'questions' object")
        if len(questions) > 256:
            raise HTTPException(400, "at most 256 questions per request")
        # 第 2 步：展开成 Decision，并逐题校验（选项数、题型等）。
        try:
            decisions = decisions_from_jev(body["state"], questions, group="request",
                                           lang=_detect_lang(body["state"], questions))
            for d in decisions:
                d.validate()
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise HTTPException(400, f"invalid question: {e}") from e
        # 第 3 步：推理。
        t0 = time.perf_counter()
        try:
            with lock:
                probs = predictor.predict(decisions)
                input_tokens = sum(len(predictor.encoder.encode(d).input_ids) for d in decisions)
        except ValueError as e:  # 例如选项多到在 max_len 内放不下
            raise HTTPException(400, str(e)) from e
        # 第 4 步：组装 Jev 响应。
        answers = {d.meta["question_id"]: jev_answer(d, p) for d, p in zip(decisions, probs)}
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model directory, e.g. runs/sft4/best")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept requests from other machines")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(args.checkpoint, args.device, args.batch_size), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
