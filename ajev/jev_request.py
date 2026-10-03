"""Jev 格式请求的预处理（HTTP 服务和排行榜引擎共用）。

Jev / Decision Index 允许 instructions 和选项说明是 JSON 对象，我们的提示词只收字符串；
是非题的默认选项说明分中英文，按材料和问题里的中文字符比例决定。
"""

from __future__ import annotations

import json
import re

CJK = re.compile(r"[一-鿿]")


def text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def detect_lang(state, questions: dict) -> str:
    """材料和问题里中文字符占比超过 10% 就按中文题处理。"""
    s = text(state) + "".join(text(q.get("instructions", "")) for q in questions.values())
    return "zh" if s and len(CJK.findall(s)) / len(s) > 0.1 else "en"


def plain_questions(questions: dict) -> dict:
    """instructions / 选项说明统一转成字符串；choice 题缺说明的选项用空字符串。"""
    out = {}
    for k, q in questions.items():
        q = dict(q)
        q["instructions"] = text(q.get("instructions", ""))
        if q["type"] == "choice":
            q["criteria"] = {name: "" if desc is None else text(desc) for name, desc in q["criteria"].items()}
        elif q.get("criteria") and isinstance(q["criteria"], dict):
            q["criteria"] = {name: text(desc) for name, desc in q["criteria"].items()}
        out[k] = q
    return out
