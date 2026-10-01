"""外部评测转换（ajev/eval/external.py）的测试：标准答案格式统一、重合检查。"""

import json

from ajev.eval import external as ex
from ajev.schema import Decision, Option


def test_gold_probs_formats():
    """各评测集答案写法不同：noul 有 True/False 和 "yes"/"no"，score 有 int 和字符串，都统一成选项名。"""
    assert ex._gold_probs("noul", True) == {"true": 1.0}
    assert ex._gold_probs("noul", "no") == {"false": 1.0}
    assert ex._gold_probs("noul", "Yes") == {"true": 1.0}
    assert ex._gold_probs("score", 2) == {"2": 1.0} == ex._gold_probs("score", "2")
    assert ex._gold_probs("choice", "refund") == {"refund": 1.0}


def test_fragments_ignore_json_shape_and_short_text():
    """重合检查只看文字内容：JSON 字段名不同不影响；大小写、空白规范化；短于 40 字符的片段忽略。"""
    long = "The quick brown fox jumps over the lazy dog near the river bank."
    a = json.dumps({"a": long, "b": "short"})
    b = json.dumps({"sentence1": "  " + long.upper() + "  "})
    assert ex._fragments(a) == ex._fragments(b) == {long.lower()}
    assert ex._fragments("too short") == set()


def test_mark_seen():
    seen = ex._fragments("x" * 50)
    d = Decision(id="d", source="s", type="noul", state=json.dumps({"k": "X" * 50}), instructions="q",
                 options=[Option("true"), Option("false")], target=[1.0, 0.0])
    assert ex.mark_seen([d], seen) == {"s": 1} and d.meta["seen_in_train"]
