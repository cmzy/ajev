"""第二批数据源转换器的单元测试（ajev/data/more_sources.py）。

每个测试手写一行“和真实数据格式一样”的原始数据，交给转换器，检查生成的 Decision：
题型、选项、target、id / group 是否符合预期。不需要联网，也不需要 torch。
"""

import json
import random

import pytest

from ajev.data import more_sources as ms
from ajev.data.sources import Ctx


def ctx(name, label_names=(), seed=0):
    """构造转换器需要的上下文（随机数生成器、数据源、类别名列表）。"""
    return Ctx(rng=random.Random(seed), source=ms.MORE_SOURCES[name], label_names=list(label_names))


def test_bev_all_three_types_and_soft_labels():
    """bev-decision：一行 3 个问题 → 3 道题，共享 group；noul 的 bool 答案、score 的整数答案、
    choice 的 label_probs 软标签都被正确转换。"""
    questions = {
        "ok": {"type": "noul", "instructions": "Does it hold?", "label": False},
        "intent": {"type": "choice", "instructions": "Intent?", "criteria": {"a": "A", "b": "B", "c": "C"},
                   "label": "b", "label_probs": {"a": 0.2, "b": 0.7, "c": 0.1}},
        "band": {"type": "score", "instructions": "Band?", "criteria": ["low", "mid", "high"], "label": 2,
                 "label_probs": [0.0, 0.25, 0.75]},  # 打分题的软标签是列表
    }
    ds = ms.conv_bev({"state": "s", "questions_json": json.dumps(questions), "domain": "x"}, 5, ctx("bev_skills"))
    by = {d.meta["question_id"]: d for d in ds}
    assert set(by) == {"ok", "intent", "band"} and len({d.group for d in ds}) == 1
    assert dict(zip(by["ok"].option_names, by["ok"].target)) == {"true": 0.0, "false": 1.0}
    assert by["intent"].target == pytest.approx([0.2, 0.7, 0.1])
    assert by["band"].target == pytest.approx([0.0, 0.25, 0.75])


def test_bev_drops_invalid_and_unknown_label():
    """答案不在选项里的题、超过 10 级的打分题会被丢掉，而不是让构建中断。"""
    questions = {
        "bad_label": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}, "label": "zzz"},
        "too_many": {"type": "score", "instructions": "?", "criteria": [str(k) for k in range(12)], "label": 1},
        "good": {"type": "noul", "instructions": "?", "label": True},
    }
    ds = ms.conv_bev({"state": "s", "questions_json": json.dumps(questions)}, 0, ctx("bev_default"))
    assert [d.meta["question_id"] for d in ds] == ["good"]


def test_ticket_three_questions():
    """客服工单：一张工单 → 队列（choice）、类型（choice）、优先级（score，高 = 第 2 级）。"""
    queues = ["Billing and Payments", "Technical Support"]
    row = {"subject": "Down", "body": "Portal offline", "queue": "Technical Support", "type": "Incident",
           "priority": "high"}
    ds = ms.conv_ticket(row, 3, ctx("support_tickets", queues))
    assert [d.id.split("/")[-1] for d in ds] == ["queue", "type", "priority"]
    q, t, p = ds
    assert q.option_names[q.gold_index] == "Technical Support"
    assert t.option_names[t.gold_index] == "Incident" and t.type == "choice"
    assert p.type == "score" and p.gold_index == 2
    for d in ds:
        d.validate()
    # 队列选项固定是 10 个团队队列，不受数据里收集到的话题标签影响
    assert q.option_names == list(ms.TICKET_QUEUES) and all(o.desc for o in q.options)


def test_ticket_topic_tag_queue_is_skipped():
    """queue 字段是话题标签（不是 10 个团队队列之一）的行直接跳过。"""
    row = {"subject": "x", "body": "y", "queue": "IT & Technology/Software Development", "type": "Incident",
           "priority": "high"}
    assert ms.conv_ticket(row, 0, ctx("support_tickets", ["IT & Technology/Software Development"])) is None


def test_helpsteer3_soft_target_from_individual_votes():
    """HelpSteer3：3 个标注者打 -1、-1、-2 → 第 2 级（-1）占 2/3，第 1 级（-2）占 1/3。"""
    row = {"context": [{"role": "user", "content": "hi"}], "response1": "a", "response2": "b",
           "overall_preference": -1, "language": "chinese",
           "individual_preference": [{"score": -1}, {"score": -1}, {"score": -2}]}
    d = ms.conv_helpsteer3(row, 0, ctx("helpsteer3"))
    d.validate()
    assert d.target == pytest.approx([0, 1 / 3, 2 / 3, 0, 0, 0, 0])
    assert d.lang == "zh" and len(d.options) == 7


def test_pku_safe_and_severity():
    """PKU-SafeRLHF：随机取一个回复，出“是否安全”和“危害程度”两道题，答案与所选回复对应。"""
    row = {"prompt": "p", "response_0": "r0", "response_1": "r1", "is_response_0_safe": True,
           "is_response_1_safe": False, "response_0_severity_level": 0, "response_1_severity_level": 3}
    c = ctx("pku_saferlhf")
    k = random.Random(0).choice([0, 1])  # 与转换器里的第一次随机抽取一致
    safe, sev = ms.conv_pku(row, 0, c)
    assert json.loads(safe.state)["response"] == f"r{k}"
    assert safe.option_names[safe.gold_index] == ("true" if k == 0 else "false")
    assert sev.gold_index == (0 if k == 0 else 3)


def test_when2call_choice():
    """When2Call：correct_answer=request_for_info → 4 选 1 的正确项；工具只保留名字、描述和必填参数。"""
    tool = json.dumps({"name": "get_weather", "description": "Weather", "parameters": {"required": ["city"]}})
    d = ms.conv_when2call({"question": "weather?", "correct_answer": "request_for_info", "tools": [tool]}, 0,
                          ctx("when2call"))
    d.validate()
    assert d.option_names[d.gold_index] == "request_for_info"
    assert json.loads(d.state)["tools"] == [{"name": "get_weather", "description": "Weather", "required": ["city"]}]


def test_wanli_and_chinese_toxicity():
    """WANLI 的字符串标签、ToxiCN / COLD 的中文 noul 都能正确转换。"""
    d = ms.conv_wanli({"premise": "p", "hypothesis": "h", "gold": "contradiction"}, 0, ctx("wanli"))
    d.validate()
    assert d.option_names[d.gold_index] in ("contradiction", "false")
    t = ms.conv_toxicn({"content": "你好", "toxic": 1}, 0, ctx("toxicn"))
    c = ms.conv_cold({"TEXT": "你好", "label": 0}, 0, ctx("cold"))
    assert t.lang == c.lang == "zh"
    assert t.option_names[t.gold_index] == "true" and c.option_names[c.gold_index] == "false"


def test_ultrafeedback_zh_score():
    """中文 UltraFeedback：唯一有分数的维度是 helpfulness=4 → 5 级打分题的第 3 级（从 0 数）。"""
    row = {"instruction": "列出三位唐代诗人", "completions": [
        {"response": "李白、杜甫、白居易", "annotations": {"helpfulness": {"Rating": "4"}, "honesty": {"Rating": "N/A"}}}]}
    d = ms.conv_uf_zh(row, 0, ctx("ultrafeedback_zh"))
    d.validate()
    assert d.gold_index == 3 and d.lang == "zh"
    assert d.instructions == ms.UF_ASPECTS["helpfulness"][0]


def test_clip_keeps_head_and_tail():
    """_clip 保留开头约 70% 和结尾约 30%，中间用“…”省略，总长度正好等于上限；不超长的文本原样返回。"""
    assert ms._clip("abcdefghijklmnopqrstuvwxyz", 10) == "abcdefg…yz"
    assert ms._clip("short", 10) == "short"


def test_helpsteer3_state_puts_key_parts_first():
    """HelpSteer3 的材料：用户最后一句话和两个回复放在最前面，早期对话放最后且只留 4 条，长回复被截短。"""
    context = [{"role": "user" if k % 2 == 0 else "assistant", "content": f"turn {k}"} for k in range(9)]
    row = {"context": context, "response1": "x" * 5000, "response2": "y", "overall_preference": 1,
           "language": "english", "individual_preference": []}
    d = ms.conv_helpsteer3(row, 0, ctx("helpsteer3"))
    s = json.loads(d.state)
    assert list(s) == ["last_user_turn", "response_1", "response_2", "earlier_conversation"]
    assert s["last_user_turn"] == "turn 8"
    assert len(s["response_1"]) == 1200 and s["response_2"] == "y"
    assert [m["content"] for m in s["earlier_conversation"]] == ["turn 4", "turn 5", "turn 6", "turn 7"]


def test_shuffled_holdout_split(monkeypatch):
    """"#train" / "#eval" 先用固定种子打乱再切分：两部分不重叠、合起来是全部数据，而且标签分布接近。

    模拟 When2Call 的情况：100 行数据按标签排好序（前 70 行是 a，后 30 行是 b）。
    如果按原顺序取后 15%，评测部分会全是 b；打乱后切分，评测部分应该两种标签都有。
    """
    from datasets import Dataset

    from ajev.data import sources

    data = Dataset.from_dict({"x": list(range(100)), "y": ["a"] * 70 + ["b"] * 30})
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: data)
    src = sources.Source("toy", "toy/path", None, "train#train", "train#eval", "en", lambda r, i, c: None,
                         holdout=0.15)
    tr, ev = src.load("train#train"), src.load("train#eval")
    assert len(ev) == 15 and len(tr) == 85
    assert set(tr["x"]) | set(ev["x"]) == set(range(100)) and not set(tr["x"]) & set(ev["x"])
    assert set(ev["y"]) == {"a", "b"}
    assert src.load("train#eval")["x"] == ev["x"]  # 固定种子：每次切分结果一样
