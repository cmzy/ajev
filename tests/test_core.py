"""核心模块的单元测试：数据结构（schema）、评测指标（metrics）、基线预测器、数据转换器（sources）。

================================================================
什么是单元测试，什么是 pytest
================================================================
单元测试（unit test）是一小段自动运行的检查代码：用一个已知答案的小例子调用某个函数，
再检查结果是否符合预期。好处是：以后修改代码时，只要一条命令就能确认原有功能没被改坏。

pytest 是 Python 最常用的测试框架，规则很简单:
    * 文件名以 ``test_`` 开头（比如本文件 test_core.py）；
    * 里面以 ``test_`` 开头的函数，每个就是一个测试用例；
    * 在函数里用 ``assert 条件`` 断言：条件成立则通过，不成立则测试失败并显示原因。

运行方法（在仓库根目录）::

    uv run pytest -q           # -q 表示简洁输出，每个通过的测试显示为一个点 “.”

本文件的测试不依赖 torch，在本地基础环境下即可运行；模型相关的测试见 ``tests/test_model.py``。

几个常用的 pytest 工具:
    * ``pytest.approx(x)``：浮点数“约等于”比较。因为 0.1 + 0.2 在计算机里等于 0.30000000000000004，
      不能直接用 == 比较小数。
    * ``with pytest.raises(ValueError):``：断言这个代码块“一定会”抛出 ValueError，没抛出就算失败。
    * ``tmp_path``：测试函数的参数写上它，pytest 会自动提供一个临时目录，测试结束后自动清理。
"""

import json
import random

import pytest

from ajev import metrics
from ajev.data import sources
from ajev.predictors import PriorPredictor, UniformPredictor
from ajev.schema import Decision, Option, decisions_from_jev, jev_answer, jev_confidence, read_jsonl, write_jsonl

# 一个覆盖三种题型的 Jev 请求示例:
#   refund  —— noul（是/否题），没给 criteria，会使用默认的“是/否”说明；
#   topic   —— choice（单选题），两个选项各有说明；
#   urgency —— score（打分题），三个等级 low / mid / high。
JEV_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "The customer is asking for money back."},
    "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"billing": "Charges", "delivery": "Shipping"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
}


def make(target, type_="choice", names=("a", "b", "c"), **meta):
    """测试辅助函数：快速构造一道题。

    只需指定 target、题型和选项名，其余字段用固定的占位值。
    ``**meta`` 会把多余的关键字参数收集成一个字典，例如 make([1,0,0], gold_label="a")
    得到 meta = {"gold_label": "a"}。
    """
    return Decision(id="x", source="s", type=type_, state="st", instructions="q",
                    options=[Option(n) for n in names], target=list(target), meta=meta)


def test_jev_roundtrip():
    """Jev 请求 → Decision → Jev 响应 的完整往返。

    检查点:
        1. 拆出来的每道题都能通过 validate；
        2. noul 题的选项是 [true, false]；choice 题的 target 来自 gold；score 题的选项名是 "0","1","2"；
        3. 三种题型的响应格式正确。
    """
    gold = {"topic": {"probabilities": {"billing": 0.2, "delivery": 0.8}}}
    ds = decisions_from_jev({"msg": "where is my parcel"}, JEV_QUESTIONS, gold=gold)
    for d in ds:
        d.validate()
    by_q = {d.meta["question_id"]: d for d in ds}  # 按问题名建立查找表，方便下面逐个检查
    assert by_q["refund"].option_names == ["true", "false"]
    assert by_q["topic"].target == pytest.approx([0.2, 0.8])
    assert by_q["urgency"].option_names == ["0", "1", "2"]
    assert jev_answer(by_q["refund"], [0.9, 0.1]) == {"noul": 0.9}
    assert jev_answer(by_q["topic"], [0.2, 0.8])["choice"] == "delivery"
    # score = 0×0 + 1×0.5 + 2×0.5 = 1.5（概率加权的期望等级）。
    assert jev_answer(by_q["urgency"], [0.0, 0.5, 0.5])["score"] == pytest.approx(1.5)


def test_validate_rejects_bad_targets():
    """不合法的题必须被 validate 拦下来。

    情况 1：target 加起来不等于 1（0.5 + 0.2 + 0.2 = 0.9）；
    情况 2：target 只有 2 个数，但 score 题有 3 个等级。
    """
    with pytest.raises(ValueError):
        make([0.5, 0.2, 0.2]).validate()
    with pytest.raises(ValueError):
        make([0.5, 0.5], type_="score", names=("0", "1", "2")).validate()


def test_shuffle_keeps_name_target_pairs_and_score_order():
    """打乱选项后，“选项名 ↔ 概率”的对应关系必须保持不变；score 题的等级顺序永远不打乱。"""
    d = make([0.7, 0.2, 0.1])
    s = d.shuffled(random.Random(1))
    # dict(zip(名字, 概率)) 得到 {"a": 0.7, "b": 0.2, "c": 0.1}；字典比较与顺序无关，正好检验对应关系。
    assert dict(zip(s.option_names, s.target)) == dict(zip(d.option_names, d.target))
    sc = make([0.7, 0.2, 0.1], type_="score", names=("0", "1", "2"))
    assert sc.shuffled(random.Random(1)).option_names == ["0", "1", "2"]


def test_jsonl_roundtrip(tmp_path):
    """写入 JSONL 再读回来，得到的 Decision 必须和原来完全相等（包括 meta）。

    tmp_path 是 pytest 自动提供的临时目录；``tmp_path / "x.jsonl"`` 用 “/” 拼接路径。
    """
    d = make([0.7, 0.2, 0.1], gold_label="a")
    path = str(tmp_path / "x.jsonl")
    write_jsonl(path, [d])
    (back,) = read_jsonl(path)  # 解包：断言读回的列表恰好有 1 个元素，并把它赋给 back
    assert back == d  # dataclass 自动生成的 __eq__ 会逐个字段比较


def test_confidence_bounds():
    """Jev confidence 的两个极端：完全确定时为 1，均匀分布（完全拿不准）时为 0。"""
    assert jev_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert jev_confidence([0.25] * 4) == pytest.approx(0.0)


def test_metrics_perfect_and_uniform():
    """两种极端预测下的指标。

    完美预测：准确率 1，Brier 和 ECE 都是 0。
    均匀预测：所有选项概率相同时，argmax 选第一个选项 → 第 1 题对、第 2 题错，准确率 0.5；
              三选一的随机水平是 1/3，扣除随机猜测后的准确率 = (0.5 - 1/3) / (1 - 1/3) = 0.25。
    """
    ds = [make([1, 0, 0]), make([0, 1, 0])]
    perfect = metrics.compute(ds, [[1, 0, 0], [0, 1, 0]])
    assert perfect["accuracy"] == 1 and perfect["brier"] == pytest.approx(0) and perfect["ece"] == pytest.approx(0)
    # 均匀分布时并列最大值取第一个选项：准确率 0.5，随机水平 1/3，
    # 扣除随机猜测后的准确率 = (0.5 - 1/3) / (2/3) = 0.25。
    uni = metrics.compute(ds, UniformPredictor().predict(ds))
    assert uni["accuracy"] == 0.5 and uni["chance_acc"] == pytest.approx(0.25)


def test_gold_label_meta_overrides_argmax():
    """target 出现并列（0.5 / 0.5）时，以 meta 里的 gold_label 为准判断对错。

    这里 gold_label 是 "b"，模型预测 b 的概率最大（0.8），所以算答对。
    如果按 argmax(target) 判断，并列时会取第一个 "a"，就会被误判为答错。
    """
    d = make([0.5, 0.5, 0.0], gold_label="b")
    assert metrics.compute([d], [[0.1, 0.8, 0.1]])["accuracy"] == 1


def test_ece_overconfident():
    """过度自信的例子：两道题都说有 90% 把握，但只对了一半。

    两道题落在同一个区间，平均置信度 0.9，实际准确率 0.5 → ECE = |0.9 - 0.5| = 0.4。
    """
    assert metrics.ece([0.9, 0.9], [True, False]) == pytest.approx(0.4)


def test_flip_rate():
    """选项打乱翻转率：有位置偏置的预测器会“翻转”，按内容判断的预测器不会。"""
    d = make([1, 0, 0])
    # s 是把 d 的选项倒序后的版本：选项 a 从第 1 位移到了第 3 位，target 也跟着移动。
    # {**d.__dict__, ...}：复制 d 的所有字段，再覆盖 options 和 target。
    s = Decision(**{**d.__dict__, "options": [Option("c"), Option("b"), Option("a")], "target": [0, 0, 1]})
    # 有位置偏置的预测器：永远选第一个位置 → 打乱前选 a、打乱后选 c，答案变了，算一次翻转。
    assert metrics.flip_rate([[1, 0, 0]], [d], [s], [[1, 0, 0]]) == 1.0
    # 打乱后仍然选中 a（它现在在第 3 位）→ 答案没变，不算翻转。
    assert metrics.flip_rate([[1, 0, 0]], [d], [s], [[0, 0, 1]]) == 0.0


def test_prior_predictor():
    """先验频率基线：训练集中同一道题的答案是 a、a、b → 预测 [2/3, 1/3, 0]（不加平滑）。"""
    train = [make([1, 0, 0]), make([1, 0, 0]), make([0, 1, 0])]
    (p,) = PriorPredictor(train, smoothing=0).predict([make([0, 0, 1])])
    assert p == pytest.approx([2 / 3, 1 / 3, 0])


def _ctx(src_name, label_names, seed=0):
    """测试辅助函数：为数据转换器构造一个最小的运行上下文。

    转换器正常运行时需要先下载数据集；测试时我们直接手写一行假数据，
    并提供标签列表，这样不联网也能测试转换逻辑。
    """
    return sources.Ctx(rng=random.Random(seed), source=sources.SOURCES[src_name], label_names=label_names)


def test_intent_candidates_contain_gold():
    """意图识别数据（MASSIVE 中文）：随机抽取的候选意图里必须包含正确答案，且候选数在 4～20 之间。

    背景：MASSIVE 有 60 个意图，转换器每道题随机抽一部分作为选项，让模型适应不同的选项数量。
    这里用 50 个不同的随机种子反复测试，覆盖各种抽取结果。
    """
    labels = [f"intent_{i}" for i in range(60)]
    conv = sources.SOURCES["massive_zh"].convert
    for i in range(50):
        d = conv({"text": "星期五早上九点叫醒我", "label": "intent_7"}, i, _ctx("massive_zh", labels, i))
        d.validate()
        assert d.option_names[d.gold_index] == "intent_7"
        assert 4 <= len(d.options) <= 20


def test_stsb_soft_target():
    """STS-B 的连续相似度分数转成软标签。

    sentence-transformers 版本的分数在 0～1 之间，乘以 5 还原成原始的 0～5 分：0.5 → 2.5 分。
    2.5 恰好在第 2 级和第 3 级中间，所以概率平分：[0, 0, 0.5, 0.5, 0, 0]。
    同时检查 state 是由两个句子组成的 JSON。
    """
    d = sources.conv_stsb({"sentence1": "a", "sentence2": "b", "score": 0.5}, 0, _ctx("stsb", []))
    d.validate()
    assert d.target == pytest.approx([0, 0, 0.5, 0.5, 0, 0])
    assert json.loads(d.state) == {"a": "a", "b": "b"}


def test_tnews_chinese_options():
    """TNEWS 新闻分类：原始数据的类别是数字代码，转换器要把它映射成中文类别名。

    label = 3 对应代码列表里的第 4 个 "103"，即“体育”；同时语言应标记为 zh。
    """
    names = ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116"]
    d = sources.conv_tnews({"sentence": "国足比赛", "label": 3}, 0, _ctx("tnews", names))
    d.validate()
    assert d.option_names[d.gold_index] == "体育" and d.lang == "zh"


def test_nli_skips_unlabeled():
    """CLUE 等数据集中 label = -1 表示“没有标注”（通常出现在测试集），转换器应返回 None 跳过这一行。"""
    conv = sources.SOURCES["ocnli"].convert
    assert conv({"sentence1": "a", "sentence2": "b", "label": -1}, 0,
                _ctx("ocnli", ["neutral", "entailment", "contradiction"])) is None
