import json
import random

import pytest

from ajev import metrics
from ajev.data import sources
from ajev.predictors import PriorPredictor, UniformPredictor
from ajev.schema import Decision, Option, decisions_from_jev, jev_answer, jev_confidence, read_jsonl, write_jsonl

JEV_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "The customer is asking for money back."},
    "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"billing": "Charges", "delivery": "Shipping"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
}


def make(target, type_="choice", names=("a", "b", "c"), **meta):
    return Decision(id="x", source="s", type=type_, state="st", instructions="q",
                    options=[Option(n) for n in names], target=list(target), meta=meta)


def test_jev_roundtrip():
    gold = {"topic": {"probabilities": {"billing": 0.2, "delivery": 0.8}}}
    ds = decisions_from_jev({"msg": "where is my parcel"}, JEV_QUESTIONS, gold=gold)
    for d in ds:
        d.validate()
    by_q = {d.meta["question_id"]: d for d in ds}
    assert by_q["refund"].option_names == ["true", "false"]
    assert by_q["topic"].target == pytest.approx([0.2, 0.8])
    assert by_q["urgency"].option_names == ["0", "1", "2"]
    assert jev_answer(by_q["refund"], [0.9, 0.1]) == {"noul": 0.9}
    assert jev_answer(by_q["topic"], [0.2, 0.8])["choice"] == "delivery"
    assert jev_answer(by_q["urgency"], [0.0, 0.5, 0.5])["score"] == pytest.approx(1.5)


def test_validate_rejects_bad_targets():
    with pytest.raises(ValueError):
        make([0.5, 0.2, 0.2]).validate()
    with pytest.raises(ValueError):
        make([0.5, 0.5], type_="score", names=("0", "1", "2")).validate()


def test_shuffle_keeps_name_target_pairs_and_score_order():
    d = make([0.7, 0.2, 0.1])
    s = d.shuffled(random.Random(1))
    assert dict(zip(s.option_names, s.target)) == dict(zip(d.option_names, d.target))
    sc = make([0.7, 0.2, 0.1], type_="score", names=("0", "1", "2"))
    assert sc.shuffled(random.Random(1)).option_names == ["0", "1", "2"]


def test_jsonl_roundtrip(tmp_path):
    d = make([0.7, 0.2, 0.1], gold_label="a")
    path = str(tmp_path / "x.jsonl")
    write_jsonl(path, [d])
    (back,) = read_jsonl(path)
    assert back == d


def test_confidence_bounds():
    assert jev_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert jev_confidence([0.25] * 4) == pytest.approx(0.0)


def test_metrics_perfect_and_uniform():
    ds = [make([1, 0, 0]), make([0, 1, 0])]
    perfect = metrics.compute(ds, [[1, 0, 0], [0, 1, 0]])
    assert perfect["accuracy"] == 1 and perfect["brier"] == pytest.approx(0) and perfect["ece"] == pytest.approx(0)
    # Uniform ties break to the first option: acc 0.5 vs chance 1/3 -> (0.5 - 1/3) / (2/3) = 0.25.
    uni = metrics.compute(ds, UniformPredictor().predict(ds))
    assert uni["accuracy"] == 0.5 and uni["chance_acc"] == pytest.approx(0.25)


def test_gold_label_meta_overrides_argmax():
    d = make([0.5, 0.5, 0.0], gold_label="b")
    assert metrics.compute([d], [[0.1, 0.8, 0.1]])["accuracy"] == 1


def test_ece_overconfident():
    assert metrics.ece([0.9, 0.9], [True, False]) == pytest.approx(0.4)


def test_flip_rate():
    d = make([1, 0, 0])
    s = Decision(**{**d.__dict__, "options": [Option("c"), Option("b"), Option("a")], "target": [0, 0, 1]})
    # Position-biased predictor: always picks the first option -> flips.
    assert metrics.flip_rate([[1, 0, 0]], [d], [s], [[1, 0, 0]]) == 1.0
    assert metrics.flip_rate([[1, 0, 0]], [d], [s], [[0, 0, 1]]) == 0.0


def test_prior_predictor():
    train = [make([1, 0, 0]), make([1, 0, 0]), make([0, 1, 0])]
    (p,) = PriorPredictor(train, smoothing=0).predict([make([0, 0, 1])])
    assert p == pytest.approx([2 / 3, 1 / 3, 0])


def _ctx(src_name, label_names, seed=0):
    return sources.Ctx(rng=random.Random(seed), source=sources.SOURCES[src_name], label_names=label_names)


def test_intent_candidates_contain_gold():
    labels = [f"intent_{i}" for i in range(60)]
    conv = sources.SOURCES["massive_zh"].convert
    for i in range(50):
        d = conv({"text": "星期五早上九点叫醒我", "label": "intent_7"}, i, _ctx("massive_zh", labels, i))
        d.validate()
        assert d.option_names[d.gold_index] == "intent_7"
        assert 4 <= len(d.options) <= 20


def test_stsb_soft_target():
    d = sources.conv_stsb({"sentence1": "a", "sentence2": "b", "score": 0.5}, 0, _ctx("stsb", []))
    d.validate()
    assert d.target == pytest.approx([0, 0, 0.5, 0.5, 0, 0])
    assert json.loads(d.state) == {"a": "a", "b": "b"}


def test_tnews_chinese_options():
    names = ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116"]
    d = sources.conv_tnews({"sentence": "国足比赛", "label": 3}, 0, _ctx("tnews", names))
    d.validate()
    assert d.option_names[d.gold_index] == "体育" and d.lang == "zh"


def test_nli_skips_unlabeled():
    conv = sources.SOURCES["ocnli"].convert
    assert conv({"sentence1": "a", "sentence2": "b", "label": -1}, 0,
                _ctx("ocnli", ["neutral", "entailment", "contradiction"])) is None
