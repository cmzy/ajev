"""推理服务（ajev/server.py）的测试：在临时目录现场造一个很小的模型和分词器，用 FastAPI 的测试客户端发请求。

不需要下载真实模型，也不需要启动真正的服务器。需要 torch、transformers、fastapi、httpx，缺任何一个就整体跳过。
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """造一个 2 层、隐藏维度 32 的迷你 ModernBERT + 词级分词器，保存成 checkpoint，再用它创建服务。"""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    from ajev.model.encoder import DecisionModel
    from ajev.server import create_app

    path = str(tmp_path_factory.mktemp("ckpt"))
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    words = ["yes/no:", "choice:", "score:", "true", "false", "a", "b", "c", "0", "1", "2", "order", "late", "?", ":"]
    raw = Tokenizer(models.WordLevel({w: i for i, w in enumerate(specials + words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                  unk_token="<unk>", mask_token="<mask>", cls_token="<bos>", sep_token="<eos>")
    cfg = ModernBertConfig(vocab_size=len(specials) + len(words), hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=64, local_attention=16, pad_token_id=0)
    DecisionModel(ModernBertModel(cfg)).save(path, tok, extra={"encoding": {"max_len": 256}, "step": 1,
                                                               "temperatures": {"noul": 1.0}, "calibrated_at_step": 1})
    return TestClient(create_app(path, device="cpu"))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["device"] == "cpu"


def test_systemone_three_types(client):
    """一个请求里 3 种题型：noul 返回 0–1 的概率；choice 返回选中项、各选项概率（和为 1）、confidence；
    score 返回期望分数、等级描述和各级概率。"""
    body = {"state": {"ticket": "order late"}, "questions": {
        "urgent": {"type": "noul", "instructions": "late ?"},
        "topic": {"type": "choice", "instructions": "a ?", "criteria": {"a": "a", "b": "b", "c": "c"}},
        "level": {"type": "score", "instructions": "c ?", "criteria": ["0", "1", "2"]},
    }}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    a = out["answers"]
    assert set(a) == {"urgent", "topic", "level"}
    assert 0 <= a["urgent"]["noul"] <= 1
    assert a["topic"]["choice"] in {"a", "b", "c"} and abs(sum(a["topic"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["topic"]["confidence"] <= 1
    assert len(a["level"]["probabilities"]) == 3 and 0 <= a["level"]["score"] <= 2
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0


@pytest.mark.parametrize("body", [
    {"questions": {"q": {"type": "noul", "instructions": "x"}}},                       # 缺 state
    {"state": "s", "questions": {}},                                                   # 没有问题
    {"state": "s", "questions": {"q": {"type": "maybe", "instructions": "x"}}},        # 未知题型
    {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "a"}}}},  # 只有 1 个选项
    {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["0"] * 11}}},   # 超过 10 级
])
def test_bad_requests_return_400(client, body):
    assert client.post("/v1/systemone", json=body).status_code == 400
