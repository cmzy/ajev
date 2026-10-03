"""共享开头的 KV cache（LMPredictor.prefix_cache）必须和逐题计算给出相同的概率。

用一个随机初始化的小 Gemma 3 模型（float32），局部注意力窗口只有 8 个 token，
材料远长于窗口，这样也覆盖了局部注意力层的 cache 处理。需要 Gemma 4 的分词器（本地缓存或可下载）。
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ajev.lm.predictor import LMPredictor, common_prefix_len, load_tokenizer  # noqa: E402
from ajev.schema import Decision, Option  # noqa: E402


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3], [1, 2, 4]]) == 2
    assert common_prefix_len([[1, 2], [1, 2, 3]]) == 2
    assert common_prefix_len([[5], [6]]) == 0


def _decision(i, state, n, qtype="choice"):
    if qtype == "noul":
        opts = [Option("true"), Option("false")]
    else:
        opts = [Option(f"o{j}", f"option number {j} for q{i}") for j in range(n)]
    return Decision(id=f"d{i}", source="t", type=qtype, state=state,
                    instructions=f"Question {i}: which one fits best? " + "x " * i,
                    options=opts, target=[1 / len(opts)] * len(opts))


def test_prefix_cache_matches_per_question():
    try:
        tok = load_tokenizer("google/gemma-4-12B-it")
    except Exception:  # noqa: BLE001
        pytest.skip("Gemma 4 tokenizer not available")
    torch.manual_seed(0)
    cfg = transformers.Gemma3TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, sliding_window=8, max_position_embeddings=4096,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
    model = transformers.Gemma3ForCausalLM(cfg).eval().float()
    p = LMPredictor("x", tok=tok, model=model, device="cpu", prefix_cache=False)
    p.min_shared_prefix = 4
    long = "The customer wrote a long message about refunds and invoices. " * 30
    ds = [_decision(i, long, 3 + i % 5) for i in range(7)] + [_decision(9, long, 2, "noul"),
                                                              _decision(12, long, 40),
                                                              _decision(10, "short other state", 4)]
    plain = p.predict(ds)
    p.prefix_cache = True
    for budget in (16 * 2 ** 30, 1):  # 1 字节：强制每批只放一条尾巴
        p.prefix_cache_bytes = budget
        cached = p.predict(ds)
        diff = max(abs(x - y) for a, b in zip(plain, cached) for x, y in zip(a, b))
        assert diff < 1e-5, diff
