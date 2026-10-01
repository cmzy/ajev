"""Model-side tests; skipped where torch/transformers are unavailable (e.g. the base uv env)."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ajev.calibrate import fit_temperature  # noqa: E402
from ajev.model.batching import collate  # noqa: E402
from ajev.model.encoding import DecisionEncoder, Encoded  # noqa: E402
from ajev.schema import Decision, Option  # noqa: E402
from ajev.train.losses import rps, soft_ce, symmetric_kl, unpermute  # noqa: E402


class FakeTok:
    """Char-level tokenizer with the special ids DecisionEncoder needs."""

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    bos_token_id = eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 1000) for c in text]}


def decision(n_opts=3, state="s" * 50, desc="d" * 20):
    return Decision(id="x", source="t", type="choice", state=state, instructions="q?",
                    options=[Option(f"o{i}", desc) for i in range(n_opts)], target=[1.0] + [0.0] * (n_opts - 1))


def test_encoder_markers_and_truncation():
    enc = DecisionEncoder(FakeTok(), max_len=64, max_options=40, min_desc=4)
    e = enc.encode(decision())
    assert len(e.input_ids) <= 64 and e.truncated_state
    assert [e.input_ids[p] for p in e.marker_pos] == [3, 3, 3]
    assert e.input_ids[0] == 1 and e.input_ids[-1] == 2


def test_encoder_shrinks_descriptions_then_fails():
    enc = DecisionEncoder(FakeTok(), max_len=512, max_options=60, min_desc=4)
    e = enc.encode(decision(n_opts=5, state=""))
    assert len(e.marker_pos) == 5
    with pytest.raises(ValueError):
        DecisionEncoder(FakeTok(), max_options=10).encode(decision(n_opts=5))


def test_collate_and_unpermute():
    b = collate([Encoded([1, 3, 5, 3, 6, 2], [1, 3], False), Encoded([1, 3, 3, 3, 2], [1, 2, 3], False)], 0)
    assert b["input_ids"].shape == (2, 6) and b["option_mask"].tolist() == [[True, True, False], [True, True, True]]
    logits = torch.tensor([[1.0, 2.0, float("-inf")], [1.0, 2.0, 3.0]])
    perm = torch.tensor([[1, 0, 2], [2, 0, 1]])
    out = unpermute(logits, perm, b["option_mask"])
    assert out[0].tolist()[:2] == [2.0, 1.0] and out[0, 2] == float("-inf")
    assert out[1].tolist() == [2.0, 3.0, 1.0]


def test_losses():
    mask = torch.tensor([[True, True, False]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    good = torch.tensor([[5.0, -5.0, float("-inf")]])
    bad = torch.tensor([[-5.0, 5.0, float("-inf")]])
    assert soft_ce(good, target, mask) < soft_ce(bad, target, mask)
    assert rps(good, target, mask) < rps(bad, target, mask)
    assert symmetric_kl(good, good, mask).item() == pytest.approx(0, abs=1e-6)
    assert symmetric_kl(good, bad, mask).item() > 1
    loss = soft_ce(good.clone().requires_grad_(), target, mask)
    loss.sum().backward()  # -inf padding must not produce NaN gradients


def test_temperature_recovers_scale():
    rng = random.Random(0)
    logits, targets = [], []
    for _ in range(400):
        z = [rng.gauss(0, 1) for _ in range(4)]
        p = torch.softmax(torch.tensor(z), -1).tolist()
        k = rng.choices(range(4), weights=p)[0]
        logits.append([3 * x for x in z])  # model 3x overconfident
        targets.append([1.0 if i == k else 0.0 for i in range(4)])
    assert fit_temperature(logits, targets) == pytest.approx(3, rel=0.35)
