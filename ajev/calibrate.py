"""Fit one softmax temperature per question type on validation data.

    python -m ajev.calibrate --checkpoint runs/sft1/best --data data/build/val.jsonl data/build/val_typed.jsonl

Temperatures minimise soft-label log loss per type (grid + golden-section refine) and are
written into the checkpoint's ``ajev_config.json``, which ``EncoderPredictor`` reads.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from ajev import metrics
from ajev.schema import Decision, read_jsonl


def _nll(logits: list[list[float]], targets: list[list[float]], t: float) -> float:
    total = 0.0
    for lg, tg in zip(logits, targets):
        z = [x / t for x in lg]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        total -= sum(p * (x - lse) for p, x in zip(tg, z) if p > 0)
    return total / max(1, len(logits))


def fit_temperature(logits: list[list[float]], targets: list[list[float]]) -> float:
    grid = [0.05 * 1.15**i for i in range(46)]  # 0.05 .. ~27
    best = min(grid, key=lambda t: _nll(logits, targets, t))
    lo, hi = best / 1.15, best * 1.15
    g = (math.sqrt(5) - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if _nll(logits, targets, a) < _nll(logits, targets, b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def fit(decisions: list[Decision], logits: list[list[float]]) -> dict[str, float]:
    temps = {}
    for t in ("noul", "choice", "score"):
        idx = [i for i, d in enumerate(decisions) if d.type == t]
        if idx:
            temps[t] = round(fit_temperature([logits[i] for i in idx], [decisions[i].target for i in idx]), 4)
    return temps


def main(argv: list[str] | None = None) -> None:
    from ajev.model.encoder import AJEV_CONFIG
    from ajev.model.predictor import EncoderPredictor, softmax

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    decisions = [d for p in args.data for d in read_jsonl(p)]
    pred = EncoderPredictor(args.checkpoint, batch_size=args.batch_size, temperatures={})
    logits = pred.predict_logits(decisions)
    temps = fit(decisions, logits)

    before = metrics.compute(decisions, [softmax(lg) for lg in logits])
    after = metrics.compute(decisions, [softmax(lg, temps.get(d.type, 1.0)) for d, lg in zip(decisions, logits)])
    print(f"temperatures: {temps}")
    for k in ("accuracy", "brier", "kl", "ece"):
        print(f"{k:10s} before {before[k]:.4f}  after {after[k]:.4f}")

    path = os.path.join(args.checkpoint, AJEV_CONFIG)
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg["temperatures"] = temps
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
