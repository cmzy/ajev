"""Decision-quality metrics: accuracy and calibration.

All functions take parallel lists of Decisions (gold) and predicted distributions.
Conventions follow the open Jev replications so numbers are comparable:

* accuracy      argmax(pred) == gold label (``meta["gold_label"]`` if present, else argmax(target))
* chance_acc    (acc - c) / (1 - c) with c = mean 1/K, clipped at 0 (JevBench "intelligence")
* brier         mean over decisions of mean_k (p_k - t_k)^2 against the soft target
* kl            KL(target || pred), the soft-label log loss minus target entropy
* ece           top-label ECE, 15 equal-width bins
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from ajev.schema import Decision, jev_confidence

EPS = 1e-12


def _argmax(ps: list[float]) -> int:
    return max(range(len(ps)), key=ps.__getitem__)


def gold_index(d: Decision) -> int:
    label = d.meta.get("gold_label")
    if label is not None and label in d.option_names:
        return d.option_names.index(label)
    return d.gold_index


def ece(confidences: list[float], correct: list[bool], n_bins: int = 15) -> float:
    if not confidences:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    total = len(confidences)
    return sum(
        len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins if b
    )


def compute(decisions: list[Decision], preds: list[list[float]]) -> dict[str, float]:
    assert len(decisions) == len(preds), "decisions and predictions differ in length"
    n = len(decisions)
    if n == 0:
        return {"n": 0}
    acc = chance = brier = kl = 0.0  # chance: sum of 1/K
    top_conf: list[float] = []
    correct: list[bool] = []
    jev_conf = 0.0
    for d, p in zip(decisions, preds):
        k = len(d.options)
        assert len(p) == k, f"{d.id}: {len(p)} probabilities for {k} options"
        ok = _argmax(p) == gold_index(d)
        acc += ok
        chance += 1 / k
        brier += sum((pi - ti) ** 2 for pi, ti in zip(p, d.target)) / k
        kl += sum(ti * (math.log(ti + EPS) - math.log(pi + EPS)) for pi, ti in zip(p, d.target) if ti > 0)
        top_conf.append(max(p))
        correct.append(ok)
        jev_conf += jev_confidence(p)
    return {
        "n": n,
        "accuracy": acc / n,
        "chance_acc": max(0.0, (acc / n - chance / n) / (1 - chance / n)),
        "brier": brier / n,
        "kl": kl / n,
        "ece": ece(top_conf, correct),
        "mean_confidence": jev_conf / n,
    }


def breakdown(
    decisions: list[Decision], preds: list[list[float]], key: Callable[[Decision], str]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        groups[key(d)].append(i)
    return {
        g: compute([decisions[i] for i in idx], [preds[i] for i in idx]) for g, idx in sorted(groups.items())
    }


def flip_rate(base_preds: list[list[float]], decisions: list[Decision], shuffled: list[Decision],
              shuffled_preds: list[list[float]]) -> float:
    """Share of choice/noul decisions whose predicted option *name* changes after reordering options."""
    flips = total = 0
    for d, p, sd, sp in zip(decisions, base_preds, shuffled, shuffled_preds):
        if d.type == "score":
            continue
        total += 1
        flips += d.option_names[_argmax(p)] != sd.option_names[_argmax(sp)]
    return flips / total if total else float("nan")
