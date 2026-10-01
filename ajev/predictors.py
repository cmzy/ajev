"""Predictor interface plus trivial baselines.

Anything that maps Decisions to option distributions can be evaluated. The trained
encoder model will implement the same ``predict`` signature.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Protocol

from ajev.schema import Decision


class Predictor(Protocol):
    def predict(self, decisions: list[Decision]) -> list[list[float]]: ...


class UniformPredictor:
    """Every option equally likely: accuracy = chance (ties broken towards the first option)."""

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        return [[1.0 / len(d.options)] * len(d.options) for d in decisions]


class RandomPredictor:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        out = []
        for d in decisions:
            w = [self.rng.random() for _ in d.options]
            s = sum(w)
            out.append([x / s for x in w])
        return out


class PriorPredictor:
    """Predicts the training label frequencies of the same question (source + instructions + options).

    A strong-ish floor for typed-decisions, where each workflow re-asks the same 5 questions.
    Falls back to uniform for unseen questions.
    """

    def __init__(self, train: list[Decision], smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for d in train:
            for name, t in zip(d.option_names, d.target):
                self.counts[self._key(d)][name] += t

    @staticmethod
    def _key(d: Decision) -> tuple:
        return d.source, d.instructions, tuple(sorted(d.option_names))

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        out = []
        for d in decisions:
            c = self.counts.get(self._key(d), Counter())
            w = [c[n] + self.smoothing for n in d.option_names]
            s = sum(w)
            out.append([x / s for x in w])
        return out
