"""Evaluate a predictor (or a predictions file) on a Decision JSONL file.

    python -m ajev.eval.evaluate --data data/build/test_typed.jsonl --predictor prior --train data/build/train.jsonl
    python -m ajev.eval.evaluate --data data/build/test_public.jsonl --predictions runs/x/preds.jsonl

Predictions file format: one JSON object per line, ``{"id": ..., "probs": [...]}`` with
probabilities in the same option order as the data file.
"""

from __future__ import annotations

import argparse
import json
import random

from ajev import metrics
from ajev.predictors import Predictor, PriorPredictor, RandomPredictor, UniformPredictor
from ajev.schema import Decision, read_jsonl

COLS = ("n", "accuracy", "chance_acc", "brier", "kl", "ece", "mean_confidence")
HEADERS = ("n", "acc", "chance_acc", "brier", "kl", "ece", "jev_conf")


def evaluate(decisions: list[Decision], preds: list[list[float]]) -> dict:
    return {
        "overall": metrics.compute(decisions, preds),
        "by_type": metrics.breakdown(decisions, preds, lambda d: d.type),
        "by_lang": metrics.breakdown(decisions, preds, lambda d: d.lang),
        "by_source": metrics.breakdown(decisions, preds, lambda d: d.source),
    }


def shuffle_check(predictor: Predictor, decisions: list[Decision], preds: list[list[float]], seed: int = 0) -> float:
    rng = random.Random(seed)
    shuffled = [d.shuffled(rng) for d in decisions]
    return metrics.flip_rate(preds, decisions, shuffled, predictor.predict(shuffled))


def format_table(report: dict) -> str:
    lines = []
    for section in ("overall", "by_type", "by_lang", "by_source"):
        rows = {"all": report[section]} if section == "overall" else report[section]
        lines.append(f"\n## {section}")
        lines.append(f"{'':34s}" + "".join(f"{h:>12s}" for h in HEADERS))
        for name, m in rows.items():
            cells = "".join(f"{m.get(c, float('nan')):>12.4f}" if c != "n" else f"{m['n']:>12d}" for c in COLS)
            lines.append(f"{name[:34]:34s}{cells}")
    if "flip_rate" in report:
        lines.append(f"\noption-shuffle flip rate: {report['flip_rate']:.4f}")
    return "\n".join(lines)


def _load_predictions(path: str, decisions: list[Decision]) -> list[list[float]]:
    with open(path, encoding="utf-8") as f:
        by_id = {r["id"]: r["probs"] for r in map(json.loads, f) if r}
    missing = [d.id for d in decisions if d.id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} decisions have no prediction, e.g. {missing[:3]}")
    return [by_id[d.id] for d in decisions]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictor", choices=["uniform", "random", "prior"])
    src.add_argument("--predictions")
    ap.add_argument("--train", help="train JSONL (needed by --predictor prior)")
    ap.add_argument("--shuffle-check", action="store_true", help="also measure option-order flip rate")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args(argv)

    decisions = read_jsonl(args.data)
    predictor: Predictor | None = None
    if args.predictions:
        preds = _load_predictions(args.predictions, decisions)
    else:
        if args.predictor == "prior":
            if not args.train:
                raise SystemExit("--predictor prior needs --train")
            predictor = PriorPredictor(read_jsonl(args.train))
        else:
            predictor = UniformPredictor() if args.predictor == "uniform" else RandomPredictor()
        preds = predictor.predict(decisions)

    report = evaluate(decisions, preds)
    if args.shuffle_check and predictor is not None:
        report["flip_rate"] = shuffle_check(predictor, decisions, preds)
    print(format_table(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
