"""Build the training / validation / test mix as JSONL.

    python -m ajev.data.build --out data/build
    python -m ajev.data.build --out data/smoke --sources boolq,tnews --train-cap 50 --eval-cap 20

Outputs:
    train.jsonl         public train splits (capped per source) + typed-decisions train
    val.jsonl           public eval splits, first --eval-cap per source (model selection, temperature fit)
    test_public.jsonl   public eval splits, next --eval-cap per source (per-type / per-language report)
    val_typed.jsonl     typed-decisions train, held-out 10% of states (in-domain model selection)
    test_typed.jsonl    typed-decisions test, 2,000 decisions (headline benchmark)
    stats.json          counts per file / source / type / language
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from itertools import islice

from ajev.data.sources import SOURCES
from ajev.data.typed_decisions import iter_typed_decisions
from ajev.schema import Decision, write_jsonl


def _key(d: Decision) -> str:
    return hashlib.sha1(f"{d.state}\x00{d.instructions}".encode()).hexdigest()


def _state_key(d: Decision) -> str:
    return hashlib.sha1(d.state.encode()).hexdigest()


def _stats(ds: list[Decision]) -> dict:
    return {
        "total": len(ds),
        "by_source": dict(Counter(d.source for d in ds).most_common()),
        "by_type": dict(Counter(d.type for d in ds)),
        "by_lang": dict(Counter(d.lang for d in ds)),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/build")
    ap.add_argument("--sources", default="all", help="comma-separated source names, or 'all'")
    ap.add_argument("--train-cap", type=int, default=3000, help="max train decisions per public source")
    ap.add_argument("--eval-cap", type=int, default=300, help="max decisions per source in val and in test_public")
    ap.add_argument("--zh-cap-mult", type=float, default=2.0,
                    help="multiply --train-cap for Chinese sources (target: 30-50%% Chinese in train)")
    ap.add_argument("--zh-instr-prob", type=float, default=0.2,
                    help="probability an English source gets a Chinese instruction (cross-lingual)")
    ap.add_argument("--no-typed-decisions", action="store_true")
    ap.add_argument("--typed-val-frac", type=float, default=0.1,
                    help="fraction of typed-decisions train states held out as val_typed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    names = list(SOURCES) if args.sources == "all" else args.sources.split(",")
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"unknown sources: {unknown}; available: {list(SOURCES)}")

    rng = random.Random(args.seed)
    train: list[Decision] = []
    val: list[Decision] = []
    test_public: list[Decision] = []
    val_typed: list[Decision] = []
    test_typed: list[Decision] = []

    for name in names:
        src = SOURCES[name]
        print(f"[build] {name}: {src.path} {src.config or ''}", flush=True)
        cap = int(args.train_cap * (args.zh_cap_mult if src.lang == "zh" else 1))
        train += list(src.iter_decisions(src.train_split, cap, args.seed, args.zh_instr_prob))
        held = src.iter_decisions(src.eval_split, 2 * args.eval_cap, args.seed, args.zh_instr_prob)
        val += list(islice(held, args.eval_cap))
        test_public += list(held)

    if not args.no_typed_decisions:
        print("[build] typed_decisions", flush=True)
        typed_train = list(iter_typed_decisions("train"))
        groups = sorted({d.group for d in typed_train})
        held = set(random.Random(args.seed).sample(groups, int(len(groups) * args.typed_val_frac)))
        train += [d for d in typed_train if d.group not in held]
        val_typed = [d for d in typed_train if d.group in held]
        test_typed = list(iter_typed_decisions("test"))

    # Contamination guard: drop train items whose state shows up in any held-out file,
    # then exact-duplicate (state, instructions) pairs within train.
    held_states = {_state_key(d) for d in val + test_public + val_typed + test_typed}
    seen: set[str] = set()
    clean: list[Decision] = []
    dropped = Counter()
    for d in train:
        k = _key(d)
        if _state_key(d) in held_states and d.state:
            dropped["overlaps_heldout"] += 1
        elif k in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(k)
            clean.append(d)
    train = clean

    # Gold sits in a fixed position in many sources; shuffle choice/noul options once here
    # (training adds further on-the-fly permutations).
    train = [d.shuffled(rng) for d in train]
    val = [d.shuffled(rng) for d in val]
    test_public = [d.shuffled(rng) for d in test_public]
    rng.shuffle(train)

    os.makedirs(args.out, exist_ok=True)
    files = {"train": train, "val": val, "test_public": test_public, "val_typed": val_typed, "test_typed": test_typed}
    for fname, ds in files.items():
        if ds:
            write_jsonl(os.path.join(args.out, f"{fname}.jsonl"), ds)
    stats = {fname: _stats(ds) for fname, ds in files.items()}
    stats["train_dropped"] = dict(dropped)
    stats["args"] = vars(args)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    for fname, s in stats.items():
        if isinstance(s, dict) and "total" in s:
            print(f"[build] {fname}: {s['total']} {s['by_type']} {s['by_lang']}")
    print(f"[build] dropped from train: {dict(dropped)}")


if __name__ == "__main__":
    main()
