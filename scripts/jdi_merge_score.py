#!/usr/bin/env python3
"""合并多台机器的排行榜结果快照，用官方代码按完整题库打分。

    python scripts/jdi_merge_score.py --suite-dir runs/disuite --out runs/jdi_lora5_full/final \\
        runs/jdi_lora5_full/snapshots/*.jsonl.gz

同一请求出现多次时以 status=ok 的为准（机器被回收后续跑会有少量重复）。没跑到的请求按排行榜规则算错，
脚本会报告覆盖率，所以中途也可以用它看“到目前为止”的分数。
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def main() -> None:
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+")
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.2.1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best = {}
    for p in a.snapshots:
        opener = gzip.open if p.endswith(".gz") else open
        for line in opener(p, "rt", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["run_id"] not in best or best[r["run_id"]]["status"] != "ok":
                    best[r["run_id"]] = r
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in best.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    suite = Suite(a.suite_dir, a.edition)
    total = sum(1 for _ in suite.rows(apply_exclusions=True))
    s = score_run(suite, out / "results.jsonl", "ajev-lora5-vllm", out)
    print(json.dumps({"results": len(best), "scoreable": total, "coverage": round(len(best) / total, 4),
                      "status": dict(collections.Counter(r["status"] for r in best.values())),
                      "decision_index": s["decision_index"], "raw_index": s["raw_index"],
                      "areas": {x["id"]: round(x["skill"] * 100, 1) for x in s["areas"]},
                      "latency_ms": s["latency_ms"]}, indent=1))


if __name__ == "__main__":
    main()
