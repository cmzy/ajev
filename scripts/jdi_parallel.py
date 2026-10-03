#!/usr/bin/env python3
"""并行跑 Decision Index：题库切成 N 份，N 个官方运行器同时连同一个 AJev vLLM 服务，最后合并结果并打分。

    # 先启动服务（另一个终端 / 后台）：python -m ajev.serve_vllm --adapter runs/gemma_lora5/best
    python scripts/jdi_parallel.py --suite-dir /content/disuite --out runs/jdi_lora5_full --shards 8
    python scripts/jdi_parallel.py --suite-dir /content/disuite --rows sample.jsonl.gz --out runs/jdi_lora5_vllm_sample

为什么这样做：官方运行器一次只发一个请求（为了测单请求延迟），一个运行器喂不饱 GPU；
vLLM 服务会把同时到达的请求合并成批次，所以开多个运行器就能成倍提速。
每道题的输入、输出和串行运行完全一样，只是同时在算的请求多了，不影响排行榜规则。

断点续跑：每个分片的 results.jsonl 由官方运行器逐条追加，重新执行本脚本会跳过已完成的请求
（官方运行器只重试 status=error 的行）。合并时同一请求以最后一条为准。
打分：给了 --rows（样本）时只按样本里的请求计分；否则按完整题库计分（官方 score_run）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_shards(rows, out: Path, n: int) -> list[Path]:
    """轮流分配（第 i 行进第 i % n 份），每份里各个基准的比例相同，几个运行器差不多同时跑完。"""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"shard{i}.jsonl.gz" for i in range(n)]
    if all(p.exists() for p in paths):
        return paths
    files = [gzip.open(p, "wt", encoding="utf-8") for p in paths]
    for i, r in enumerate(rows):
        files[i % n].write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in files:
        f.close()
    return paths


def merge(shard_dirs: list[Path], target: Path) -> int:
    best = {}
    for d in shard_dirs:
        p = d / "results.jsonl"
        if p.exists():
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r["run_id"] not in best or best[r["run_id"]]["status"] != "ok":
                        best[r["run_id"]] = r
    with target.open("w", encoding="utf-8") as f:
        for r in best.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(best)


def main() -> None:
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite, read_jsonl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.2.1")
    ap.add_argument("--rows", help="only these rows (e.g. a stratified sample); default: every scoreable row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ajev")
    a = ap.parse_args()

    out = Path(a.out)
    suite = Suite(a.suite_dir, a.edition)
    sample_ids = None
    if a.rows:
        rows = list(read_jsonl(a.rows))
        sample_ids = {r["_evaluation"]["run_id"] for r in rows}
    else:
        rows = suite.rows(apply_exclusions=True)
    shards = write_shards(rows, out / "shards", a.shards)
    dirs = [out / f"run{i}" for i in range(a.shards)]
    t = time.time()
    procs = [subprocess.Popen([sys.executable, "-m", "decision_index", "run", "--engine", "http",
                               "--option", f"base_url={a.base_url}", "--option", f"model={a.model}",
                               "--rows", str(s), "--out", str(d)],
                              stdout=open(out / f"run{i}.log", "a"), stderr=subprocess.STDOUT)
             for i, (s, d) in enumerate(zip(shards, dirs))]
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        done = sum(sum(1 for _ in open(d / "results.jsonl")) for d in dirs if (d / "results.jsonl").exists())
        print(json.dumps({"event": "progress", "completed": done, "elapsed_s": round(time.time() - t)}), flush=True)
    codes = [p.returncode for p in procs]
    n = merge(dirs, out / "results.jsonl")
    print(json.dumps({"event": "merged", "results": n, "exit_codes": codes, "elapsed_s": round(time.time() - t)}), flush=True)

    if sample_ids is not None:
        class SampleSuite(Suite):
            def rows(self, apply_exclusions=False):
                for r in super().rows(apply_exclusions):
                    if r["_evaluation"]["run_id"] in sample_ids:
                        yield r
        suite = SampleSuite(a.suite_dir, a.edition)
    s = score_run(suite, out / "results.jsonl", "ajev-vllm", out)
    print(json.dumps({"event": "scored", "decision_index": s["decision_index"], "raw_index": s["raw_index"],
                      "counts": s["counts"], "latency_ms": s["latency_ms"],
                      "areas": {x["id"]: round(x["skill"] * 100, 1) for x in s["areas"]}}), flush=True)


if __name__ == "__main__":
    main()
