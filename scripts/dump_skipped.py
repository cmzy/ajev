#!/usr/bin/env python3
"""把训练中被跳过的步用到的**全部**题目找出来并保存，用于排查梯度尖峰（不需要 GPU、不需要 torch）。

    python scripts/dump_skipped.py --run runs/gemma_lora4 --train data/lm2/train.jsonl --out data/lm2/skipped_lora4.jsonl

训练日志里每个被跳过的步只记录了损失最大的 5 道题。但分批是确定的（同样的题目长度、种子、参数 → 同样的分批），
所以用和训练时**完全相同**的分词器（transformers 版本也要一致）重算每道题的提示词长度，就能重建每一步的 32 道题。
脚本会用日志里的 top_loss 题目检查重建结果：这些题必须都出现在重建出的那一步里，否则说明长度算得不一样，直接报错。
只支持固定题数分批（--decisions-per-step > 0）的训练。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.lm.prompt import MAX_OPTIONS, build_user_message  # noqa: E402
from ajev.lm.train_utils import fixed_count_steps  # noqa: E402
from ajev.schema import read_jsonl, write_jsonl  # noqa: E402


def prompt_len(tok, d, max_state_tokens: int) -> int:
    """与 ajev/lm/predictor.py 的 prompt_ids 完全相同的计算，只是不依赖 torch。"""
    state_ids = tok.encode(d.state, add_special_tokens=False)
    state = d.state if len(state_ids) <= max_state_tokens else tok.decode(state_ids[:max_state_tokens]) + " …[truncated]"
    ids = tok.apply_chat_template([{"role": "user", "content": build_user_message(d, state)}],
                                  add_generation_prompt=True, tokenize=True)
    return len(ids if isinstance(ids, list) else ids["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="log file (default: {run}/log.jsonl)")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    args = json.load(open(os.path.join(a.run, "args.json")))
    log = [json.loads(line) for line in open(a.log or os.path.join(a.run, "log.jsonl"))]
    assert args["decisions_per_step"], "only fixed-count runs are supported"
    tok = AutoTokenizer.from_pretrained(args["model"])
    train = [d for d in read_jsonl(a.train) if len(d.options) <= MAX_OPTIONS]
    if args.get("over_limit") == "drop":
        train = [d for d in train if len(tok.encode(d.state, add_special_tokens=False)) <= args["max_state_tokens"]]
    lengths = [prompt_len(tok, d, args["max_state_tokens"]) for d in train]
    steps = fixed_count_steps(lengths, args["decisions_per_step"], args["max_tokens"], args["batch_size"],
                              seed=args["seed"] * 1000, sort_block=args.get("step_sort_block", 0))
    plan = [[i for mb in st for i in mb] for st in steps]

    out = []
    for r in log:
        if not r.get("skipped_update"):
            continue
        idx = plan[r["step"]]  # 日志里的 step 是更新前的计数：第 step+1 个优化器步，即 plan[step]
        ids = {train[i].id for i in idx}
        logged = {t["id"]: t["loss"] for t in r["top_loss"]}
        missing = [i for i in logged if i not in ids]
        if missing:
            raise SystemExit(f"step {r['step']}: logged items {missing} not in the reconstructed step — lengths differ")
        for i in idx:
            d = train[i]
            d.meta = {**d.meta, "skip_step": r["step"], "gnorm": r["gnorm"], "median_gnorm": r.get("median_gnorm"),
                      "prompt_tokens": lengths[i], "logged_loss": logged.get(d.id)}
            out.append(d)
    write_jsonl(a.out, out)
    print(f"{sum(1 for r in log if r.get('skipped_update'))} skipped steps, {len(out)} decisions -> {a.out}")


if __name__ == "__main__":
    main()
