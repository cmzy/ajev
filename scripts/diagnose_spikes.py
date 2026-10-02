#!/usr/bin/env python3
"""排查 LoRA 训练中的梯度尖峰：重建每一步用到的题，找出“制造尖峰”的数据。

    python scripts/diagnose_spikes.py --run runs/gemma_lora1b --train data/lm1/train.jsonl

原理：训练的分批是**确定的**——由题目长度、``--max-tokens``、``--batch-size`` 和随机种子决定
（ajev/train/train.py 的 ``token_budget_batches``）。只要用同样的分词器和参数重算一遍，就能知道第 N 步
具体用了哪几道题。第 N 个优化器步对应第 (N-1)×grad_accum … N×grad_accum-1 个 micro-batch（只训 1 个 epoch 时）；
固定题数分批（``decisions_per_step``）时直接对应计划表里的第 N 步。

训练日志每 10 步记一条，包含**该步**的梯度范数 ``gnorm`` 和**累计**被跳过的步数 ``skipped``，所以有两种分析：

1. 精确定位：日志里 gnorm 超过阈值的那几步（只有记日志的那一步的 gnorm 是已知的），直接列出它们用到的题；
2. 整体对比：把每 10 步分成一个窗口，``skipped`` 增加了的窗口叫“尖峰窗口”，其余叫“正常窗口”，
   比较两类窗口里各数据源、被截断的题、长题的占比。某个特征在尖峰窗口里占比明显更高（富集），
   就是可疑的尖峰来源。

必须用与训练时**完全相同**的分词器（transformers 版本也要一致），否则长度不同、分批就不同，所以要在训练环境
（Colab VM）上运行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.lm.predictor import load_tokenizer, prompt_ids  # noqa: E402
from ajev.lm.prompt import MAX_OPTIONS  # noqa: E402
from ajev.lm.train_utils import fixed_count_steps  # noqa: E402
from ajev.schema import read_jsonl  # noqa: E402
from ajev.train.train import token_budget_batches  # noqa: E402


def share(counter: Counter, total: int) -> dict:
    return {k: round(v / total, 3) for k, v in counter.most_common()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing args.json and log.jsonl")
    ap.add_argument("--train", required=True)
    ap.add_argument("--gnorm", type=float, default=100.0, help="threshold for 'spike' steps")
    a = ap.parse_args()

    args = json.load(open(os.path.join(a.run, "args.json")))
    log = [json.loads(line) for line in open(os.path.join(a.run, "log.jsonl"))]
    accum, max_state = args["grad_accum"], args["max_state_tokens"]

    # 第 1 步：用与训练相同的分词器和截断长度重算每道题的提示词长度，重建第 0 个 epoch 的分批。
    tok = load_tokenizer(args["model"])
    train = [d for d in read_jsonl(a.train) if len(d.options) <= MAX_OPTIONS]
    if args.get("over_limit") == "drop":
        train = [d for d in train if len(tok.encode(d.state, add_special_tokens=False)) <= max_state]
    state_len = [len(tok.encode(d.state, add_special_tokens=False)) for d in train]
    lengths = [len(prompt_ids(tok, d, max_state)) for d in train]
    if args.get("decisions_per_step"):
        # 固定题数分批（gemma_lora2 之后）：第 N 步就是计划表里的第 N 个步。
        steps = fixed_count_steps(lengths, args["decisions_per_step"], args["max_tokens"], args["batch_size"],
                                  seed=args["seed"] * 1000, sort_block=args.get("step_sort_block", 0))
        plan = [[i for mb in st for i in mb] for st in steps]
    else:
        batches = token_budget_batches(lengths, args["max_tokens"], args["batch_size"], seed=args["seed"] * 1000)
        plan = [[i for mb in batches[s: s + accum] for i in mb] for s in range(0, len(batches), accum)]

    def step_items(step: int) -> list[int]:
        return plan[step - 1] if step <= len(plan) else []

    def describe(idx: list[int]) -> dict:
        n = len(idx)
        return {"n": n,
                "truncated": round(sum(state_len[i] > max_state for i in idx) / max(1, n), 3),
                "mean_prompt_tokens": round(sum(lengths[i] for i in idx) / max(1, n)),
                "types": share(Counter(train[i].type for i in idx), n),
                "sources": share(Counter(train[i].source for i in idx), n)}

    # 第 2 步：精确定位——日志里 gnorm 超过阈值的步。
    print(f"== steps with logged gnorm > {a.gnorm}")
    for r in log:
        if "gnorm" in r and r.get("gnorm", 0) > a.gnorm and "loss" in r:
            idx = step_items(r["step"])
            d = describe(idx)
            print(f"step {r['step']} gnorm {r['gnorm']}: n={d['n']} truncated={d['truncated']} "
                  f"mean_tokens={d['mean_prompt_tokens']} sources={dict(list(d['sources'].items())[:6])}")

    # 第 3 步：整体对比——尖峰窗口 vs 正常窗口。
    rows = [r for r in log if "skipped" in r and "loss" in r]
    spike_idx, normal_idx = [], []
    for prev, cur in zip(rows, rows[1:]):
        idx = [i for s in range(prev["step"] + 1, cur["step"] + 1) for i in step_items(s)]
        (spike_idx if cur["skipped"] > prev["skipped"] else normal_idx).extend(idx)
    S, N = describe(spike_idx), describe(normal_idx)
    print(f"\n== spike windows: {S['n']} decisions | normal windows: {N['n']} decisions")
    print(f"truncated share: spike {S['truncated']}  normal {N['truncated']}")
    print(f"mean prompt tokens: spike {S['mean_prompt_tokens']}  normal {N['mean_prompt_tokens']}")
    print(f"types: spike {S['types']}  normal {N['types']}")
    print("source enrichment (share in spike windows / share in normal windows), sources with >= 1% share:")
    rows_out = []
    for src, sh in S["sources"].items():
        base = N["sources"].get(src, 0.0)
        if sh >= 0.01:
            rows_out.append((sh / base if base else float("inf"), src, sh, base))
    for ratio, src, sh, base in sorted(rows_out, reverse=True):
        print(f"  {src:40s} spike {sh:.3f}  normal {base:.3f}  x{ratio:.2f}")


if __name__ == "__main__":
    main()
