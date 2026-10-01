#!/usr/bin/env python3
"""部署前的一致性检查：在新机器上重新预测，和训练环境（Colab）保存的预测逐题比较。

    python scripts/verify_deploy.py --checkpoint runs/sft4/best \\
        --data data/build4/test_typed.jsonl --reference runs/sft4/preds_test_typed.jsonl

为什么要做这一步：同一个模型换了机器、换了 PyTorch / transformers 版本、从 GPU 换到 Apple 芯片或 CPU，
理论上结果应该一样，但实际可能因为实现差异而不同。我们就遇到过：在 Intel Mac 上用旧版 transformers
加载同一个模型，60 道题里只有 51 道的答案和 Colab 上一致。上线前先确认一致，才能放心使用。

检查内容：
1. 最高票答案一致的比例（目标 ≥ 99%）；
2. 概率的最大差、平均差（GPU 用 bf16、Mac 用 fp32，会有很小的数值差异，通常在 0.01 以内）；
3. 本机预测的准确率，应与训练环境报告的准确率基本相同。
一致率低于 ``--min-agree`` 时以非 0 状态退出，方便放进自动化脚本。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

from ajev import metrics
from ajev.model.predictor import EncoderPredictor
from ajev.schema import read_jsonl


def argmax(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="decision JSONL that the reference predictions were made on")
    ap.add_argument("--reference", required=True, help="{id, probs} JSONL saved in the training environment")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=0, help="check a fixed random sample of N decisions (0 = all)")
    ap.add_argument("--min-agree", type=float, default=0.99)
    args = ap.parse_args()

    # 第 1 步：读题目和参考预测，只保留两边都有的题（可选随机抽样，种子固定，每次抽到同一批）。
    with open(args.reference, encoding="utf-8") as f:
        ref = {r["id"]: r["probs"] for r in map(json.loads, f) if r}
    decisions = [d for d in read_jsonl(args.data) if d.id in ref]
    if args.limit:
        decisions = random.Random(0).sample(decisions, min(args.limit, len(decisions)))
    if not decisions:
        sys.exit("no overlap between --data and --reference")

    # 第 2 步：本机预测（自动使用 checkpoint 里的校准温度，和参考预测保存时一致）。
    pred = EncoderPredictor(args.checkpoint, device=args.device)
    t0 = time.perf_counter()
    local = pred.predict(decisions)
    secs = time.perf_counter() - t0

    # 第 3 步：逐题比较。
    agree = sum(argmax(p) == argmax(ref[d.id]) for d, p in zip(decisions, local)) / len(decisions)
    diffs = [abs(a - b) for d, p in zip(decisions, local) for a, b in zip(p, ref[d.id])]
    acc_local = metrics.compute(decisions, local)["accuracy"]
    acc_ref = metrics.compute(decisions, [ref[d.id] for d in decisions])["accuracy"]
    print(f"device            {pred.device}")
    print(f"decisions         {len(decisions)}  ({secs:.1f}s, {1000 * secs / len(decisions):.1f} ms/decision)")
    print(f"argmax agreement  {agree:.4f}")
    print(f"max |prob diff|   {max(diffs):.4f}   mean {sum(diffs) / len(diffs):.5f}")
    print(f"accuracy          local {acc_local:.4f}   reference {acc_ref:.4f}")
    if agree < args.min_agree:
        print(f"FAIL: agreement {agree:.4f} < {args.min_agree}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
