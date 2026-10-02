#!/usr/bin/env python3
"""组装补训数据：被跳过步里的题 + 若干倍随机抽取的原训练题（防止模型偏向这几类题、遗忘其他能力）。

    python scripts/build_supplement.py --skipped data/lm2/skipped_lora4.jsonl --train data/lm2/train.jsonl \\
        --out data/lm2/supplement_lora4.jsonl --replay 2

然后从训练好的适配器出发补训（学习率低、只拦截极端梯度，否则这批长题会被同样的规则再次跳过）：

    python -m ajev.lm.train --model google/gemma-4-12B-it --init-adapter runs/gemma_lora4/best \\
        --train data/lm2/supplement_lora4.jsonl --val data/lm2/val.jsonl data/lm2/val_typed.jsonl data/lm2/dev_hard.jsonl \\
        --out runs/gemma_lora4s --lr 1e-5 --skip-gnorm-ratio 0 --skip-gnorm 2000 \\
        --decisions-per-step 32 --max-tokens 24576 --batch-size 32 --max-state-tokens 16384
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.schema import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skipped", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", type=float, default=2.0, help="random training decisions per skipped decision")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    skipped = read_jsonl(a.skipped)
    ids = {d.id for d in skipped}
    for d in skipped:  # 去掉排查时加的元信息，保持和训练集一样的格式
        for k in ("skip_step", "gnorm", "median_gnorm", "prompt_tokens", "logged_loss"):
            d.meta.pop(k, None)
    rest = [d for d in read_jsonl(a.train) if d.id not in ids]
    replay = random.Random(a.seed).sample(rest, min(len(rest), int(len(skipped) * a.replay)))
    out = skipped + replay
    random.Random(a.seed + 1).shuffle(out)
    write_jsonl(a.out, out)
    print(f"{len(skipped)} skipped + {len(replay)} replay = {len(out)} decisions -> {a.out}")


if __name__ == "__main__":
    main()
