"""组装第二版大模型训练数据 data/lm2：lm1 原样保留 + 8,000 道难题，并建立“难题开发集”。

    python -m ajev.lm.build_lm2 --lm1 data/lm1 --build data/build4 --out data/lm2

为什么这样组：lora2 / lora3 在难题和长材料题上都没有进步，统计发现训练集里推理类难题只占 18%，
材料超过 1,500 token 的只有 2.1%（JevBench hard 档是 33%）。所以这一版只改一件事：**加难题**，
原来 33,700 道题一道不动（中文能力主要来自其中的简单分类题，保持不变也让对比更干净）。

新增的 8,000 道难题：
    1. bev-decision 四个难题子集里 lm1 还没用上的题，共 3,500 道（真实数据）：
       skills（长政策 / 合同）1,500、hard（陷阱题）1,000、numeric_temporal（日期数字）700、counterfactual 300；
    2. 程序生成的难题 4,500 道（ajev/data/hard_gen.py）：退款政策、发票对账、合同期限、审批路由、工单 SLA，
       答案由程序计算，约 30% 是长材料。

难题开发集 dev_hard.jsonl（只用来比较版本、选择配置，不参与训练）：
    1. bev-decision 四个子集的**测试部分**（Hugging Face 的 test split，与训练部分天然不重叠）各 150 道；
    2. 另外用不同的随机种子生成 500 道难题（与训练用的生成题按材料内容去重）。
所有候选题都按“材料 + 问题”的内容和训练集去重（不按编号：不同 split 的编号会重复）。

输出：train.jsonl、val.jsonl / val_typed.jsonl（从 lm1 复制，常规题开发集，用于选 checkpoint 和校准）、dev_hard.jsonl。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from collections import Counter

from ajev.data.hard_gen import generate
from ajev.schema import Decision, read_jsonl, write_jsonl

EXTRA_BEV = {"bev_skills": 1500, "bev_hard": 1000, "bev_numeric": 700, "bev_counterfactual": 300}
DEV_BEV_PER_SOURCE = 150


def key(d: Decision) -> str:
    """按内容判重：材料 + 问题 + 选项名。"""
    return hashlib.sha1((d.state + "\x00" + d.instructions + "\x00" + "|".join(o.name for o in d.options)).encode()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm1", default="data/lm1")
    ap.add_argument("--build", default="data/build4")
    ap.add_argument("--out", default="data/lm2")
    ap.add_argument("--n-gen", type=int, default=4500)
    ap.add_argument("--n-gen-dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # 第 1 步：lm1 原样保留。
    base = read_jsonl(os.path.join(a.lm1, "train.jsonl"))
    seen = {key(d) for d in base}
    print(f"[lm2] lm1 train: {len(base)}")

    # 第 2 步：从 build4 的训练部分补 bev 难题（排除 lm1 已经用过的内容）。
    pool: dict[str, list[Decision]] = {s: [] for s in EXTRA_BEV}
    for d in read_jsonl(os.path.join(a.build, "train.jsonl")):
        if d.source in pool and "#r" not in d.id and key(d) not in seen:
            pool[d.source].append(d)
    extra = []
    for s, n in EXTRA_BEV.items():
        pick = random.Random(f"{a.seed}/{s}").sample(pool[s], min(n, len(pool[s])))
        extra += pick
        print(f"[lm2] extra {s}: {len(pick)} (available {len(pool[s])})")
    seen |= {key(d) for d in extra}

    # 第 3 步：程序生成的难题（训练用）。
    gen = generate(a.n_gen, seed=a.seed + 1, split="t")
    gen = [d for d in gen if key(d) not in seen]
    seen |= {key(d) for d in gen}

    train = base + extra + gen
    random.Random(a.seed).shuffle(train)
    write_jsonl(os.path.join(a.out, "train.jsonl"), train)

    # 第 4 步：难题开发集 = bev 测试部分 + 另一批生成题，全部与训练集按内容去重。
    dev: list[Decision] = []
    by_src: dict[str, list[Decision]] = {s: [] for s in EXTRA_BEV}
    for d in read_jsonl(os.path.join(a.build, "test_public.jsonl")):
        if d.source in by_src and key(d) not in seen:
            by_src[d.source].append(d)
    for s, items in by_src.items():
        dev += random.Random(f"dev/{a.seed}/{s}").sample(items, min(DEV_BEV_PER_SOURCE, len(items)))
    gen_dev = [d for d in generate(a.n_gen_dev, seed=a.seed + 1000, split="d") if key(d) not in seen]
    dev += gen_dev
    write_jsonl(os.path.join(a.out, "dev_hard.jsonl"), dev)

    # 第 5 步：常规题开发集沿用 lm1 的 val / val_typed。
    for f in ("val.jsonl", "val_typed.jsonl"):
        shutil.copy(os.path.join(a.lm1, f), os.path.join(a.out, f))

    hard_src = set(EXTRA_BEV) | {d.source for d in gen}
    n_hard = sum(d.source in hard_src for d in train)
    print(f"[lm2] train {len(train)} decisions: hard-type {n_hard} ({n_hard / len(train):.1%}), "
          f"types {dict(Counter(d.type for d in train))}, zh {sum(d.lang == 'zh' for d in train) / len(train):.1%}")
    print(f"[lm2] dev_hard {len(dev)} decisions: {dict(Counter(d.source for d in dev))}")


if __name__ == "__main__":
    main()
