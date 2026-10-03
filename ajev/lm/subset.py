"""从完整训练集里按数据源配额挑出一个子集，用于大模型（Gemma 4 12B）的 LoRA 训练。

    python -m ajev.lm.subset --train data/build4/train.jsonl --out data/lm1/train.jsonl
    python -m ajev.lm.subset --train data/build4/train.jsonl --out data/lm2/train.jsonl --scale 2   # 配额翻倍

为什么不用全部 21 万道题（详见对话记录中的讨论，这里只列要点）：
1. 成本：12B 模型每个 token 的训练计算量约是 mmBERT 的 100 倍，全量 1 个 epoch 要十几个小时；
2. 大模型本来就会读题和推理，LoRA 主要教它“按格式回答、熟悉业务约定、给出合理概率”，几万道题足够；
3. 模板化的公开数据集练太多，模型会去迎合这些数据集的标注习惯，反而削弱原有的推理能力。

所以按“价值”分配配额，而不是随机抽：

=========================  ==============  ======================================================
类别                        默认配额         理由
=========================  ==============  ======================================================
typed-decisions            全部（去掉副本）  零样本最弱（0.695）、业务价值最高；build4 里复制的 ``#r`` 副本去掉
bev-decision 五个子集        共 9,000         Jev 格式、多领域、难题和反事实题
业务 / agent 类              共 5,100         客服工单、When2Call、PKU、HelpSteer3、Feedback
中文数据源                   每个 1,300        保证中文约 30%
其他公开数据集               每个 200          模型本来就会，少量即可学会格式和标注约定
=========================  ==============  ======================================================

``--scale`` 把除 typed-decisions 之外的所有配额同乘一个系数，方便下一轮扩大数据量做对比。
每个数据源内部随机抽样（种子固定），抽完后整体打乱。
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict

from ajev.schema import read_jsonl, write_jsonl

QUOTAS = {
    "bev_default": 3000, "bev_hard": 2000, "bev_skills": 2000, "bev_counterfactual": 1000, "bev_numeric": 1000,
    "support_tickets": 1500, "pku_saferlhf": 1000, "helpsteer3": 800, "feedback_collection": 800,
}
ZH_SOURCES = ["afqmc", "cmnli", "ocnli", "tnews", "massive_zh", "toxicn", "cold", "ultrafeedback_zh"]
TYPED_PREFIX = "typed_decisions/"


def quota_for(source: str, zh_quota: int, default_quota: int) -> int | None:
    """某个数据源的配额；None 表示全部保留（typed-decisions）。"""
    if source.startswith(TYPED_PREFIX):
        return None
    if source in QUOTAS:
        return QUOTAS[source]
    if source in ZH_SOURCES:
        return zh_quota
    return default_quota


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zh-quota", type=int, default=1300)
    ap.add_argument("--default-quota", type=int, default=200)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every quota except typed-decisions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # 第 1 步：按数据源分组；去掉 typed-decisions 的 "#r1"、"#r2" 副本（build4 用 --typed-repeat 3 复制出来的）。
    by_source: dict[str, list] = defaultdict(list)
    for d in read_jsonl(args.train):
        if "#r" in d.id:
            continue
        by_source[d.source].append(d)

    # 第 2 步：每个数据源按配额随机抽样（为每个数据源单独建随机数生成器，增减其他数据源不影响它的抽样结果）。
    picked = []
    for source in sorted(by_source):
        items = by_source[source]
        q = quota_for(source, args.zh_quota, args.default_quota)
        if q is not None:
            q = int(q * args.scale)
            if len(items) > q:
                items = random.Random(f"{args.seed}/{source}").sample(items, q)
        picked += items

    # 第 3 步：整体打乱后写出，并打印统计。
    random.Random(args.seed).shuffle(picked)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, picked)
    langs, types = Counter(d.lang for d in picked), Counter(d.type for d in picked)
    typed = sum(d.source.startswith(TYPED_PREFIX) for d in picked)
    print(f"[subset] {len(picked)} decisions -> {args.out}")
    print(f"[subset] types {dict(types)} | zh {langs['zh'] / len(picked):.1%} | typed-decisions {typed} "
          f"({typed / len(picked):.1%})")


if __name__ == "__main__":
    main()
