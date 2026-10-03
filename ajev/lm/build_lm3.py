"""组装第三版大模型训练数据 data/lm3（lora5 用）：在 lm2 基础上做三项修正。

    python -m ajev.lm.build_lm3 --lm2 data/lm2 --build data/build4 --out data/lm3

1. **去掉 When2Call**：它的 test/mcq 是 Jev Decision Index 排行榜的测试集，lm2 里混入了 1,000 道；
2. **修正客服工单队列题**：选项从 52 个（混入 42 个永远不会正确的话题标签）改为 10 个团队队列 + 说明，
   这 485 道题原来因为超过 26 个选项从未参与训练，修正后可以正常训练（见 scripts/fix_ticket_queues.py）；
3. **加入多选项题**：BANKING77（77 类）和 CLINC150（151 类）用**完整标签集**作为选项，让模型练习
   A–Z 之后的两字母编码（AA、AB……）。只用这两个数据集的训练部分里 lm2 没用过的题
   （它们的测试部分是排行榜的测试集，从不用于训练）；
4. **补回工具调用类决策**（去掉 When2Call 测试集后这类题为 0）：
   - When2Call 官方训练数据 train_pref 1,500 道（“两个候选回复该发哪个”，标准答案是官方标注的好回复）；
   - 程序生成的工具调用题 1,500 道（ajev/data/hard_gen.py 的 gen_tool：该调用工具 / 追问 / 直接回答 / 做不到，
     以及该用哪个工具；长材料版本工具列表超过 26 个，同时练习两字母编码）。
   另外各留出一份组成工具类开发集 dev_tool.jsonl（只评测，不训练）；
5. **加强业务决策**（领域分析发现：客服、安全长材料为 0，中文推理几乎为 0，缺 HR、医疗分诊）：
   中文为主（80%）的五类原有生成难题 2,000 道；长对话客服约 1,000、安全日志 1,000、HR 休假 800、急诊分诊 800；
6. **补充排行榜各领域的训练数据**（ajev/data/bench_sources.py，只用排行榜不评测的部分）：
   HellaSwag、WinoGrande、GSM8K、MMLU auxiliary_train、RAGTruth、ContractNLI、Humicroedit、NLI4CT、
   iSarcasmEval、ACOS、New Yorker、ANLI r1/r2、钓鱼邮件，共约 1.4 万道；各数据源的评测部分各取 100 道
   组成 dev_bench.jsonl（只评测，不训练）；
7. **数据审计后的修正**（docs/audit_lm3.md）：
   - 用修好的转换器重新转换 HelpSteer3（回答不再被截到 1,200 字符）、Feedback-Collection（去掉混入的评分员评语）、
     客服工单（优先级说明不再与标签矛盾），替换 lm2 里的旧版本；
   - 用修好的生成器重新生成 lm2 带来的 4,500 道生成难题（干扰条款按领域、审批人必须高于申请人、写明边界等）；
   - 最后去污染（ajev/data/decontam.py）：删掉与排行榜评测数据有文本重叠的训练题。

其余（lm2 的 8,000 道难题、原有数据、开发集）不变；开发集 val / val_typed / dev_hard 原样复制。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter

from ajev.data.bench_sources import BENCH_QUOTAS, BENCH_SOURCES
from ajev.data.decontam import contaminated, load_index
from ajev.data.hard_gen import generate
from ajev.data.more_sources import MORE_SOURCES, TICKET_QUEUES
from ajev.schema import Decision, Option, read_jsonl, write_jsonl

WIDE = {"banking77": 1000, "clinc150": 1000}
# 审计后重新转换的数据源及题数（与 lm2 中的数量一致）
RECONVERT = {"helpsteer3": 800, "feedback_collection": 800, "support_tickets": 1500}
DROP_SOURCES = {"when2call"}


def full_label_options(build_train: str) -> dict[str, dict[str, str]]:
    """从已构建的数据里收集 BANKING77 / CLINC150 的全部标签名和说明（原题只随机抽了 4–20 个候选）。"""
    labels: dict[str, dict[str, str]] = {s: {} for s in WIDE}
    with open(build_train) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in labels:
                for o in r["options"]:
                    labels[r["source"]][o["name"]] = o.get("desc", "")
    return labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm2", default="data/lm2")
    ap.add_argument("--build", default="data/build4")
    ap.add_argument("--out", default="data/lm3")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # 第 1、2 步：lm2 去掉 When2Call；去掉需要重新转换 / 重新生成的旧版题（第 7 步补回）。
    out, dropped, fixed = [], Counter(), 0
    for d in read_jsonl(os.path.join(a.lm2, "train.jsonl")):
        if d.source in DROP_SOURCES or d.source in RECONVERT or d.source.startswith("gen_"):
            dropped[d.source.split("_")[0] if d.source.startswith("gen_") else d.source] += 1
            continue
        if d.source == "support_tickets" and d.id.endswith("/queue"):
            gold = d.options[d.gold_index].name
            d.options = [Option(q, desc) for q, desc in TICKET_QUEUES.items()]
            d.target = [1.0 if q == gold else 0.0 for q in TICKET_QUEUES]
            fixed += 1
        out.append(d)
    used = {d.id for d in out}
    print(f"[lm3] from lm2: {len(out)} kept, dropped {dict(dropped)}, ticket queues fixed {fixed}")

    # 第 3 步：多选项题（训练部分里没用过的题，选项换成完整标签集，顺序随机）。
    labels = full_label_options(os.path.join(a.build, "train.jsonl"))
    pool: dict[str, list[Decision]] = {s: [] for s in WIDE}
    for d in read_jsonl(os.path.join(a.build, "train.jsonl")):
        if d.source in pool and d.id not in used and "#r" not in d.id:
            pool[d.source].append(d)
    rng = random.Random(a.seed)
    for src, n in WIDE.items():
        names = sorted(labels[src])
        for d in rng.sample(pool[src], min(n, len(pool[src]))):
            gold = d.options[d.gold_index].name
            opts = [Option(x, labels[src][x]) for x in names]
            rng.shuffle(opts)
            w = Decision(id=d.id + "#full", source=src + "_full", type="choice", state=d.state,
                         instructions=d.instructions, options=opts,
                         target=[1.0 if o.name == gold else 0.0 for o in opts], lang=d.lang, group=d.group,
                         meta={"gold": gold})
            w.validate()
            out.append(w)
        print(f"[lm3] wide {src}: {min(n, len(pool[src]))} decisions with {len(names)} options")

    # 第 4 步：工具调用类决策（When2Call 官方训练数据 + 程序生成），各留一份做开发集。
    w2c = MORE_SOURCES["when2call_pref"]
    out += list(w2c.iter_decisions("train#train", 1500, a.seed))
    dev_tool = list(w2c.iter_decisions("train#eval", 300, a.seed))
    out += generate(1500, seed=a.seed + 7, split="t", families=["tool"])
    dev_tool += generate(300, seed=a.seed + 1007, split="d", families=["tool"])
    write_jsonl(os.path.join(a.out, "dev_tool.jsonl"), dev_tool)
    print(f"[lm3] tool decisions: when2call_pref + gen_tool added; dev_tool {len(dev_tool)}")

    # 第 5 步：加强业务决策（新的生成族 + 中文为主的原有五类难题）。
    out += generate(2000, seed=a.seed + 21, split="z", zh_share=0.8)
    for fam, n in (("chat", 667), ("seclog", 1000), ("hr", 800), ("triage", 800)):
        out += generate(n, seed=a.seed + 31, split="t", zh_share=0.5, families=[fam])

    # 第 6 步：排行榜各领域的训练数据（只用排行榜不评测的部分）。
    dev_bench = []
    for name, n in BENCH_QUOTAS.items():
        src = BENCH_SOURCES[name]
        got = list(src.iter_decisions(src.train_split, n, a.seed))
        out += got
        dev_bench += list(src.iter_decisions(src.eval_split, 100, a.seed))
        print(f"[lm3] bench {name}: {len(got)}")
    # 第 7 步：重新转换审计发现有问题的数据源、用修好的生成器重新生成 lm2 的生成难题。
    for name, n in RECONVERT.items():
        src = MORE_SOURCES[name]
        got = list(src.iter_decisions(src.train_split, n, a.seed))
        out += got
        print(f"[lm3] reconverted {name}: {len(got)}")
    out += generate(4500, seed=a.seed + 1, split="t")

    # 去污染：删掉与排行榜评测数据有文本重叠的训练题。
    index = load_index(os.path.join(os.path.dirname(a.out) or ".", "decontam_index.json"))
    bad = [d for d in out if contaminated(d, index)]
    bad_ids = {d.id for d in bad}
    out = [d for d in out if d.id not in bad_ids]
    print(f"[lm3] decontam removed {len(bad)}: {dict(Counter(d.source for d in bad).most_common())}")

    # 去重：材料 + 问题 + 选项名完全相同的题只留一道（原数据集里偶有重复文本）。
    seen, uniq = set(), []
    for d in out:
        k = hashlib.sha1((d.state + "\0" + d.instructions + "\0" + "|".join(o.name for o in d.options)).encode()).hexdigest()
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    print(f"[lm3] removed {len(out) - len(uniq)} exact duplicates")
    out = uniq
    # 开发集里去掉材料与训练题相同的题（空材料除外，例如问答题的题目写在问题里、材料为空）。
    train_states = {d.state for d in out if d.state.strip()}
    dev_bench = [d for d in dev_bench if d.state not in train_states]
    dev_tool = [d for d in dev_tool if d.state not in train_states]
    write_jsonl(os.path.join(a.out, "dev_bench.jsonl"), dev_bench)
    write_jsonl(os.path.join(a.out, "dev_tool.jsonl"), dev_tool)
    print(f"[lm3] dev_bench {len(dev_bench)}, dev_tool {len(dev_tool)} after removing items whose state is in train")

    rng.shuffle(out)
    write_jsonl(os.path.join(a.out, "train.jsonl"), out)
    for f in ("val.jsonl", "val_typed.jsonl", "dev_hard.jsonl"):
        shutil.copy(os.path.join(a.lm2, f), os.path.join(a.out, f))
    print(f"[lm3] train {len(out)} decisions, options>26: {sum(len(d.options) > 26 for d in out)}")


if __name__ == "__main__":
    main()
