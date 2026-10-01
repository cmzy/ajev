"""外部评测集：把社区公开的 Jev 类评测统一转换成 Decision JSONL，并检查与训练数据的重合。

    git clone --depth 1 https://github.com/adii-py/jevbench   /tmp/jevbench
    git clone --depth 1 https://github.com/jaredpalmer/kev    /tmp/kev
    python -m ajev.eval.external --jevbench-dir /tmp/jevbench --kev-dir /tmp/kev \\
        --train data/build4/train.jsonl --out data/external

然后用现有的评测命令跑任意一个文件，例如：

    python -m ajev.eval.evaluate --data data/external/kev_transfer_v9.jsonl --predictor model \\
        --checkpoint runs/sft4/best

为什么需要外部评测：我们一直只看 typed-decisions 一个测试集，而且已经在它上面迭代了好几个版本。
反复根据同一个测试集的结果调整做法，分数会慢慢变得“偏乐观”。外部评测集是别人出的题，
数据来源、出题方式都和我们的训练数据不同，更能反映模型的真实泛化能力。

收录的三套评测：

=======================  ======  ==============================================================
评测                      题数     说明
=======================  ======  ==============================================================
JevBench 公开集            231    社区常用的 Jev 类模型榜单（easy / original / hard 三档）；
                                  人工编写的政策判断、路由、抽取、打分等题，硬标签
Kev transfer-v4            764    Kev 训练时从没见过的数据源：MMLU、SciQ、QNLI、PAWS、Emotion、
                                  推特冒犯言论、组合规则等；硬标签
Kev transfer-v9           1264    v4 全部题 + 10 选 1 的 MMLU-Pro、关键记录埋在无关文本里（buried）、
                                  删掉关键证据的“无法判断”题（unknowable）、选项里加/不加
                                  “以上都不对”（none_present / none_absent）、选项重新排列（permuted）
eikos-decisions heldout   1191    英语 / 西班牙语 / 巴西葡萄牙语的业务场景题（发票对账、政策、
                                  权衡取舍、多跳推理、陷阱题……），答案是教师模型给出的概率分布
=======================  ======  ==============================================================

每道题的 ``source`` 写成 ``<评测名>/<子类>``（例如 ``kev_v9/mmlu_pro``），评测报告会按它分组；
Kev 的非 clean 变体写成 ``kev_v9/<来源>/<变体>``，方便单独看这些难题。

关于“重合检查”（``--train``）：如果评测题的材料在训练集里出现过，模型可能是“记住了”而不是“理解了”，
分数会虚高。这里把每道题材料里的所有文字片段（JSON 的每个字符串值）规范化后，
和训练集中所有材料的文字片段比较：评测题里任何一个 ≥ 40 个字符的片段在训练集出现过，就算重合。
重合的题仍然写入文件，但在 ``meta.seen_in_train`` 标记出来，并在输出里汇总，方便单独评估。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from typing import Any, Iterator

from ajev.schema import Decision, decisions_from_jev, read_jsonl, write_jsonl

LANG_CODES = {"English": "en", "Spanish": "es", "Brazilian Portuguese": "pt"}


def _gold_probs(qtype: str, label: Any) -> dict[str, float]:
    """把各种写法的标准答案统一成“选项名 → 概率”的 one-hot 字典。

    - noul：True / False，或 "yes" / "no"、"true" / "false" → 选项名 "true" / "false"；
    - choice：选项名字符串，原样使用；
    - score：等级数字（0、1、2…，可能是 int 也可能是字符串）→ "0"、"1"、"2"…
    """
    if qtype == "noul":
        if isinstance(label, str):
            label = label.strip().lower() in ("true", "yes")
        return {"true" if label else "false": 1.0}
    return {str(label): 1.0}


def _finish(ds: list[Decision]) -> list[Decision]:
    """丢掉答案不在选项里的题（target 变成均匀分布），以及格式不合法的题。"""
    out = []
    for d in ds:
        if max(d.target) <= 1.0 / len(d.options) + 1e-9:
            continue
        try:
            d.validate()
        except ValueError:
            continue
        out.append(d)
    return out


# ---- JevBench ---------------------------------------------------------------------------------


def load_jevbench(repo_dir: str) -> Iterator[Decision]:
    """读取 JevBench 公开集（``datasets/public/{easy,original,hard}.jsonl``）。

    每行就是一个 Jev 问题：``state`` + ``question``（type / instructions / criteria），
    标准答案在 ``expected``（noul 写成 "yes" / "no"，score 写成等级数字）。
    """
    for path in sorted(glob.glob(os.path.join(repo_dir, "datasets", "public", "*.jsonl"))):
        tier = os.path.basename(path).removesuffix(".jsonl")
        with open(path, encoding="utf-8") as f:
            for row in map(json.loads, f):
                q = row["question"]
                ds = decisions_from_jev(row["state"], {"q": q}, group=row["id"], source=f"jevbench/{tier}",
                                        gold={"q": {"probabilities": _gold_probs(q["type"], row["expected"])}})
                for d in _finish(ds):
                    d.id = f"jevbench/{row['id']}"
                    d.meta.update({"tier": tier, "family": row.get("family")})
                    yield d


# ---- Kev transfer suites -------------------------------------------------------------------


def load_kev(path: str, suite: str) -> Iterator[Decision]:
    """读取 Kev 的评测套件文件（例如 ``evals/v9/transfer-v9/test.jsonl``）。

    每行是一个 Jev 请求：``state`` + ``questions``，标准答案写在每个问题自己的 ``label`` 字段里；
    ``_meta.source`` 是数据来源，``_meta.variant`` 是题目变体（clean 为原题）。
    """
    with open(path, encoding="utf-8") as f:
        for row in map(json.loads, f):
            meta = row.get("_meta", {})
            src, variant = meta.get("source", "unknown"), meta.get("variant", "clean")
            source = f"{suite}/{src}" if variant == "clean" else f"{suite}/{src}/{variant}"
            gold = {qid: {"probabilities": _gold_probs(q["type"], q["label"])} for qid, q in row["questions"].items()}
            ds = decisions_from_jev(row["state"], row["questions"], group=meta.get("id", ""), source=source, gold=gold)
            for d in _finish(ds):
                d.id = f"{suite}/{meta.get('id')}/{d.meta['question_id']}"
                d.meta.update({"variant": variant, "upstream": meta.get("repo")})
                yield d


# ---- eikos-decisions ------------------------------------------------------------------------


def load_eikos(split: str = "heldout") -> Iterator[Decision]:
    """读取 ``caiovicentino1/eikos-decisions``（core 子集）。

    选项是 ``[{"label": ..., "description": ...}, ...]`` 的列表，``target_probs`` 与之一一对应，
    是教师模型给出的概率分布（软标签），所以除了准确率，KL / Brier 也有意义。
    noul 题的选项名是 "yes" / "no"，这里改成我们统一使用的 "true" / "false"（概率跟着对应过去）。
    """
    from datasets import load_dataset

    for row in load_dataset("caiovicentino1/eikos-decisions", "core", split=split):
        qtype = row["question_type"]
        labels = [o["label"] for o in row["options"]]
        probs = dict(zip(labels, row["target_probs"]))
        if qtype == "noul":
            criteria = {("true" if o["label"].lower() in ("yes", "true") else "false"): o["description"]
                        for o in row["options"]}
            probs = {("true" if k.lower() in ("yes", "true") else "false"): v for k, v in probs.items()}
        elif qtype == "score":
            criteria = [o["description"] for o in sorted(row["options"], key=lambda o: int(o["label"]))]
        else:
            criteria = {o["label"]: o["description"] for o in row["options"]}
        q = {"type": qtype, "instructions": row["instructions"], "criteria": criteria}
        # 题目都写了选项描述，lang 参数（只影响 noul 默认描述）用 "en" 即可；真实语言记在 d.lang。
        ds = decisions_from_jev(row["state"], {"q": q}, group=row["id"], source=f"eikos/{row['family']}",
                                lang="en", gold={"q": {"probabilities": probs}})
        for d in _finish(ds):
            d.id = f"eikos/{row['id']}"
            d.lang = LANG_CODES.get(row["lang"], row["lang"])
            d.meta.update({"family": row["family"], "difficulty": row["difficulty"], "topic": row["topic"]})
            yield d


# ---- 与训练集的重合检查 ---------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _fragments(state: str, min_len: int = 40) -> set[str]:
    """把一份材料拆成规范化的文字片段：JSON 就取所有字符串值，否则整段文本；
    统一小写、合并空白，只保留至少 ``min_len`` 个字符的片段（太短的片段如“yes”到处都有，没有区分度）。"""
    def leaves(x: Any) -> Iterator[str]:
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v)

    try:
        parsed = json.loads(state)
        texts = list(leaves(parsed)) if isinstance(parsed, (dict, list)) else [state]
    except (json.JSONDecodeError, TypeError):
        texts = [state]
    out = set()
    for t in texts:
        t = _WS.sub(" ", t).strip().lower()
        if len(t) >= min_len:
            out.add(t)
    return out


def train_fragments(train_path: str) -> set[str]:
    """训练集所有材料的文字片段（只读一遍，供所有评测集共用）。"""
    seen: set[str] = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            seen |= _fragments(json.loads(line)["state"])
    return seen


def mark_seen(decisions: list[Decision], seen: set[str]) -> Counter:
    """在 ``meta.seen_in_train`` 标记材料与训练集重合的题，返回每个 source 的重合题数。"""
    hits = Counter()
    for d in decisions:
        d.meta["seen_in_train"] = bool(_fragments(d.state) & seen)
        if d.meta["seen_in_train"]:
            hits[d.source] += 1
    return hits


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevbench-dir", help="clone of github.com/adii-py/jevbench")
    ap.add_argument("--kev-dir", help="clone of github.com/jaredpalmer/kev")
    ap.add_argument("--no-eikos", action="store_true")
    ap.add_argument("--train", help="training JSONL for the overlap check (e.g. data/build4/train.jsonl)")
    ap.add_argument("--out", default="data/external")
    args = ap.parse_args(argv)

    sets: dict[str, list[Decision]] = {}
    if args.jevbench_dir:
        sets["jevbench_public"] = list(load_jevbench(args.jevbench_dir))
    if args.kev_dir:
        for suite, rel in (("kev_v4", "evals/v4/transfer-v4/test.jsonl"), ("kev_v9", "evals/v9/transfer-v9/test.jsonl")):
            sets[f"kev_transfer_{suite[-2:]}"] = list(load_kev(os.path.join(args.kev_dir, rel), suite))
    if not args.no_eikos:
        sets["eikos_heldout"] = list(load_eikos())

    os.makedirs(args.out, exist_ok=True)
    seen = train_fragments(args.train) if args.train else set()
    for name, ds in sets.items():
        line = f"[external] {name}: {len(ds)} decisions {dict(Counter(d.type for d in ds))}"
        if args.train:
            hits = mark_seen(ds, seen)
            line += f" | seen in train: {sum(hits.values())}" + (f" {dict(hits.most_common(5))}" if hits else "")
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), ds)
        print(line, flush=True)


if __name__ == "__main__":
    main()
