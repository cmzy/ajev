#!/usr/bin/env python3
"""用排行榜真正的评测数据检查训练集有没有被污染（逐题比对文本，而不是只看“用了哪个 split”）。

    python scripts/check_leaderboard_overlap.py --train data/lm3/train.jsonl --out docs/leaderboard_overlap_lm3.md

做法：
1. 下载 Jev Decision Index 各基准评测用的那部分数据（与我们训练数据源相关的那些）；
2. 把评测数据和训练数据里的文本（材料、问题、选项说明；JSON 材料先取出其中的字符串）规范化
   （小写、合并空白），两种方式比对：
   - 整段相同：长度 ≥ 20 字的文本整段出现在训练文本集合里（适合推文、用户问句等短文本）；
   - 片段相同：按句子切开，≥ 60 字的句子出现在训练里（适合长材料部分重叠）。
3. 每个基准报告：评测条数、命中条数、命中的训练来源。命中不一定是泄漏（例如合同里的标准条款、
   常见套话），但每一条都应该看一眼。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.schema import read_jsonl  # noqa: E402

WS = re.compile(r"\s+")
SPLIT = re.compile(r"[\n。！？!?]|(?<=[.;])\s")


def norm(t: str) -> str:
    return WS.sub(" ", str(t).lower()).strip()


def leaves(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in leaves(v)]
    if isinstance(x, list):
        return [s for v in x for s in leaves(v)]
    return []


def texts_of_decision(d) -> list[str]:
    try:
        st = json.loads(d.state)
        out = leaves(st) if isinstance(st, (dict, list)) else [d.state]
    except (json.JSONDecodeError, TypeError):
        out = [d.state]
    return out + [d.instructions] + [o.desc for o in d.options if o.desc]


def frags(t: str) -> set[str]:
    return {norm(p) for p in SPLIT.split(t) if len(norm(p)) >= 60}


# 排行榜评测数据：(名字, 加载函数参数, 取文本的字段)。只列与我们训练数据源相关的基准。
PQ = "hf://datasets/kiddothe2b/contract-nli@refs%2Fconvert%2Fparquet/contractnli_b/test/0000.parquet"
EVALS = [
    ("HellaSwag (validation)", ("Rowan/hellaswag", None, "validation"), ["ctx", "endings"]),
    ("WinoGrande (dev)", ("allenai/winogrande", "winogrande_xl", "validation"), ["sentence"]),
    ("GSM8K (test)", ("openai/gsm8k", "main", "test"), ["question"]),
    ("RAGTruth (test)", ("wandb/RAGTruth-processed", None, "test"), ["context", "output"]),
    ("ContractNLI (test)", ("parquet", {"test": PQ}, "test"), ["premise", "hypothesis"]),
    ("Humicroedit (test)", ("tasksource/humicroedit", "subtask-2", "test"), ["original1", "original2"]),
    ("iSarcasmEval A-En (test)", ("viethq1906/isarcasm_2022_taskA_En", None, "test"), ["sentence"]),
    ("ACOS (test)", ("NEUDM/acos", None, "test"), ["input"]),
    ("New Yorker matching (test)", ("jmhessel/newyorker_caption_contest", "matching", "test"), ["image_description", "caption_choices"]),
    ("ANLI (test r1)", ("facebook/anli", "plain_text", "test_r1"), ["premise", "hypothesis"]),
    ("ANLI (test r2)", ("facebook/anli", "plain_text", "test_r2"), ["premise", "hypothesis"]),
    ("ANLI (test r3)", ("facebook/anli", "plain_text", "test_r3"), ["premise", "hypothesis"]),
    ("BANKING77 (test)", ("mteb/banking77", None, "test"), ["text"]),
    ("CLINC150+OOS (test)", ("clinc/clinc_oos", "plus", "test"), ["text"]),
    ("When2Call MCQ (test)", ("nvidia/When2Call", "test", "mcq"), ["question"]),
    ("PhishNChips", ("AreLit/PhishNChips", None, None), None),
    ("NLI4CT (tasksource test)", ("tasksource/nli4ct", None, "test"), ["Statement"]),
    ("MMLU (test)", ("cais/mmlu", "all", "test"), ["question"]),
    ("MMLU-Pro (test)", ("TIGER-Lab/MMLU-Pro", None, "test"), ["question"]),
]


def load(args):
    from datasets import Image, load_dataset

    path, cfg, split = args
    if path == "parquet":
        ds = load_dataset("parquet", data_files=cfg, split=split)
    elif split is None:  # 不知道 split 名：全部 split 都算
        dd = load_dataset(path, cfg)
        from datasets import concatenate_datasets

        ds = concatenate_datasets([dd[k] for k in dd])
    else:
        ds = load_dataset(path, cfg, split=split)
    img = [c for c, f in ds.features.items() if isinstance(f, Image)]
    return ds.remove_columns(img) if img else ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = read_jsonl(a.train)
    full, frag = defaultdict(set), defaultdict(set)
    for d in rows:
        for t in texts_of_decision(d):
            n = norm(t)
            if len(n) >= 20:
                full[n].add(d.source)
            for f in frags(t):
                frag[f].add(d.source)
    lines = [f"# 排行榜评测数据与训练集的重叠：{a.train}", "",
             "| 排行榜评测数据 | 条数 | 整段相同 | 共享 ≥60 字句子 | 命中的训练来源 | 示例 |", "|---|---:|---:|---:|---|---|"]
    for name, args, fields in EVALS:
        try:
            ds = load(args)
        except Exception as e:  # noqa: BLE001
            lines.append(f"| {name} | 加载失败：{type(e).__name__} {str(e)[:80]} | | | | |")
            continue
        fields = fields or [c for c, f in ds.features.items() if getattr(f, "dtype", "") == "string"]
        n_full = n_frag = 0
        srcs, example = Counter(), ""
        for row in ds:
            ts = [s for f in fields for s in leaves(row.get(f))]
            hit_full = [t for t in ts if len(norm(t)) >= 20 and norm(t) in full]
            hit_frag = [f for t in ts for f in frags(t) if f in frag]
            n_full += bool(hit_full)
            n_frag += bool(hit_frag)
            for t in hit_full:
                srcs.update(full[norm(t)])
            for f in hit_frag:
                srcs.update(frag[f])
            if (hit_full or hit_frag) and not example:
                example = (hit_full or hit_frag)[0][:80].replace("|", "/")
        lines.append(f"| {name} | {len(ds)} | {n_full} | {n_frag} | {', '.join(f'{s}({n})' for s, n in srcs.most_common(4))} | {example} |")
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
