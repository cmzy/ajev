#!/usr/bin/env python3
"""训练数据审计：格式、标签分布、语言、长度、重复 / 近似重复、与评测数据的重叠。

    python scripts/audit_data.py --train data/lm3/train.jsonl \\
        --eval data/external/*.jsonl data/build4/test_typed.jsonl data/lm3/dev_*.jsonl data/wide/dev_wide.jsonl \\
        --out docs/audit_lm3.md

检查项（每个数据源分别统计，异常的会标出来）：
1. 格式：validate() 是否通过；选项数是否超过 255；target 是否合法分布；是否有空问题。
2. 标签分布：最常见答案的占比（“全猜同一个答案能对多少”）；是/否题里“是”的比例；软标签占比。
3. 语言：lang 标签和材料里中文字符比例是否一致（标 zh 却几乎没有中文、或标 en 却大量中文）。
4. 长度：材料字符数的中位数、最长；空材料的比例。
5. 重复：完全相同（材料 + 问题 + 选项）的题；近似重复（规范化后材料前 300 字相同）的组。
6. 与评测数据的重叠：按“规范化后的材料全文”和“长片段（≥ 60 字）”两种方式比对 --eval 给出的文件。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
import statistics as st
from collections import Counter, defaultdict

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.schema import read_jsonl  # noqa: E402

CJK = re.compile(r"[一-鿿]")
WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", text.lower()).strip()


def fragments(text: str, n: int = 60) -> set[str]:
    """把材料按句子 / 行切开，保留规范化后至少 n 个字符的片段，用来发现“部分重叠”。"""
    parts = re.split(r"[\n。！？!?]|(?<=[.;])\s", text)
    return {norm(p) for p in parts if len(norm(p)) >= n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = read_jsonl(a.train)
    by = defaultdict(list)
    for d in rows:
        by[d.source].append(d)
    lines = [f"# 数据审计：{a.train}", "", f"共 {len(rows)} 道题，{len(by)} 个数据源。", ""]

    # 1–4. 每个数据源的格式、标签、语言、长度
    lines += ["## 1. 各数据源概况", "",
              "| 数据源 | 题数 | 格式错误 | 最常见答案占比 | 是/否题“是”占比 | 软标签 | 标注语言 | 中文字符占比 | 材料中位字符 | 最长 | 空材料 | 标记 |",
              "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    flags_all = []
    for src, ds in sorted(by.items(), key=lambda x: -len(x[1])):
        bad = 0
        for d in ds:
            try:
                d.validate()
                if len(d.options) > 255 or not d.instructions.strip():
                    bad += 1
            except ValueError:
                bad += 1
        gold = Counter(d.options[d.gold_index].name for d in ds)
        top = gold.most_common(1)[0][1] / len(ds)
        nouls = [d for d in ds if d.type == "noul"]
        yes = sum(d.options[d.gold_index].name == "true" for d in nouls) / len(nouls) if nouls else None
        soft = sum(max(d.target) < 0.999 for d in ds) / len(ds)
        langs = Counter(d.lang for d in ds).most_common(1)[0][0]
        cjk = sum(len(CJK.findall(d.state)) for d in ds) / max(1, sum(len(d.state) for d in ds))
        lens = sorted(len(d.state) for d in ds)
        empty = sum(not d.state.strip() for d in ds) / len(ds)
        flags = []
        if bad:
            flags.append("格式错误")
        if top > 0.6 and len(gold) > 1:
            flags.append("答案偏斜")
        if yes is not None and not 0.25 <= yes <= 0.75:
            flags.append("是/否失衡")
        if langs == "zh" and cjk < 0.1 and lens[len(lens) // 2] > 0:
            flags.append("标中文但少中文")
        if langs == "en" and cjk > 0.2:
            flags.append("标英文但多中文")
        if empty > 0.5:
            flags.append("材料多为空")
        if flags:
            flags_all.append((src, flags))
        lines.append(f"| {src} | {len(ds)} | {bad} | {top:.0%} | {'-' if yes is None else f'{yes:.0%}'} | {soft:.0%} | "
                     f"{langs} | {cjk:.0%} | {lens[len(lens) // 2]} | {lens[-1]} | {empty:.0%} | {'、'.join(flags)} |")

    # 5. 重复与近似重复
    exact = Counter(hashlib.sha1((d.state + "\0" + d.instructions + "\0" + "|".join(o.name for o in d.options)).encode())
                    .hexdigest() for d in rows)
    n_exact = sum(v - 1 for v in exact.values() if v > 1)
    near = defaultdict(list)
    for d in rows:
        s = norm(d.state)
        if len(s) >= 80:
            near[s[:300]].append(d)
    near_groups = [g for g in near.values() if len({x.gold_index for x in g}) > 0 and len(g) > 1]
    # 标注冲突：材料全文、问题、选项都相同，但标准答案不同
    same = defaultdict(set)
    for d in rows:
        same[(d.state, d.instructions, tuple(o.name for o in d.options))].add(d.options[d.gold_index].name)
    conflict = [k for k, v in same.items() if len(v) > 1]
    big = sorted(near_groups, key=len, reverse=True)[:8]
    lines += ["", "## 2. 重复", "", f"- 完全重复（材料 + 问题 + 选项都相同）：{n_exact} 道",
              f"- 材料前 300 字相同的组：{len(near_groups)} 组，涉及 {sum(len(g) for g in near_groups)} 道"
              "（同一份材料配多个问题是正常的，例如 bev / typed / 生成题）",
              f"- 材料全文、问题、选项都相同但答案不同（标注冲突）：{len(conflict)} 组", "",
              "最大的几组（来源、组大小）：" + "；".join(f"{Counter(x.source for x in g).most_common(1)[0][0]} × {len(g)}" for g in big)]

    # 6. 与评测数据的重叠
    lines += ["", "## 3. 与评测数据的重叠", "",
              "| 评测文件 | 题数 | 材料全文相同 | 共享 ≥60 字片段 | 重叠的训练来源 |", "|---|---:|---:|---:|---|"]
    full = defaultdict(set)
    frag = defaultdict(set)
    for d in rows:
        if d.state.strip():
            full[norm(d.state)].add(d.source)
            for f in fragments(d.state):
                frag[f].add(d.source)
    for pattern in a.eval:
        for path in sorted(glob.glob(pattern)):
            ev = read_jsonl(path)
            hit_full = [e for e in ev if e.state.strip() and norm(e.state) in full]
            hit_frag = [e for e in ev if e.state.strip() and fragments(e.state) & frag.keys()]
            srcs = Counter()
            for e in hit_frag:
                for f in fragments(e.state) & frag.keys():
                    srcs.update(frag[f])
            lines.append(f"| {path} | {len(ev)} | {len(hit_full)} | {len(hit_frag)} | "
                         f"{', '.join(f'{s}({n})' for s, n in srcs.most_common(4))} |")

    lines += ["", "## 4. 需要关注的数据源", ""] + [f"- **{s}**：{'、'.join(f)}" for s, f in flags_all]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
