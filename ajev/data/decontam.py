"""去污染：删掉与 Jev Decision Index 排行榜评测数据有文本重叠的训练题。

    from ajev.data.decontam import load_index, contaminated
    index = load_index("data/decontam_index.json")       # 第一次运行会下载评测数据并缓存
    clean = [d for d in train if not contaminated(d, index)]

为什么需要：只按“用了哪个 split”判断不够。审计时用排行榜真正的评测数据逐题比对（scripts/check_leaderboard_overlap.py），
发现几类“不同 split 但文本重叠”的情况：
- ContractNLI：官方训练集里有和测试合同几乎一样的模板合同（12/122 份测试合同超过一半的句子出现在训练集里），
  bev_hard 里也有少量相同的合同条款；
- ANLI：训练和测试共用同一段前提文字（假设不同）；
- RAGTruth：个别模型回答与测试集逐字相同；
- 零星的：bev_hard 里有 MMLU 测试题、HellaSwag 的通用短句、BANKING77 的问句。

判定规则（满足任意一条就删）：
1. 训练题的任一文本（材料里的字符串、问题、选项说明）**整段**等于某条评测文本（≥ 20 字）；
2. 训练题与评测数据共享 ≥ 60 字的句子：
   - 对 ContractNLI、HellaSwag（含大量标准合同条款、WikiHow 通用句）：共享句子占训练题句子的 **≥ 20%** 才删；
   - 对其他基准（题目、新闻、前提文字都有区分度）：共享**任何一句**就删
     （第一版只用 20% 阈值，复查时发现 RAGTruth 测试新闻的部分句子、一道 MMLU 测试题仍留在训练集里）；
3. 程序生成的题（gen_*）不检查：内容由我们的代码生成，不可能来自评测数据。
"""

from __future__ import annotations

import json
import os
import re

WS = re.compile(r"\s+")
SPLIT = re.compile(r"[\n。！？!?]|(?<=[.;])\s")
FRAG_SHARE = 0.2
LENIENT = {"contractnli", "hellaswag"}  # 这些评测集的句子多为标准条款 / 通用句，用 20% 阈值

PQ = "hf://datasets/kiddothe2b/contract-nli@refs%2Fconvert%2Fparquet/contractnli_b/test/0000.parquet"
# 排行榜评测数据（与我们训练数据源相关的部分）：(加载参数, 文本字段)
EVAL_SETS = {
    "hellaswag": (("Rowan/hellaswag", None, "validation"), ["ctx", "endings"]),
    "winogrande": (("allenai/winogrande", "winogrande_xl", "validation"), ["sentence"]),
    "gsm8k": (("openai/gsm8k", "main", "test"), ["question"]),
    "ragtruth": (("wandb/RAGTruth-processed", None, "test"), ["context", "output"]),
    "contractnli": (("parquet", {"test": PQ}, "test"), ["premise"]),
    "humicroedit": (("tasksource/humicroedit", "subtask-2", "test"), ["original1", "original2"]),
    "isarcasm": (("viethq1906/isarcasm_2022_taskA_En", None, "test"), ["sentence"]),
    "acos": (("NEUDM/acos", None, "test"), ["input"]),
    "newyorker": (("jmhessel/newyorker_caption_contest", "matching", "test"), ["image_description", "caption_choices"]),
    "anli_r1": (("facebook/anli", "plain_text", "test_r1"), ["premise", "hypothesis"]),
    "anli_r2": (("facebook/anli", "plain_text", "test_r2"), ["premise", "hypothesis"]),
    "anli_r3": (("facebook/anli", "plain_text", "test_r3"), ["premise", "hypothesis"]),
    "banking77": (("mteb/banking77", None, "test"), ["text"]),
    "clinc150": (("clinc/clinc_oos", "plus", "test"), ["text"]),
    "when2call": (("nvidia/When2Call", "test", "mcq"), ["question"]),
    "mmlu": (("cais/mmlu", "all", "test"), ["question"]),
    "mmlu_pro": (("TIGER-Lab/MMLU-Pro", None, "test"), ["question"]),
}


def norm(t: str) -> str:
    return WS.sub(" ", str(t).lower()).strip()


def frags(t: str) -> set[str]:
    return {norm(p) for p in SPLIT.split(t) if len(norm(p)) >= 60}


def leaves(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in leaves(v)]
    if isinstance(x, list):
        return [s for v in x for s in leaves(v)]
    return []


def decision_texts(d) -> list[str]:
    try:
        st = json.loads(d.state)
        out = leaves(st) if isinstance(st, (dict, list)) else [d.state]
    except (json.JSONDecodeError, TypeError):
        out = [d.state]
    return out + [d.instructions] + [o.desc for o in d.options if o.desc]


def build_index() -> dict[str, list[str]]:
    from datasets import Image, load_dataset

    full, frag, strict = set(), set(), set()
    for name, ((path, cfg, split), fields) in EVAL_SETS.items():
        ds = (load_dataset("parquet", data_files=cfg, split=split) if path == "parquet"
              else load_dataset(path, cfg, split=split))
        img = [c for c, f in ds.features.items() if isinstance(f, Image)]
        if img:
            ds = ds.remove_columns(img)
        for row in ds:
            for f in fields:
                for t in leaves(row.get(f)):
                    n = norm(t)
                    if len(n) >= 20:
                        full.add(n)
                    fs = frags(t)
                    frag |= fs
                    if name not in LENIENT:
                        strict |= fs
        print(f"[decontam] {name}: {len(ds)} eval rows", flush=True)
    return {"full": sorted(full), "frag": sorted(frag), "strict": sorted(strict)}


def load_index(path: str) -> dict[str, set[str]]:
    """读取评测文本索引；不存在时下载评测数据构建并缓存到 path。"""
    if os.path.exists(path):
        with open(path) as f:
            if "strict" not in json.load(f):  # 旧版索引：重建
                os.remove(path)
    if not os.path.exists(path):
        idx = build_index()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(idx, f)
    with open(path) as f:
        idx = json.load(f)
    return {"full": set(idx["full"]), "frag": set(idx["frag"]), "strict": set(idx["strict"])}


def contaminated(d, index: dict[str, set[str]]) -> bool:
    if d.source.startswith("gen_"):
        return False
    texts = decision_texts(d)
    if any(len(norm(t)) >= 20 and norm(t) in index["full"] for t in texts):
        return True
    fs = set().union(*(frags(t) for t in texts)) if texts else set()
    if fs & index["strict"]:
        return True
    return bool(fs) and len(fs & index["frag"]) / len(fs) >= FRAG_SHARE
