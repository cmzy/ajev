"""补充 Jev Decision Index 排行榜各领域的训练数据——只用各数据集中**不被排行榜评测**的部分。

为什么：我们的训练集偏重业务决策，排行榜考的知识推理、语言理解（常识、合同 / 临床 NLI、幻觉、讽刺、方面情感）、
幽默等领域几乎没有覆盖（对比见对话记录）。这里把这些领域的**训练集**转换成决策题。

**绝不训练排行榜的评测题**。每个数据源都核对过排行榜用的是哪一份（来源：排行榜公开的 methodology.json
与 decision-index 仓库的 manifest），这里只用其余部分：

======================  =======================================  ==========================================
数据源                   排行榜评测的部分                            这里训练用的部分
======================  =======================================  ==========================================
HellaSwag               validation（10,042）                        train（评测用 train 里留出的 10%）
WinoGrande              dev（1,267）                                xl train（同上）
GSM8K                   test（1,319，配干扰选项）                    train（自己生成数字干扰选项）
MMLU auxiliary_train    —（MMLU / MMLU-Pro 的 test 绝不使用）         auxiliary_train
RAGTruth                test（2,700）                               train（核实过：与 test 无共享原文）
ContractNLI             contractnli_b test（2,091）                  contractnli_b train / validation
Humicroedit             subtask-2 test                             subtask-2 train / validation
NLI4CT                  官方 gold_test（5,500）                      train / validation（核实过：无共享陈述）
iSarcasmEval            官方 test（任务 A 英文 1,400 等）            任务 A 英文 train / validation
ACOS                    test 评论                                   train / validation
New Yorker 配图标题      matching fold-0 test（528）                 matching fold-0 train / validation（无共享漫画）
ANLI                    test r1+r2+r3（3,200）                      train_r1、train_r2（r3 已在 sources.py）/ dev
钓鱼邮件                 PhishNChips core（2,000）                   zefang-liu/phishing-email-dataset（不同数据集）
======================  =======================================  ==========================================

暂未使用：SuperGPQA（约 1,000 题与 MMLU-Pro 测试集逐字重复，需先去重）、Amazon ESCI（训练与测试的查询是否
不重叠尚未核实）、BBH / HoVer / BFCL 等（需要额外去重或证据库）。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ajev.data.sources import Ctx, Source, _choice, _noul, _nli, state_to_text
from ajev.schema import Decision, Option

Row = dict[str, Any]


def _clip(text: str, limit: int) -> str:
    """过长时保留开头约 70% 和结尾约 30%（与 more_sources._clip 相同）。"""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + "…" + text[len(text) - (limit - head - 1):]


def _parse(x: Any) -> Any:
    """有些字段在数据集里是“列表 / 字典的字符串形式”，这里统一解析成 Python 对象。"""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return x
    return x


# ---- 1. HellaSwag：给出情境开头，选最合理的后续 -------------------------------------------------
T_HELLA = {"en": ["Which ending most plausibly continues the description?", "What happens next?"],
           "zh": ["哪个结尾最合理地接续了这段描述？", "接下来最可能发生什么？"]}


def conv_hellaswag(row: Row, i: int, ctx: Ctx) -> Decision | None:
    endings = _parse(row.get("endings"))
    if not isinstance(endings, list) or len(endings) != 4 or str(row.get("label", "")) == "":
        return None
    state = state_to_text({"activity": row.get("activity_label") or "", "context": row["ctx"]})
    opts = [Option(f"ending_{k + 1}", e.strip()) for k, e in enumerate(endings)]
    return _choice(ctx, i, state, ctx.pick(T_HELLA, ctx.instr_lang()), opts, int(row["label"]))


# ---- 2. WinoGrande：代词 / 空格指代消解 ---------------------------------------------------------
T_WINO = {"en": ["Which option correctly fills the blank (_)?", "Who or what does the blank (_) refer to?"],
          "zh": ["哪个选项能正确填入空格（_）？", "空格（_）指的是谁或什么？"]}


def conv_winogrande(row: Row, i: int, ctx: Ctx) -> Decision | None:
    o1, o2, ans = row.get("option1"), row.get("option2"), str(row.get("answer", ""))
    if not o1 or not o2 or o1 == o2 or ans not in ("1", "2"):
        return None
    opts = [Option("option_1", o1), Option("option_2", o2)]
    return _choice(ctx, i, row["sentence"], ctx.pick(T_WINO, ctx.instr_lang()), opts, int(ans) - 1)


# ---- 3. GSM8K：数学应用题，从 4 个数字里选最终答案（干扰项由程序生成）---------------------------
T_GSM = {"en": ["What is the final numerical answer?", "Which number answers the question?"],
         "zh": ["这道题的最终答案是多少？", "哪个数字是这道题的答案？"]}


def _distractors(n: float, rng) -> list[float]:
    """常见的错误答案：差一步、乘除错、少算一项、数位错等。"""
    cands = {n + 1, n - 1, n * 2, n + 10, n - 10, n + 5, round(n * 1.5), round(n / 2), n * 10, n + 2}
    if n == int(n) and abs(n) >= 10:
        s = str(int(abs(n)))
        cands.add(float(s[::-1]) * (1 if n >= 0 else -1))
    cands = [c for c in cands if c != n and c >= 0]
    rng.shuffle(cands)
    return cands[:3]


def conv_gsm8k(row: Row, i: int, ctx: Ctx) -> Decision | None:
    m = re.search(r"####\s*([-\d,\.]+)", row.get("answer") or "")
    if not m:
        return None
    try:
        n = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    fmt = lambda x: str(int(x)) if x == int(x) else f"{x:g}"
    nums = [n] + _distractors(n, ctx.rng)
    if len({fmt(x) for x in nums}) != 4:
        return None
    opts = [Option(fmt(x)) for x in nums]
    return _choice(ctx, i, row["question"], ctx.pick(T_GSM, ctx.instr_lang()), opts, 0)


# ---- 4. MMLU auxiliary_train：多学科选择题（来源 ARC / OBQA / RACE 等，不含 MMLU 测试题）--------------
T_MCQ = {"en": ["Which option correctly answers the question?", "Choose the correct answer."],
         "zh": ["哪个选项正确回答了这个问题？", "选出正确答案。"]}


def conv_mmlu_aux(row: Row, i: int, ctx: Ctx) -> Decision | None:
    r = row.get("train") if isinstance(row.get("train"), dict) else row
    choices, ans = r.get("choices"), r.get("answer")
    if not r.get("question") or not isinstance(choices, list) or len(choices) < 2 or ans is None:
        return None
    if len(set(choices)) != len(choices) or not 0 <= int(ans) < len(choices):
        return None
    opts = [Option(f"option_{k + 1}", c) for k, c in enumerate(choices)]
    return _choice(ctx, i, _clip(r["question"], 6000), ctx.pick(T_MCQ, ctx.instr_lang()), opts, int(ans))


# ---- 5. RAGTruth：回答里有没有原文不支持的内容（幻觉检测）-----------------------------------------
T_RAG = {"en": ["The response contains information that is not supported by, or contradicts, the provided context.",
                "The response includes hallucinated content."],
         "zh": ["这个回答包含了所给材料不支持或与之矛盾的内容。", "这个回答存在幻觉（编造的内容）。"]}


def conv_ragtruth(row: Row, i: int, ctx: Ctx) -> Decision | None:
    if not row.get("output") or not row.get("context"):
        return None
    labels = _parse(row.get("hallucination_labels"))
    halluc = bool(labels) and labels != "[]"
    state = state_to_text({"task": row.get("task_type") or "", "instruction": row.get("query") or "",
                           "context": _clip(str(row["context"]), 8000), "response": _clip(row["output"], 3000)})
    lang = ctx.instr_lang()
    return _noul(ctx, i, state, ctx.pick(T_RAG, lang), halluc, lang)


# ---- 6. ContractNLI：保密协议全文 + 一条假设 → 蕴含 / 矛盾 / 未提及 ------------------------------
T_CNLI = {"en": ["What does the contract say about this statement: \"{h}\"",
                 "Does the agreement support, contradict, or not mention: \"{h}\""],
          "zh": ["合同对这条陈述是什么态度：“{h}”", "这份协议是支持、否定还是没有提到：“{h}”"]}
CNLI_DESC = {"entailment": "The contract supports the statement.",
             "contradiction": "The contract contradicts the statement.",
             "neutral": "The contract does not mention it."}


def conv_contractnli(row: Row, i: int, ctx: Ctx) -> Decision | None:
    names = ctx.label_names
    try:
        gold = names[int(row["label"])]
    except (ValueError, IndexError, TypeError):
        return None
    if gold not in CNLI_DESC:
        return None
    opts = [Option(k, v) for k, v in CNLI_DESC.items()]
    instr = ctx.pick(T_CNLI, ctx.instr_lang(), h=row["hypothesis"])
    return _choice(ctx, i, _clip(row["premise"], 24000), instr, opts, list(CNLI_DESC).index(gold))


# ---- 7. Humicroedit：标题改一个词，哪一个更好笑 -----------------------------------------------
T_FUNNY = {"en": ["Which edited headline is funnier?", "Which of the two edited headlines would readers find funnier?"],
           "zh": ["哪个改过的标题更好笑？", "两个改过一个词的新闻标题，哪个更搞笑？"]}


def _edit(headline: str, word: str) -> str:
    return re.sub(r"<[^>]*/>", word, headline).strip().strip('"').strip()


def conv_humicroedit(row: Row, i: int, ctx: Ctx) -> Decision | None:
    try:
        g1, g2 = float(row["meanGrade1"]), float(row["meanGrade2"])
    except (KeyError, TypeError, ValueError):
        return None
    if g1 == g2:  # 打平的题排行榜也不计分
        return None
    h1, h2 = _edit(row["original1"], row["edit1"]), _edit(row["original2"], row["edit2"])
    if h1 == h2:
        return None
    state = state_to_text({"headline_1": h1, "headline_2": h2})
    opts = [Option("headline_1", h1), Option("headline_2", h2)]
    return _choice(ctx, i, state, ctx.pick(T_FUNNY, ctx.instr_lang()), opts, 0 if g1 > g2 else 1)


# ---- 8. NLI4CT：临床试验报告 + 陈述 → 是否成立 ------------------------------------------------
def _ct_section(ct: Any, section: str) -> str:
    ct = _parse(ct)
    if not isinstance(ct, dict):
        return ""
    part = ct.get(section) or []
    return "\n".join(s.strip() for s in part) if isinstance(part, list) else str(part)


def conv_nli4ct(row: Row, i: int, ctx: Ctx) -> Decision | None:
    label = row.get("Label")
    if label not in ("Entailment", "Contradiction") or not row.get("Statement"):
        return None
    sec = row.get("Section_id") or ""
    st: dict[str, str] = {"section": sec, "primary_trial": _clip(_ct_section(row.get("Primary_ct"), sec), 6000)}
    if row.get("Secondary_ct"):
        st["secondary_trial"] = _clip(_ct_section(row.get("Secondary_ct"), sec), 6000)
    lang = ctx.instr_lang()
    return _noul(ctx, i, state_to_text(st), row["Statement"], label == "Entailment", lang)


# ---- 9. iSarcasmEval：推文是否在讽刺 ------------------------------------------------------------
T_SARC = {"en": ["This tweet is sarcastic.", "The author is being sarcastic."],
          "zh": ["这条推文在讽刺。", "作者是在说反话（讽刺）。"]}


def conv_isarcasm(row: Row, i: int, ctx: Ctx) -> Decision | None:
    if not row.get("sentence") or str(row.get("sentiment")) not in ("0", "1"):
        return None
    lang = ctx.instr_lang()
    return _noul(ctx, i, row["sentence"], ctx.pick(T_SARC, lang), str(row["sentiment"]) == "1", lang)


# ---- 10. ACOS：评论里对某个方面（类别）的情感 ------------------------------------------------------
SENTS = ["positive", "negative", "neutral"]


def conv_acos(row: Row, i: int, ctx: Ctx) -> Decision | None:
    sent = _parse(row.get("input"))
    sent = sent[0] if isinstance(sent, list) and sent else sent
    quads = _parse(row.get("output"))
    if not isinstance(sent, str) or not isinstance(quads, list) or not quads:
        return None
    quads = [q for q in quads if isinstance(q, (list, tuple)) and len(q) >= 3]
    if not quads:
        return None
    pairs = {(q[1], q[2]) for q in quads}
    cat, sentiment = ctx.rng.choice(sorted(pairs))
    if ctx.rng.random() < 0.5:  # 一半是正确的（类别, 情感），一半把情感换成别的、且这个组合确实不存在
        others = [s for s in SENTS if (cat, s) not in pairs]
        if not others:
            return None
        sentiment, ans = ctx.rng.choice(others), False
    else:
        ans = True
    lang = ctx.instr_lang()
    aspect = cat.replace("#", " / ").replace("_", " ")
    instr = (f"The review expresses {sentiment} sentiment about \"{aspect}\"." if lang == "en"
             else f"这条评论对“{aspect}”表达了{ {'positive': '正面', 'negative': '负面', 'neutral': '中性'}[sentiment] }情感。")
    return _noul(ctx, i, sent, instr, ans, lang)


# ---- 11. New Yorker 漫画配图标题：5 个标题里哪个是为这幅漫画写的 ----------------------------------
T_NYC = {"en": ["Which caption was written for this cartoon?", "Which caption matches this cartoon?"],
         "zh": ["哪个标题是为这幅漫画写的？", "哪条配文和这幅漫画相配？"]}


def conv_newyorker(row: Row, i: int, ctx: Ctx) -> Decision | None:
    caps = _parse(row.get("caption_choices"))
    label = str(row.get("label", ""))
    if not isinstance(caps, list) or len(caps) != 5 or label not in "ABCDE" or not label:
        return None
    state = state_to_text({"location": row.get("image_location") or "", "scene": row.get("image_description") or "",
                           "what is unusual": row.get("image_uncanny_description") or ""})
    opts = [Option(f"caption_{k + 1}", c) for k, c in enumerate(caps)]
    return _choice(ctx, i, state, ctx.pick(T_NYC, ctx.instr_lang()), opts, "ABCDE".index(label))


# ---- 12. 钓鱼邮件 --------------------------------------------------------------------------------
T_PHISH = {"en": ["This email is a phishing attempt.", "This message is trying to phish the recipient."],
           "zh": ["这封邮件是钓鱼邮件。", "这封邮件在试图骗取收件人的信息或钱财。"]}


def conv_phishing(row: Row, i: int, ctx: Ctx) -> Decision | None:
    text, kind = row.get("Email Text"), row.get("Email Type")
    if not text or kind not in ("Safe Email", "Phishing Email") or len(text.strip()) < 20:
        return None
    lang = ctx.instr_lang()
    return _noul(ctx, i, _clip(text, 6000), ctx.pick(T_PHISH, lang), kind == "Phishing Email", lang)


# ---- 注册表 -------------------------------------------------------------------------------------
# Source 参数：内部名、HF 路径、config、训练 split、评测 split、语言、转换器。
# 评测 split 一律选**排行榜不用**的部分：排行榜用了 validation / test 的，就从 train 里留出（"train#eval"）。
BENCH_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("hellaswag", "Rowan/hellaswag", None, "train#train", "train#eval", "en", conv_hellaswag),
        Source("winogrande", "allenai/winogrande", "winogrande_xl", "train#train", "train#eval", "en", conv_winogrande),
        Source("gsm8k", "openai/gsm8k", "main", "train#train", "train#eval", "en", conv_gsm8k),
        Source("mmlu_aux", "cais/mmlu", "auxiliary_train", "train#train", "train#eval", "en", conv_mmlu_aux),
        Source("ragtruth", "wandb/RAGTruth-processed", None, "train#train", "train#eval", "en", conv_ragtruth),
        # 这个数据集用的是 datasets 已不再支持的加载脚本，所以直接读 Hugging Face 自动转换好的 parquet 文件。
        Source("contractnli", "parquet", None, "train", "validation", "en", conv_contractnli,
               data_files={sp: f"hf://datasets/kiddothe2b/contract-nli@refs%2Fconvert%2Fparquet/contractnli_b/{sp}/0000.parquet"
                           for sp in ("train", "validation")}),
        Source("humicroedit", "tasksource/humicroedit", "subtask-2", "train", "validation", "en", conv_humicroedit),
        Source("nli4ct", "tasksource/nli4ct", None, "train", "validation", "en", conv_nli4ct),
        Source("isarcasm", "viethq1906/isarcasm_2022_taskA_En", None, "train", "validation", "en", conv_isarcasm,
               label_field="sentiment"),
        Source("acos", "NEUDM/acos", None, "train", "validation", "en", conv_acos),
        Source("newyorker", "jmhessel/newyorker_caption_contest", "matching", "train", "validation", "en",
               conv_newyorker),
        Source("anli_r1", "facebook/anli", "plain_text", "train_r1", "dev_r1", "en", _nli("premise", "hypothesis")),
        Source("anli_r2", "facebook/anli", "plain_text", "train_r2", "dev_r2", "en", _nli("premise", "hypothesis")),
        Source("phishing", "zefang-liu/phishing-email-dataset", None, "train#train", "train#eval", "en",
               conv_phishing, label_field="Email Type"),
    ]
}

# 每个数据源加入 lm3 的题数（合计约 1.4 万道）
BENCH_QUOTAS = {"hellaswag": 1500, "winogrande": 1500, "gsm8k": 1500, "mmlu_aux": 2000, "ragtruth": 1000,
                "contractnli": 800, "humicroedit": 800, "nli4ct": 800, "isarcasm": 800, "acos": 800,
                "newyorker": 800, "anli_r1": 500, "anli_r2": 500, "phishing": 800}
