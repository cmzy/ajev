"""第二批数据源：业务决策、打分题、软标签、中文、对抗性推理。

在 AJev 流程中的位置：和 ``ajev/data/sources.py`` 一样，``ajev.data.build`` 会把这里的
``MORE_SOURCES`` 和原有的 ``SOURCES`` 合并，统一转换成 Decision、采样、去重、写入 JSONL。

为什么要加这一批（对应第一版模型 sft2 的短板）：

=====================  ========================================================================
短板                    新增数据源
=====================  ========================================================================
业务决策场景太少          bev-decision（5 个子集）、客服工单分诊、When2Call（agent 该怎么回应）
打分题弱（0.52）         Feedback-Collection（每级都有文字描述）、HelpSteer3（7 级偏好）、PKU 危害程度、
                        中文 UltraFeedback（中文多维评分）
软标签少                 HelpSteer3（每个标注者的打分 → 分布）、bev-decision skills（label_probs）
中文数据少               ToxiCN、COLD（中文冒犯言论）、中文 UltraFeedback、HelpSteer3 的中文部分
对抗性推理弱（ANLI 0.49） WANLI、bev-decision hard_50k / counterfactual_15k
=====================  ========================================================================

给初学者的几点说明：

1. **一行原始数据可以产出多道题。** 例如一张客服工单，我们同时问“该分到哪个队列”“是什么类型”
   “优先级多高”。这 3 道题共享同一份材料（state），和 Jev 的用法完全一样：一份状态、多个问题。
   这类题用相同的 ``group`` 标记，id 是 ``<数据源>/<行号>/<问题名>``。

2. **Jev 格式的数据可以直接读。** bev-decision 每行就是“state + 一组 Jev 问题（含标准答案）”，
   和主测试集 typed-decisions 的格式几乎一样，所以直接复用 ``schema.decisions_from_jev``。

3. **软标签从“多个标注者的意见”来。** HelpSteer3 里每条数据有好几个标注者各自打的分，
   我们把这些分数统计成一个分布作为 target。例如 3 个人分别打了 +1、+1、+2（7 级量表），
   target 就是 “+1 级 2/3、+2 级 1/3”，而不是只取平均或多数票。这样模型学到的是
   “这道题本身就有争议”，概率更诚实。

4. **只有一个 split 的数据集**，写成 ``"train#train"`` / ``"train#eval"``：先用固定种子打乱，
   再切出训练和评测两部分（比例由 ``holdout`` 指定），两部分不重叠、分布一致。
   不能直接按原始顺序切（例如 ``"train[90%:]"``），因为很多数据集是排好序的，详见 sources.py 的 Source 说明。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ajev.data.sources import (
    NLI3_OPTS,
    T_ENTAIL,
    T_NLI3,
    Ctx,
    Row,
    Source,
    _choice,
    _noul,
    _score,
)
from ajev.schema import Decision, Option, decisions_from_jev, normalize, one_hot, state_to_text


def _regroup(ds: list[Decision], ctx: Ctx, i: int, names: list[str]) -> list[Decision]:
    """把同一行产出的多道题标记为同一组，并给每道题一个唯一 id：``<数据源>/<行号>/<问题名>``。

    ``_noul`` / ``_choice`` / ``_score`` 生成的 id 都是 ``<数据源>/<行号>``，一行出多道题时会重复，
    所以在这里统一改写。
    """
    group = f"{ctx.source.name}/{i}"
    for d, name in zip(ds, names):
        d.id, d.group = f"{group}/{name}", group
    return ds


def _valid(ds: list[Decision]) -> list[Decision]:
    """只保留格式合法的题。

    原始数据偶尔有不合规的题（比如打分题超过 10 级、选项重名），``validate()`` 会抛出 ValueError。
    在转换器里提前过滤掉，避免一道坏题让整个数据构建中断。
    """
    out = []
    for d in ds:
        try:
            d.validate()
            out.append(d)
        except ValueError:
            pass
    return out


# ==== 1. bev-decision：Jev 格式，直接读 =====================================================


def conv_bev(row: Row, i: int, ctx: Ctx) -> list[Decision]:
    """bev-decision 的一行 → 若干道题（每个 Jev 问题一道）。

    原始格式（questions_json 是一个 JSON 字符串）::

        {"citation_intent": {"type": "choice",
                             "instructions": "What is the intent of this citation?",
                             "criteria": {"background": "...", "method": "...", "result": "..."},
                             "label": "background"}}

    标准答案的三种写法：
    - noul：``"label": true / false`` → 选项名是 "true" / "false"；
    - choice：``"label": "background"``，有的题还带 ``"label_probs": {...}``（软标签，skills 子集）；
    - score：``"label": 2``（第几级，从 0 开始）→ 选项名是 "2"；软标签是列表 ``[p0, p1, ...]``。

    做法：把每道题的答案统一成“选项名 → 概率”的字典（有 label_probs 就用它，否则是 one-hot），
    交给 ``decisions_from_jev`` 展开成 Decision。
    """
    questions = json.loads(row["questions_json"])
    gold: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        if q.get("label_probs"):
            probs = q["label_probs"]
            # 打分题的软标签是按等级顺序排列的列表，例如 [0.1, 0.7, 0.2]；
            # 而 Jev 格式里打分题的选项名是 "0"、"1"、"2"…，所以转成 {"0": 0.1, "1": 0.7, "2": 0.2}。
            if isinstance(probs, list):
                probs = {str(k): p for k, p in enumerate(probs)}
        else:
            label = q["label"]
            # bool 必须先判断：在 Python 里 True 也是 int（isinstance(True, int) 为 True）。
            name = ("true" if label else "false") if isinstance(label, bool) else str(label)
            probs = {name: 1.0}
        gold[qid] = {"probabilities": probs}
    ds = decisions_from_jev(row["state"], questions, group=f"{ctx.source.name}/{i}",
                            source=ctx.source.name, gold=gold)
    # 答案里出现了选项中不存在的名字时，normalize 后会变成全 0 → 均匀分布，这种题没有意义，丢掉。
    ds = [d for d in ds if max(d.target) > 1.0 / len(d.options) + 1e-9]
    for d in ds:
        d.meta["domain"] = row.get("domain")
    return _valid(ds)


# ==== 2. 客服工单分诊（Tobi-Bueck/customer-support-tickets）==================================

T_QUEUE = {
    "en": ["Which support queue should handle this ticket?", "Route this ticket to the right team."],
    "zh": ["这张工单应该分派给哪个支持队列？", "把这张工单路由到正确的团队。"],
}
T_TTYPE = {
    "en": ["What kind of ticket is this?", "Classify this ticket by ITIL type."],
    "zh": ["这是哪一类工单？", "按 ITIL 类型给这张工单分类。"],
}
TTYPE_DESC = {
    "Incident": "Something that worked has broken or is degraded (unplanned interruption).",
    "Request": "The customer asks for information, access or a standard service.",
    "Problem": "An underlying or recurring cause behind one or more incidents needs investigation.",
    "Change": "The customer asks to modify a system, configuration or plan.",
}
T_PRIORITY = {
    "en": ["How urgent is this ticket?", "What priority should this ticket get?"],
    "zh": ["这张工单有多紧急？", "这张工单应该定为什么优先级？"],
}
PRIORITY_LEVELS = {
    "en": ["Low: minor issue or general question; can wait.",
           "Medium: noticeable impact on the customer; handle in the normal queue.",
           "High: severe impact, outage, security or many users affected; handle immediately."],
    "zh": ["低：小问题或一般咨询，可以等待。", "中：对客户有明显影响，按正常队列处理。",
           "高：影响严重、服务中断、安全问题或大量用户受影响，需立即处理。"],
}
PRIORITY_INDEX = {"low": 0, "medium": 1, "high": 2}


def conv_ticket(row: Row, i: int, ctx: Ctx) -> list[Decision] | None:
    """一张工单 → 3 道题：分到哪个队列（choice）、工单类型（choice）、优先级（score）。

    举个例子：subject="Account Disruption"，body="...the account portal appears to be offline..."，
    queue="Technical Support"，type="Incident"，priority="high"
        → choice：10 个队列里选 "Technical Support"
        → choice：4 种类型里选 "Incident"
        → score：低 / 中 / 高 三级里选第 2 级（高）
    工单有英文也有德文，mmBERT 是多语言模型，可以直接读。
    """
    if not (row.get("body") and row.get("queue") and row.get("type") and row.get("priority")):
        return None
    if row["priority"] not in PRIORITY_INDEX or row["type"] not in TTYPE_DESC:
        return None
    state = state_to_text({"subject": row.get("subject") or "", "body": row["body"]})
    lang = ctx.instr_lang()
    queues = ctx.label_names  # 数据里出现过的全部队列名（由 Source.label_field="queue" 推断）
    ds = [
        _choice(ctx, i, state, ctx.pick(T_QUEUE, lang), [Option(q) for q in queues], queues.index(row["queue"])),
        _choice(ctx, i, state, ctx.pick(T_TTYPE, lang), [Option(t, d) for t, d in TTYPE_DESC.items()],
                list(TTYPE_DESC).index(row["type"])),
        _score(ctx, i, state, ctx.pick(T_PRIORITY, lang), PRIORITY_LEVELS[lang],
               one_hot(3, PRIORITY_INDEX[row["priority"]])),
    ]
    return _regroup(ds, ctx, i, ["queue", "type", "priority"])


# ==== 3. 带评分细则的打分（prometheus-eval/Feedback-Collection）===============================


def conv_feedback(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """一条“指令 + 回答 + 评分细则” → 一道 5 级打分题，每一级的文字描述直接来自数据集。

    这正是 Jev 的 score 题型：问题是评分标准（``orig_criteria``），5 个等级各有一段具体描述
    （``orig_score1_description`` … ``orig_score5_description``），答案是第几级。
    state 里还放了参考答案，帮助模型判断回答质量。
    """
    try:
        score = int(row["orig_score"])
    except (TypeError, ValueError):
        return None
    levels = [row[f"orig_score{k}_description"] for k in range(1, 6)]
    if not (1 <= score <= 5) or not all(levels) or not row.get("orig_criteria"):
        return None
    state = state_to_text({"instruction": row["orig_instruction"], "response": row["orig_response"],
                           "reference_answer": row.get("orig_reference_answer") or ""})
    return _score(ctx, i, state, row["orig_criteria"], levels, one_hot(5, score - 1))


# ==== 4. 两个回答比较，7 级偏好 + 软标签（nvidia/HelpSteer3 preference）=======================

T_PREF = {
    "en": ["Which response is better for the conversation, and by how much?",
           "Compare `response_1` and `response_2`: which one better answers `last_user_turn`?"],
    "zh": ["对这段对话来说，哪个回复更好？好多少？", "比较 `response_1` 和 `response_2`：哪个更好地回应了 `last_user_turn`？"],
}
PREF_LEVELS = {
    "en": ["Response 1 is much better than Response 2.", "Response 1 is better than Response 2.",
           "Response 1 is slightly better than Response 2.", "Both responses are about equally good.",
           "Response 2 is slightly better than Response 1.", "Response 2 is better than Response 1.",
           "Response 2 is much better than Response 1."],
    "zh": ["回复 1 远好于回复 2。", "回复 1 好于回复 2。", "回复 1 略好于回复 2。", "两个回复差不多好。",
           "回复 2 略好于回复 1。", "回复 2 好于回复 1。", "回复 2 远好于回复 1。"],
}


# 中日韩文字大约 1 个字就是 1 个 token，英文大约 4 个字符 1 个 token，所以按语言给不同的字符预算。
CJK_LANGS = {"chinese", "japanese", "korean"}


def _clip(text: str, limit: int) -> str:
    """把过长的文本截到 ``limit`` 个字符以内：保留开头约 70% 和结尾约 30%，中间用“…”省略。

    为什么保留结尾：回复的结论、代码的最后部分、对话最后的要求往往在末尾，只保留开头会丢掉这些。
    例如 limit=10，"abcdefghijklmnopqrstuvwxyz" → "abcdefg…yz"（开头 7 个 + “…” 1 个 + 结尾 2 个 = 正好 10 个）。
    """
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + "…" + text[len(text) - (limit - head - 1):]


def conv_helpsteer3(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """一段对话 + 两个候选回复 → 一道 7 级打分题（哪个更好、好多少），target 是标注者意见的分布。

    原始分数范围是 -3..+3（负数表示回复 1 更好），加 3 变成第 0..6 级。
    软标签的计算：例如 3 个标注者分别打了 -1、-1、-2
        → 第 2 级（-1）有 2 票，第 1 级（-2）有 1 票 → target = [0, 1/3, 2/3, 0, 0, 0, 0]。
    没有逐人分数时退回到总分 overall_preference 的 one-hot。
    数据中 language="chinese" 的部分是中文对话，标记为 zh。

    材料（state）的构造——为什么不能直接把整段对话和两个回复按原样放进去：
        HelpSteer3 的材料很长，中位数约 1,260 个 token，而我们的序列上限是 1,024。编码器超长时
        从材料的**尾部**截断，原先的顺序是“对话 → 回复 1 → 回复 2”，结果 67% 的题被截断，
        截掉的恰恰是最关键的两个回复（第一版的 HelpSteer3 准确率只有 0.26，原因就在这里）。
    现在的做法：
        第 1 步：最重要的放最前面：用户最后一句话、回复 1、回复 2，每段按字符预算截短（保留首尾）；
        第 2 步：更早的对话只保留最后 4 条消息，每条也截短，放在最后。
        这样即使超长，被截掉的也只是最不重要的早期对话。
    """
    votes = Counter(int(p["score"]) + 3 for p in (row.get("individual_preference") or [])
                    if p.get("score") is not None and -3 <= int(p["score"]) <= 3)
    if votes:
        target = normalize(votes.get(k, 0) for k in range(7))
    elif row.get("overall_preference") is not None:
        target = one_hot(7, int(row["overall_preference"]) + 3)
    else:
        return None
    is_zh = row.get("language") == "chinese"
    lang = "zh" if is_zh else ctx.instr_lang()
    # 字符预算：问题 + 两个回复合计约 700 个 token，给题型提示、问题和 7 个选项描述留出空间。
    cjk = row.get("language") in CJK_LANGS
    user_limit, resp_limit, turn_limit = (250, 450, 150) if cjk else (800, 1200, 400)
    context = row.get("context") or []
    # 第 1 步：找出用户的最后一句话（对话最后一条一般就是用户发言）。
    last_user = next((m["content"] for m in reversed(context) if m.get("role") == "user"), "")
    earlier = context[:-1] if context and context[-1].get("role") == "user" else context
    state = state_to_text({
        "last_user_turn": _clip(last_user, user_limit),
        "response_1": _clip(row["response1"], resp_limit),
        "response_2": _clip(row["response2"], resp_limit),
        # 第 2 步：更早的对话放最后，只留最后 4 条。
        "earlier_conversation": [{"role": m.get("role"), "content": _clip(m.get("content") or "", turn_limit)}
                                 for m in earlier[-4:]],
    })
    d = _score(ctx, i, state, ctx.pick(T_PREF, lang), PREF_LEVELS[lang], target)
    d.lang = "zh" if is_zh else "en"
    return d


# ==== 5. 安全审核 + 危害程度（PKU-Alignment/PKU-SafeRLHF）=====================================

T_SAFE = {
    "en": ["The `response` is safe and harmless.", "Is this response free of harmful content?"],
    "zh": ["这条 `response` 是安全无害的。", "这条回复不包含有害内容吗？"],
}
T_SEVERITY = {
    "en": ["How severe is the potential harm of this response?", "Rate the harm severity of the `response`."],
    "zh": ["这条回复潜在的危害有多严重？", "给这条 `response` 的危害程度打分。"],
}
SEVERITY_LEVELS = {
    "en": ["No harm: the response is safe.",
           "Minor: mildly inappropriate, limited and easily reversible impact.",
           "Moderate: could cause real harm to individuals, such as risky advice or offensive content.",
           "Severe: could enable serious or large-scale harm, such as crime, violence or danger to life."],
    "zh": ["无危害：回复是安全的。", "轻微：略有不当，影响有限且容易挽回。",
           "中等：可能对个人造成实际伤害，例如有风险的建议或冒犯性内容。",
           "严重：可能导致严重或大范围的伤害，例如犯罪、暴力或危及生命。"],
}


def conv_pku(row: Row, i: int, ctx: Ctx) -> list[Decision] | None:
    """一个提问 + 两个回复 → 随机取其中一个回复，出 2 道题：是否安全（noul）、危害程度（4 级 score）。

    为什么随机取一个：两个回复都用会让同一个提问出现两次，数据更冗余；随机取让两种回复都有机会出现。
    """
    k = ctx.rng.choice([0, 1])
    response, safe = row.get(f"response_{k}"), row.get(f"is_response_{k}_safe")
    severity = row.get(f"response_{k}_severity_level")
    if not response or safe is None or severity is None or not 0 <= int(severity) <= 3:
        return None
    lang = ctx.instr_lang()
    state = state_to_text({"prompt": row["prompt"], "response": response})
    ds = [
        _noul(ctx, i, state, ctx.pick(T_SAFE, lang), bool(safe), lang),
        _score(ctx, i, state, ctx.pick(T_SEVERITY, lang), SEVERITY_LEVELS[lang], one_hot(4, int(severity))),
    ]
    return _regroup(ds, ctx, i, ["safe", "severity"])


# ==== 6. agent 该怎么回应（nvidia/When2Call）=================================================

T_WHEN2CALL = {
    "en": ["Given the available tools, how should the assistant respond to the user?",
           "What should the agent do next with this request?"],
    "zh": ["在可用工具的前提下，助手应该怎样回应用户？", "agent 接下来应该如何处理这个请求？"],
}
WHEN2CALL_OPTS = {
    "direct": "Answer directly from its own knowledge, without calling a tool.",
    "tool_call": "Call one of the available tools; the request contains everything the tool needs.",
    "request_for_info": "Ask the user for missing information that a suitable tool requires.",
    "cannot_answer": "Explain that it cannot help: no available tool fits and it cannot answer itself.",
}


def conv_when2call(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """用户请求 + 可用工具列表 → 4 选 1：直接回答 / 调用工具 / 追问缺失信息 / 无法回答。

    这是典型的 agent 控制决策：判断“现在该不该调用工具、信息够不够”。
    工具原本是 JSON 字符串列表，这里只保留名字、描述和必填参数，避免材料太长。
    """
    label = row.get("correct_answer")
    if label not in WHEN2CALL_OPTS or not row.get("question"):
        return None
    tools = []
    for t in row.get("tools") or []:
        try:
            spec = json.loads(t)
        except (TypeError, json.JSONDecodeError):
            continue
        params = spec.get("parameters") or {}
        tools.append({"name": spec.get("name"), "description": spec.get("description"),
                      "required": params.get("required") or list((params.get("properties") or {}).keys())})
    state = state_to_text({"tools": tools, "user": row["question"]})
    opts = [Option(k, v) for k, v in WHEN2CALL_OPTS.items()]
    return _choice(ctx, i, state, ctx.pick(T_WHEN2CALL, ctx.instr_lang()), opts, list(WHEN2CALL_OPTS).index(label))


# ==== 7. 对抗性 NLI（alisawuffles/WANLI）===================================================


def conv_wanli(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """和 MNLI 的转换方式相同（见 sources.py 的 ``_nli``），只是 WANLI 的标签是字符串 ``gold``。

    一半概率出成 noul（“上文能推出这句话吗”），一半出成 3 选 1（蕴含 / 中立 / 矛盾）。
    WANLI 是用模型生成、再由人工筛选和改写的“难例”，专门针对 NLI 模型容易犯的错。
    """
    name = row.get("gold")
    names = ["entailment", "neutral", "contradiction"]
    if name not in names:
        return None
    lang = ctx.instr_lang()
    premise, hyp = row["premise"], row["hypothesis"]
    if ctx.rng.random() < 0.5:
        return _noul(ctx, i, premise, ctx.pick(T_ENTAIL, lang, h=hyp), name == "entailment", lang)
    opts = [Option(n, NLI3_OPTS[lang][n]) for n in names]
    return _choice(ctx, i, premise, ctx.pick(T_NLI3, lang, h=hyp), opts, names.index(name))


# ==== 8. 中文冒犯 / 有毒言论（JunyuLu/ToxiCN、thu-coai/cold）================================

T_TOXIC_ZH = {
    "zh": ["这段话包含冒犯、歧视或有毒的内容。", "这条评论是否带有攻击性或仇恨言论？", "内容审核应当拦截这条评论吗？"],
    "en": ["This Chinese comment is offensive, toxic or hateful.", "Should a moderation filter block this comment?"],
}


def conv_toxicn(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """ToxiCN（知乎、贴吧等平台的中文评论）→ noul：是否有毒。"""
    if row.get("toxic") is None or not row.get("content"):
        return None
    return _noul(ctx, i, row["content"], ctx.pick(T_TOXIC_ZH, "zh"), bool(row["toxic"]), "zh")


def conv_cold(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """COLD（中文冒犯言论数据集，话题包括种族、性别、地域）→ noul：是否冒犯。"""
    if row.get("label") is None or not row.get("TEXT"):
        return None
    return _noul(ctx, i, row["TEXT"], ctx.pick(T_TOXIC_ZH, "zh"), bool(row["label"]), "zh")


# ==== 9. 中文多维度评分（opencsg/UltraFeedback-chinese）=====================================

UF_ASPECTS = {
    "helpfulness": ("这条回答对完成用户的指令有多大帮助？",
                    ["严重错误或毫无帮助。", "部分错误，帮助很小。", "基本正确，满足了基本需求。",
                     "准确且信息丰富，很有帮助。", "非常准确、深入且全面，极有帮助。"]),
    "honesty": ("这条回答是否诚实，对不确定的内容是否如实表达了不确定？",
                ["自信地给出完全错误的内容。", "自信但有明显错误，或回避问题。", "不确定或有小错误，但表达了不确定。",
                 "正确，但对本该确定的内容表现得不够自信。", "正确且表达的确定程度恰当。"]),
    "instruction_following": ("这条回答在多大程度上遵循了用户的指令？",
                              ["完全没有遵循指令。", "只照顾到指令的一小部分。", "部分遵循，有明显偏差。",
                               "基本遵循，只有少量偏差。", "完全遵循指令的所有要求。"]),
    "truthfulness": ("这条回答的内容是否真实，有没有编造或幻觉？",
                     ["严重幻觉，大部分内容不真实。", "有严重的事实错误。", "有部分事实错误或误解。",
                      "基本真实，只有不影响主旨的小问题。", "完全真实，没有幻觉。"]),
}


def conv_uf_zh(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """一条中文指令 + 若干模型回答（每个回答都有 4 个维度的 1–5 分）→ 随机取一个回答、一个维度，出一道 5 级打分题。

    举个例子：指令“列出唐代最著名的三位诗人”，某个回答在 helpfulness 上得 4 分
        → 问题“这条回答对完成用户的指令有多大帮助？”，5 个等级各有中文描述，答案是第 3 级（从 0 数）。
    评分是 GPT 类模型给出的，不是人工标注，质量略低于人工数据，但胜在是中文、而且量大。
    """
    comps = [c for c in (row.get("completions") or []) if c.get("response") and c.get("annotations")]
    if not comps or not row.get("instruction"):
        return None
    comp = ctx.rng.choice(comps)
    aspects = [a for a in UF_ASPECTS if str((comp["annotations"].get(a) or {}).get("Rating", "")).isdigit()]
    if not aspects:
        return None
    aspect = ctx.rng.choice(aspects)
    rating = int(comp["annotations"][aspect]["Rating"])
    if not 1 <= rating <= 5:
        return None
    question, levels = UF_ASPECTS[aspect]
    state = state_to_text({"指令": row["instruction"], "回答": comp["response"]})
    return _score(ctx, i, state, question, levels, one_hot(5, rating - 1))


# ==== 注册表 ===============================================================================
# Source 的参数依次是：内部名字、HF 路径、config、训练 split、评测 split、材料语言、转换函数。
# train_cap 是本数据源的训练采样上限（按“道题”计）；中文数据源在 build.py 里还会再乘以 --zh-cap-mult。

MORE_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # bev-decision：5 个子集分别设上限。default 最大、领域最杂；hard 是陷阱题；
        # counterfactual 是改一处证据答案就翻转的成对题；numeric_temporal 考数字和时间；skills 带软标签。
        Source("bev_default", "avbiswas/bev-decision", "default", "train", "test", "en", conv_bev, train_cap=15000),
        Source("bev_hard", "avbiswas/bev-decision", "hard_50k", "train", "test", "en", conv_bev, train_cap=8000),
        Source("bev_counterfactual", "avbiswas/bev-decision", "counterfactual_15k", "train", "test", "en", conv_bev,
               train_cap=5000),
        Source("bev_numeric", "avbiswas/bev-decision", "numeric_temporal", "train", "test", "en", conv_bev,
               train_cap=5000),
        Source("bev_skills", "avbiswas/bev-decision", "skills", "train", "test", "en", conv_bev, train_cap=8000),
        # 客服工单：只有 train，打乱后 90% 训练、10% 评测。label_field="queue" 用来收集全部队列名。
        Source("support_tickets", "Tobi-Bueck/customer-support-tickets", None, "train#train", "train#eval", "en",
               conv_ticket, label_field="queue", train_cap=9000),
        Source("feedback_collection", "prometheus-eval/Feedback-Collection", None, "train#train", "train#eval", "en",
               conv_feedback, train_cap=5000),
        Source("helpsteer3", "nvidia/HelpSteer3", "preference", "train", "validation", "en", conv_helpsteer3,
               train_cap=5000),
        Source("pku_saferlhf", "PKU-Alignment/PKU-SafeRLHF", "default", "train", "test", "en", conv_pku,
               train_cap=6000),
        # When2Call 的训练集只有对话、没有显式标签，所以用带 correct_answer 的 test/mcq，
        # 打乱后 85% 训练、15% 评测（我们不拿 When2Call 当正式基准，所以可以这样用）。
        # 注意 mcq 是按答案类型排序的，必须打乱再切，否则评测部分全是 tool_call。
        Source("when2call", "nvidia/When2Call", "test", "mcq#train", "mcq#eval", "en", conv_when2call,
               train_cap=4000, holdout=0.15),
        Source("wanli", "alisawuffles/WANLI", None, "train", "test", "en", conv_wanli),
        Source("toxicn", "JunyuLu/ToxiCN", None, "train", "test", "zh", conv_toxicn),
        # COLD 的 test.csv 比其他文件多一列，整体加载会报错，所以按文件加载，用 dev.csv 做评测。
        Source("cold", "thu-coai/cold", None, "train", "dev", "zh", conv_cold,
               data_files={"train": "train.csv", "dev": "dev.csv"}),
        # 仓库里另有两个“二值化”衍生文件，只读主文件，避免混在一起加载。
        Source("ultrafeedback_zh", "opencsg/UltraFeedback-chinese", None, "train#train", "train#eval", "zh",
               conv_uf_zh, train_cap=4000, data_files={"train": "ultrafeedback_zh.parquet"}),
    ]
}
