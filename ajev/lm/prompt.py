"""把一道 Decision 写成给大语言模型（如 Gemma 4 12B）的提示词，选项用字母 A、B、C… 标记。

在 AJev 中的位置：这是“大模型路线”（对应 Winnow-12B / JevK5 的做法）的输入格式。mmBERT 路线用的是
``ajev/model/encoding.py``（每个选项前放一个 <mask> 标记位）；大模型路线则是让模型“读题后说出一个字母”，
我们**不让它真的生成**，只读它在“下一个 token 是哪个字母”上的打分（logits），
在 A、B、C… 这几个字母之间做 softmax，就得到了每个选项的概率。一次前向就够，不需要逐字生成。

为什么用字母而不是直接用选项名：选项名可能很长（例如“发票金额与采购单不一致”），由好多个 token 组成，
没法用“下一个 token”的一个打分来代表；而 A、B、C 各自都是一个 token，正好可以一一对应。

提示词结构（材料在前、问题和选项在后）::

    You are a decision model. Read the state and answer the question by picking exactly one option.

    ### State
    {state}

    ### Question
    {instructions}

    ### Options
    A. delivery: 物流配送
    B. refund: 退款

    Reply with the letter of the best option only.

材料放在最前面：同一份材料上的多个问题，前缀完全相同，推理框架（例如 llama.cpp / vLLM）可以复用这段的
计算结果（KV cache），只需为每个问题算后面一小段。这和 mmBERT 路线“选项在前”的考虑正好相反。
"""

from __future__ import annotations

from ajev.schema import Decision

LETTERS = [chr(ord("A") + i) for i in range(26)]
# 选项最多 255 个（与 Jev 相同）。26 个以内用字母 A–Z；超过 26 个时在 A–Z 后面接两字母编码 AA、AB……
# 只保留分词器把它当作**一个** token 的编码（由 ajev/lm/predictor.py 的 option_labels 按分词器筛选），
# 这样仍然只需一次前向、读“下一个 token 是哪个编码”。这是 Jev 复刻排行榜上前几名的通用做法。
MAX_OPTIONS = 255
MAX_LETTER_OPTIONS = len(LETTERS)

TYPE_HINT = {
    "noul": "Decide whether the statement holds.",
    "choice": "Pick the single best option.",
    "score": "Pick the level that best fits; levels are ordered from lowest to highest.",
}


def option_lines(d: Decision, labels: list[str] | None = None) -> list[str]:
    """每个选项一行：``A. 名字: 描述``。noul 的 true / false 写成 Yes / No，score 的等级名就是数字。"""
    lines = []
    for letter, o in zip(labels or LETTERS, d.options):
        name = {"true": "Yes", "false": "No"}.get(o.name, o.name) if d.type == "noul" else o.name
        lines.append(f"{letter}. {name}: {o.desc}" if o.desc else f"{letter}. {name}")
    return lines


def build_user_message(d: Decision, state: str | None = None, labels: list[str] | None = None) -> str:
    """生成用户消息正文。``state`` 可以传入截短后的材料，默认用 d.state。

    ``labels``：选项标签表（A–Z 后接两字母编码，由 option_labels 按分词器生成）；不传时只有 A–Z，
    超过 26 个选项会报错。26 个选项以内的提示词与以前**逐字相同**，已训练的适配器不受影响；
    超过 26 个时最后一句的 "letter" 改为 "code"。
    """
    labels = labels or LETTERS
    if len(d.options) > len(labels):
        raise ValueError(f"{d.id}: {len(d.options)} options > {len(labels)} labels")
    word = "letter" if len(d.options) <= MAX_LETTER_OPTIONS else "code"
    return (
        "You are a decision model. Read the state and answer the question by picking exactly one option.\n\n"
        f"### State\n{d.state if state is None else state}\n\n"
        f"### Question\n{d.instructions}\n{TYPE_HINT[d.type]}\n\n"
        "### Options\n" + "\n".join(option_lines(d, labels)) + "\n\n"
        f"Reply with the {word} of the best option only."
    )
