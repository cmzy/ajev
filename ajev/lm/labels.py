"""选项标签表：A–Z，后面接分词器里是单个 token 的两字母编码（不依赖 torch，训练、推理、排查脚本共用）。"""

from __future__ import annotations

from ajev.lm.prompt import LETTERS, MAX_OPTIONS

_LABEL_CACHE: dict[int, tuple[list[str], list[list[int]], list[list[bool]]]] = {}


def option_labels(tok) -> tuple[list[str], list[list[int]], list[list[bool]]]:
    """选项标签表：A–Z，后面接分词器里是**单个 token** 的两字母编码 AA、AB……，最多 255 个。

    返回 (标签字符串列表, 每个标签的 token id [N, 2], 有效标记 [N, 2])。
    每个标签最多两种写法（"AB" 和 " AB"），只保留恰好是单个 token 的写法；只有一种时第二格用第一格填充、
    标记为无效（合并打分时排除，不会重复计算）。

    两字母编码还要通过“答题位置检查”：把编码接在对话模板的答题位置后面重新分词，必须正好多出这一个 token，
    否则模型在答题位置上无法用一个 token 说出它（例如和前面的字符粘连成别的 token）。
    Gemma 4 的分词器里，676 个两字母组合有 653 个是单 token，凑满 255 个绰绰有余。
    结果按分词器缓存，每个分词器只算一次。
    """
    key = id(tok)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    ctx = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    ctx_ids = tok.encode(ctx, add_special_tokens=False)

    def forms_of(label: str) -> list[int]:
        forms = []
        for form in (label, " " + label):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1 and t[0] not in forms:
                forms.append(t[0])
        return forms

    labels, ids, valid, seen = [], [], [], set()
    candidates = list(LETTERS) + [a + b for a in LETTERS for b in LETTERS]
    for label in candidates:
        if len(labels) == MAX_OPTIONS:
            break
        forms = forms_of(label)
        if len(label) == 1 and not forms:
            raise ValueError(f"letter {label!r} is not a single token for this tokenizer")
        if not forms or forms[0] in seen:
            continue
        if len(label) > 1 and tok.encode(ctx + label, add_special_tokens=False) != ctx_ids + forms[:1]:
            continue  # 答题位置检查没通过
        seen.update(forms)
        labels.append(label)
        ids.append(forms + [forms[0]] * (2 - len(forms)))
        valid.append([True] * len(forms) + [False] * (2 - len(forms)))
    _LABEL_CACHE[key] = (labels, ids, valid)
    return _LABEL_CACHE[key]
