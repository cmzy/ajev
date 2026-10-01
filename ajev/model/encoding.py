"""把一道 Decision（决策题）编码成 encoder 的输入 token id，并为每个选项放一个标记位。

====================================================================
先补几个基础概念（第一次接触 NLP 模型可以先读这一段）
====================================================================

1. tokenizer（分词器）与 token id
   神经网络只能处理数字，不能直接处理文字。tokenizer 的工作就是把一段文字切成小片段
   （叫 token，可能是一个字、一个词或词的一部分），再把每个 token 换成它在“词表”里的编号，
   这个编号就是 token id。
   举例（数字是随便写的，只为说明）：
       "退款申请"  --切分-->  ["退款", "申请"]  --查词表-->  [48213, 9021]
   mmBERT 的词表有约 25.6 万个 token，覆盖中文、英文等上千种语言。

2. 特殊 token
   除了普通文字，词表里还有几个“特殊 token”，不代表任何文字，只起标记作用：
   - <bos>（begin of sequence）：放在序列最开头。模型在这个位置输出的向量常被当作“整句话的摘要”。
   - <eos>（end of sequence）：放在每一段的结尾，起“分隔符”作用，告诉模型一段内容结束了。
   - <mask>：BERT 类模型预训练时，会把句子里一些词替换成 <mask> 让模型猜，
     所以模型天生擅长“把周围上下文的信息汇聚到 <mask> 位置上”。我们正是利用这一点，
     在每个选项前放一个 <mask>，让模型在这个位置总结“这个选项和题目是否匹配”。
   - <pad>：补齐用的占位符（见 ajev/model/batching.py），本文件用不到。

3. 为什么要“编码”
   一道题包含：题型、问题、若干选项（名字 + 描述）、材料（state）。模型的输入只能是一条
   token 序列，所以要按固定格式把这些内容拼成一条序列，并记住每个选项的 <mask> 在第几个位置，
   模型之后才能到这些位置取出结果。

====================================================================
在 AJev 流程中的位置
====================================================================

数据构建产出的 JSONL → **本模块编码** → 模型前向（ajev/model/encoder.py）→ 训练 / 校准 / 评测。
训练（ajev.train.train）和推理（ajev.model.predictor）共用同一个 ``DecisionEncoder``，
保证两边看到的输入格式完全一致（格式不一致是模型“训练时好、上线后差”的常见原因）。

====================================================================
序列布局（一道题对应一条序列，所有选项在同一次前向中一起打分）
====================================================================

    <bos> {type}: {instructions} <eos> <mask> {name}: {desc} <mask> {name}: {desc} ... <eos> {state} <eos>

- ``{type}`` 是题型提示词：noul（是/否题）→ "yes/no"，choice（单选题）→ "choice"，
  score（打分题）→ "score"，让模型知道当前是哪类题。
- 每个选项前面放一个 ``<mask>`` 作为“标记位”（marker）。用已有的 ``<mask>`` 而不是新造一个特殊
  token，是因为新 token 的向量需要从零学起，而 <mask> 的“汇聚上下文”能力是预训练现成的。
- 选项放在 state（材料）**之前**：序列太长时只会截掉 state 的尾部，问题和选项永远完整。
  （材料少看几句通常还能答，选项缺了就根本没法答。）

长度预算（单位：token）：
- instructions（问题）最多 ``max_instr`` 个 token；
- 选项区最多 ``max_options`` 个 token（选项太长时先压缩每个选项的描述）；
- state 用剩下的全部长度，直到 ``max_len``。
为什么要限制长度：Transformer 的计算量和显存大约随序列长度的平方增长，长度必须有上限。

举个例子（为了直观，假设一个字就是一个 token）：
    题目：type="noul"，instructions="需要退款吗"，
          options=[("true","是"), ("false","否")]，state="我要退货"
    编码后的序列（用文字表示 token）：
        位置:  0      1..5        6     7      8..11     12     13..16     17     18..21    22
        内容: <bos> "yes/no: " ...需要退款吗 <eos> <mask> "true: 是" <mask> "false: 否" <eos> 我要退货 <eos>
    marker_pos 记录两个 <mask> 的位置，例如 [12, 17]（具体数字取决于真实分词结果）。
"""

from __future__ import annotations

# dataclass 是 Python 标准库提供的装饰器：只要声明字段，就自动生成 __init__ 等方法，
# 适合写这种“只用来装数据”的小类。
from dataclasses import dataclass

from ajev.schema import Decision

# 题型 → 写进序列开头的提示词，让模型知道当前是哪类题。
TYPE_PREFIX = {"noul": "yes/no", "choice": "choice", "score": "score"}


@dataclass
class Encoded:
    """一道题编码后的结果。

    Attributes:
        input_ids: 完整的 token id 序列（含 <bos>/<eos>/<mask> 等特殊 token），长度 ≤ max_len。
            这就是喂给模型的输入。
        marker_pos: 每个选项的 ``<mask>`` 标记位在 input_ids 中的下标，顺序与 options 一致。
            例如 [12, 17] 表示第 1 个选项的标记在第 12 位、第 2 个在第 17 位。
            模型前向后，会到这些位置取向量来给选项打分。
        truncated_state: state 是否因超长被截断（用于统计截断比例，判断 max_len 设得够不够）。
    """

    input_ids: list[int]
    marker_pos: list[int]  # 每个选项标记位的下标，按选项顺序排列
    truncated_state: bool


class DecisionEncoder:
    """把 Decision 编码成 ``Encoded``，负责拼接序列、放置标记位和执行长度预算。

    Args:
        tokenizer: HuggingFace 的 tokenizer 对象，必须有 cls/bos、sep/eos 和 mask 三种特殊 token
            （mmBERT 中 cls 就是 <bos>，sep 就是 <eos>，不同模型叫法不同但作用一样）。
        max_len: 整条序列的最大 token 数（T4 上默认 1024；实测只有约 0.3% 的题会超过）。
        max_instr: 问题（instructions）的最大 token 数，超出部分直接截掉。
        max_options: 选项区（所有标记位 + 选项名 + 描述）的 token 预算。
        min_desc: 压缩描述时，每个选项描述至少保留的 token 数；如果预算连这么多都给不了，
            干脆全部去掉描述只保留选项名（只剩半句的描述往往比没有描述更容易误导模型）。

    举个例子：
        encoder = DecisionEncoder(tokenizer, max_len=1024)
        enc = encoder.encode(decision)   # decision 有 3 个选项
        enc.input_ids    # 例如长度 120 的 int 列表
        enc.marker_pos   # 例如 [15, 27, 40]，3 个选项各一个标记位
    """

    def __init__(self, tokenizer, max_len: int = 1024, max_instr: int = 128, max_options: int = 384,
                 min_desc: int = 8) -> None:
        self.tok = tokenizer
        self.max_len = max_len
        self.max_instr = max_instr
        self.max_options = max_options
        self.min_desc = min_desc
        # 优先用 cls/sep，没有时退回 bos/eos，兼容不同 tokenizer 的命名习惯。
        # 写法 `A if 条件 else B` 是 Python 的条件表达式，相当于其他语言的 三元运算符。
        self.bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.marker = tokenizer.mask_token_id
        # 缺任何一个特殊 token 都没法按约定格式编码，尽早报错比训练到一半才发现好。
        if None in (self.bos, self.eos, self.marker):
            raise ValueError("tokenizer needs cls/bos, sep/eos and mask tokens")

    def _ids(self, text: str) -> list[int]:
        """把一段文本切成 token id（不加任何特殊 token）；空文本返回空列表。

        add_special_tokens=False：tokenizer 默认会自动在首尾加 <bos>/<eos>，
        但我们要自己控制特殊 token 的位置，所以关掉。
        例如 "需要退款吗" → [3021, 887, 1456]（数字仅示意）。
        """
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def _options(self, d: Decision) -> list[list[int]]:
        """为每个选项生成 ``[<mask>] + 选项名 + ": 描述"`` 的 token 片段，并保证总长不超预算。

        策略：
        1. 先只算“标记位 + 选项名”的长度（names_only）。如果连这都超过 max_options，
           说明选项太多，直接报错（以后可改为分组读取）。
        2. 否则用二分查找求出“每个选项描述最多能保留多少 token”（所有描述共用同一个上限 cap），
           使总长度刚好不超预算。
        3. 若求出的上限小于 min_desc（且不是因为描述本来就短），就把描述全部去掉。

        举个例子：3 个选项，选项名各占 2 个 token，描述分别长 30 / 10 / 50 个 token，max_options=60。
            names_only = 3 × (1 个标记位 + 2) = 9，剩给描述的空间是 51。
            若 cap=17：min(30,17)+min(10,17)+min(50,17) = 17+10+17 = 44 ≤ 51，可以；
            若 cap=21：21+10+21 = 52 > 51，不行。二分查找会找到最大的可行值 cap=20（20+10+20=50）。
            结果：第 1、3 个选项的描述被截到 20 个 token，第 2 个描述本来就短，保持不变。

        为什么用二分查找：cap 越大总长度越长（单调），这种“找满足条件的最大值”的问题用二分
        只需要试 log2(最长描述) 次，比从大到小一个个试快得多。

        Returns:
            每个选项一个 token 列表，第一个元素一定是标记位 ``<mask>``。
        """
        names = [self._ids(o.name) for o in d.options]
        # 描述前面加 ": "，让模型看到的是 "退款: 客户要求退钱" 这样自然的格式；没有描述就是空列表。
        descs = [self._ids(f": {o.desc}") if o.desc else [] for o in d.options]
        # 每个选项至少占 1（标记位）+ 选项名长度。
        names_only = sum(1 + len(n) for n in names)
        if names_only > self.max_options:
            raise ValueError(
                f"{d.id}: {len(d.options)} option names need {names_only} tokens > max_options={self.max_options}"
            )
        # 二分查找：在预算内，每个选项描述可以保留的最大 token 数。
        # 搜索区间是 [0, longest]：0 表示不要描述，longest 表示所有描述都完整保留。
        longest = max((len(x) for x in descs), default=0)
        lo, hi = 0, longest
        while lo < hi:
            # 取偏上的中点（+1），配合 lo = mid 才不会死循环。
            mid = (lo + hi + 1) // 2
            if names_only + sum(min(len(x), mid) for x in descs) <= self.max_options:
                lo = mid  # mid 可行，答案至少是 mid
            else:
                hi = mid - 1  # mid 不可行，答案一定小于 mid
        # lo == longest 表示描述无需截断，即使很短也全部保留；否则低于 min_desc 就整体丢弃描述。
        cap = lo if lo >= self.min_desc or lo == longest else 0
        # x[:cap] 是 Python 切片，取列表前 cap 个元素（不足 cap 个就全取）。
        return [[self.marker] + n + x[:cap] for n, x in zip(names, descs)]

    def encode(self, d: Decision) -> Encoded:
        """按模块说明中的布局编码一道题。

        拼接顺序（共 4 步）：
        第 1 步：头部 = <bos> + 题型提示 + 问题 + <eos>；
        第 2 步：选项区，一边拼接一边记录每个标记位的下标；
        第 3 步：加一个 <eos> 分隔选项区和材料；
        第 4 步：材料 state 用剩余预算，超出就截掉尾部，最后再加 <eos>。

        举个例子：max_len=20，头部占 8 个 token，选项区占 9 个 token，加上分隔 <eos> 共 18 个。
            room = 20 - 18 - 1 = 1，即材料只能放 1 个 token（末尾 <eos> 要预留 1 个位置）。
            若材料有 5 个 token，就只保留第 1 个，truncated_state=True，最终序列长度正好 20。
        """
        # 第 1 步：头部。题型提示最多 8 个 token，问题最多 max_instr 个 token。
        head = [self.bos] + self._ids(f"{TYPE_PREFIX[d.type]}: ")[:8] + self._ids(d.instructions)[: self.max_instr]
        head.append(self.eos)
        ids = list(head)
        # 第 2 步：选项区。
        marker_pos = []
        for chunk in self._options(d):
            # 标记位就是每个选项片段的第一个 token；此刻 len(ids) 正好是它将要放入的位置。
            marker_pos.append(len(ids))
            ids += chunk
        # 第 3 步：分隔符。
        ids.append(self.eos)
        # 第 4 步：材料。可用长度 = max_len - 已用长度 - 末尾 <eos> 占的 1 个位置。
        room = self.max_len - len(ids) - 1
        state = self._ids(d.state)
        truncated = len(state) > room
        # max(room, 0) 防止 room 为负数时切片出错（理论上选项预算保证了 room ≥ 0，这里是双保险）。
        ids += state[: max(room, 0)] + [self.eos]
        return Encoded(ids, marker_pos, truncated)
