"""核心数据结构：数据构建、训练、评测、部署共用的统一数据格式。

================================================================
一、先理解我们要做的事：Jev 是什么
================================================================
Jev 是 TypeSafe 公司做的一种“只做判断、不写文字”的 AI 模型。普通的大语言模型（如 ChatGPT）
会一个字一个字地“生成”回答；而 Jev 只回答“带类型的问题”，并且直接给出每个候选答案的概率。

它支持三种题型（type）：

    * noul   —— 是/否题。例如：“这位客户是在要求退款吗？”
                选项固定为 ``true``（是）和 ``false``（否），Jev 返回 P(true)，比如 0.92。
    * choice —— 单选题。例如：“这段对话主要关于什么？” 选项：billing / delivery / refund …
                最多 255 个选项，返回每个选项的概率，以及最可能的那个。
    * score  —— 打分题。例如：“这件事有多紧急？” 等级：0 不急 / 1 一般 / 2 较急 / 3 非常急。
                2～10 个“有顺序”的等级，返回每一级的概率和概率加权的平均分。

================================================================
二、什么是 wire API（线上请求/响应格式）
================================================================
“wire format / wire API” 指程序之间通过网络传输数据时约定的 JSON 格式。Jev 的请求长这样::

    {
      "state": {"thread": [...], "account": {...}},          # 一段材料（文字或 JSON）
      "questions": {
        "refund": {"type": "noul",   "instructions": "客户在要求退款吗？"},
        "topic":  {"type": "choice", "instructions": "主要关于什么？",
                   "criteria": {"billing": "扣费问题", "delivery": "物流问题"}},
        "urgency":{"type": "score",  "instructions": "有多紧急？",
                   "criteria": ["不急", "一般", "较急", "非常急"]}
      }
    }

响应里每个问题对应一个答案（answer），例如 ``{"noul": 0.92}``、
``{"choice": "delivery", "probabilities": {...}, "confidence": 0.81}``。

================================================================
三、本模块的核心：Decision（一道决策题）
================================================================
一个 Jev 请求里有“一段材料 + 多个问题”。为了方便训练，我们把它拆成多道独立的题，
每道题就是一个 ``Decision``：

    Decision = 一段材料（state） + 一个问题（instructions） + 若干选项（options）
               + 目标概率分布（target，即“标准答案”）

在 AJev 的整个流程（数据构建 → 训练 → 校准 → 评测）中，所有环节都只认这一种数据单元：
    * 数据构建：把各种公开数据集转换成 Decision，存成 JSONL 文件；
    * 训练：模型读入 Decision，学习输出接近 target 的概率分布；
    * 评测：比较模型输出和 target，计算准确率等指标；
    * 部署：收到 Jev 请求 → 拆成 Decision → 模型打分 → 用 ``jev_answer`` 拼回 Jev 响应。

================================================================
四、几个基础概念
================================================================
* 概率分布：一组非负数，加起来等于 1，表示“每个选项有多大可能是对的”。
  例如三选一的 [0.7, 0.2, 0.1] 表示模型认为第 1 个选项有 70% 可能是正确答案。
* one-hot：只有一个位置是 1、其他都是 0 的分布，例如 [0, 1, 0]，表示“确定是第 2 个”。
  普通数据集的标准答案（硬标签）就用 one-hot 表示。
* soft label（软标签）：不是 one-hot 的分布，例如 5 个人标注，3 人选 A、2 人选 B，
  就得到 [0.6, 0.4]。它比硬标签包含更多信息（告诉模型“这题本来就有争议”）。
* argmax：一组数里最大值所在的“位置（下标）”。[0.7, 0.2, 0.1] 的 argmax 是 0（第 1 个）。
  模型“选中的答案”就是预测分布的 argmax。
* JSONL（JSON Lines）：一种文本文件格式，每一行是一个独立的 JSON 对象。
  好处是可以一行一行地读写，文件很大时也不需要一次性全部载入内存。
"""

# 这行让类型注解（如 list[Option]、"Decision"）延迟求值：
# 可以在类定义完成之前就在注解里引用它，也能在较老的 Python 版本上使用新式写法。
from __future__ import annotations

import json  # 读写 JSON
import math  # 数学函数，这里用到 log（对数）
import random  # 随机数，用于打乱选项顺序
# dataclass：一种快速定义“数据类”的装饰器，见下方 Option 的说明。
# asdict：把 dataclass 对象递归转换成普通 dict；field：给字段设置默认值等额外选项。
from dataclasses import asdict, dataclass, field
# 类型注解工具：Any 表示任意类型；Iterable 表示“可以被 for 循环遍历的东西”；
# Literal 表示“只能取这几个固定值之一”。
from typing import Any, Iterable, Literal

# 三种题型。Literal["noul", "choice", "score"] 表示这个类型只能是这三个字符串之一，
# 编辑器 / 类型检查器可以据此发现拼写错误（比如写成 "chioce"）。
DecisionType = Literal["noul", "choice", "score"]
# 同样三个字符串，放进元组（tuple）里，方便在运行时检查“题型是否合法”。
TYPES: tuple[DecisionType, ...] = ("noul", "choice", "score")

# Jev 官方的限制：choice 最多 255 个选项；score 有 2～10 个等级。
MAX_CHOICE_OPTIONS = 255
# 一行同时给两个变量赋值（元组解包）：MIN_SCORE_LEVELS = 2，MAX_SCORE_LEVELS = 10。
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

# noul 题的两个选项名永远是 "true" 和 "false"；Jev 返回的 noul 值就是 P("true")，
# 即“答案为是”的概率。
NOUL_TRUE, NOUL_FALSE = "true", "false"
# 当请求里没给 noul 的 criteria（判断标准说明）时，用这里的默认描述。
# 按语言区分，是因为描述会和问题一起输入给模型：中文题配中文描述，避免语言混杂干扰模型。
# 这是一个“字典套字典”：DEFAULT_NOUL_DESC["zh"]["true"] == "是，该陈述成立。"
DEFAULT_NOUL_DESC = {
    "en": {NOUL_TRUE: "Yes, the statement holds.", NOUL_FALSE: "No, the statement does not hold."},
    "zh": {NOUL_TRUE: "是，该陈述成立。", NOUL_FALSE: "否，该陈述不成立。"},
}


# @dataclass 是 Python 的“数据类”装饰器。我们只需要写出字段名和类型，
# Python 会自动生成 __init__（构造函数）、__repr__（打印显示）、__eq__（判断相等）等方法。
# 也就是说，写完下面几行就可以直接 Option("billing", "扣费问题") 创建对象，
# 并且两个字段都相同的 Option 会被判断为相等（==）。
@dataclass
class Option:
    """一个候选选项。

    属性:
        name: 选项名，是模型最终“选中”的标识，例如 ``"billing"``、``"体育"``、``"true"``、``"2"``。
              同一道题里选项名必须唯一，否则无法分辨模型选的是哪一个。
        desc: 选项的说明文字（对应 Jev 请求里的 criteria），可以为空字符串。
              它会和 name 一起拼进模型的输入，帮助模型理解每个选项的含义和边界——
              描述写得越具体，模型越容易判断。

    举个例子:
        >>> Option("delivery", "物流、发货、配送相关的问题")
        Option(name='delivery', desc='物流、发货、配送相关的问题')
        >>> Option("true")           # desc 有默认值 ""，可以不传
        Option(name='true', desc='')
    """

    name: str
    desc: str = ""  # “= ''” 表示这个字段有默认值，创建对象时可以省略


@dataclass
class Decision:
    """一道决策题：模型打分的基本单位，也是 JSONL 数据文件里每一行的结构。

    属性:
        id:           全局唯一编号，例如 ``"tnews/12"`` 或 ``"customer_service_000000/action"``。
        source:       数据来源，例如 ``"tnews"``；typed-decisions 会带上业务流程名，如
                      ``"typed_decisions/customer_service"``。评测时可按来源分别统计成绩。
        type:         题型：noul（是/否）/ choice（单选）/ score（打分）。
        state:        材料文本。如果原始 state 是 JSON 对象，会被转成紧凑的 JSON 字符串。
                      可以为空字符串（例如 ARC 这类纯问答题，问题本身就在 instructions 里）。
        instructions: 问题本身（对应 Jev 的 instructions）。
        options:      候选选项列表，列表顺序就是输入模型时的顺序。
        target:       与 options 等长的目标概率分布（加起来为 1），也就是“标准答案”。
                      硬标签存成 one-hot；多人标注或教师模型给出的就是 soft label（软标签）。
        lang:         材料的语言（"en" 英文 / "zh" 中文），用于分语言统计成绩。
        group:        同一个 group 的多道题，是对同一段 state 提的问题（来自同一个 Jev 请求）。
                      划分验证集时按 group 整组留出，避免同一段材料同时出现在训练集和验证集里
                      （否则模型可能“背过”这段材料，验证成绩会虚高）。
        meta:         额外信息，例如 ``question_id``（原始问题名）、``gold_label``（官方标准答案名）。

    举个例子（一道中文新闻分类题，存成 JSONL 时就是一行）::

        Decision(
            id="tnews/12", source="tnews", type="choice", lang="zh",
            state="国足昨晚2比0战胜对手……",
            instructions="这篇文章属于哪个类别？",
            options=[Option("体育", "体育赛事与运动员"), Option("财经", "经济、金融、商业")],
            target=[1.0, 0.0],          # 标准答案是第 1 个选项“体育”（one-hot）
        )
    """

    id: str
    source: str
    type: DecisionType
    state: str
    instructions: str
    options: list[Option]
    target: list[float]
    lang: str = "en"
    # group 相同的多道题，是针对同一段 state 提问的（来自同一个 Jev 请求）。
    group: str = ""
    # 可变类型（dict、list）不能直接写 “= {}” 作为默认值：那样所有对象会共用同一个 dict。
    # field(default_factory=dict) 表示“每创建一个对象，都调用 dict() 新建一个空字典”。
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """dataclass 在自动生成的 __init__ 执行完后，会自动调用这个方法，用来做额外处理。

        为什么需要它：从 JSON 文件读回来时，options 里的每一项是普通 dict，例如
        ``{"name": "体育", "desc": "..."}``，而不是 ``Option`` 对象。这里统一转换一下，
        后面的代码就可以放心地写 ``o.name``，不用担心类型不一致。
        """
        # 列表推导式：对 options 中的每一项 o，如果已经是 Option 就保留，
        # 否则用 Option(**o) 创建——“**o” 把字典展开成关键字参数，等价于 Option(name=..., desc=...)。
        self.options = [o if isinstance(o, Option) else Option(**o) for o in self.options]

    # ---- 校验 -----------------------------------------------------------
    def validate(self) -> None:
        """检查这道题是否合法，不合法就抛出 ``ValueError`` 异常。

        为什么需要：数据来自二十多个格式各异的公开数据集，转换时很容易出错
        （比如选项和概率数量对不上）。每生成一道题就校验一次，能在数据构建阶段立刻发现问题，
        而不是等训练了几个小时才莫名其妙地报错。

        检查内容（按代码顺序）:
            1. 题型必须是 noul / choice / score 之一；
            2. noul 题的选项名必须正好是 true 和 false（顺序可以互换，因为训练时会打乱）；
            3. choice 题要有 2～255 个选项；score 题要有 2～10 个等级；
            4. 选项名不能重复；
            5. target 的长度必须等于选项数；
            6. target 必须是合法的概率分布：每个数 ≥ 0，总和等于 1（允许 0.001 的误差）。

        举个例子:
            >>> Decision(id="x", source="s", type="choice", state="", instructions="q",
            ...          options=[Option("a"), Option("b")], target=[0.5, 0.3]).validate()
            ValueError: x: target is not a distribution: [0.5, 0.3]      # 0.5 + 0.3 ≠ 1
        """
        k = len(self.options)  # k = 选项个数
        if self.type not in TYPES:
            # f"..." 是格式化字符串，{} 里的表达式会被替换成它的值；!r 表示用 repr() 显示（字符串带引号）。
            raise ValueError(f"{self.id}: bad type {self.type!r}")
        # noul 的两个选项允许任意顺序（训练时会打乱选项顺序），但名字必须正好是 true/false。
        if self.type == "noul" and [o.name for o in self.options] not in (
            [NOUL_TRUE, NOUL_FALSE],
            [NOUL_FALSE, NOUL_TRUE],
        ):
            raise ValueError(f"{self.id}: noul options must be true/false")
        # Python 支持连写比较：2 <= k <= 255 等价于 (2 <= k) and (k <= 255)。
        if self.type == "choice" and not 2 <= k <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"{self.id}: choice needs 2..{MAX_CHOICE_OPTIONS} options, got {k}")
        if self.type == "score" and not MIN_SCORE_LEVELS <= k <= MAX_SCORE_LEVELS:
            raise ValueError(f"{self.id}: score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {k}")
        # 用集合（set，自动去重）判断有没有重名：如果去重后数量变少，说明有重复的选项名。
        # 选项名是答案的唯一标识，重名会导致“模型到底选了哪个”说不清楚。
        if len({o.name for o in self.options}) != k:
            raise ValueError(f"{self.id}: duplicate option names")
        if len(self.target) != k:
            raise ValueError(f"{self.id}: target has {len(self.target)} entries for {k} options")
        # 软标签来自多人投票的比例，四舍五入后加起来可能是 0.999999 或 1.000001，
        # 所以不要求严格等于 1，只要误差不超过 0.001 就算合法。
        # any(...) 只要有一个元素满足条件就返回 True。
        if any(p < 0 for p in self.target) or abs(sum(self.target) - 1) > 1e-3:
            raise ValueError(f"{self.id}: target is not a distribution: {self.target}")

    # ---- 辅助属性与方法 ---------------------------------------------------
    # @property 让一个方法可以像属性一样使用：写 d.option_names，而不是 d.option_names()。
    @property
    def option_names(self) -> list[str]:
        """按当前顺序返回所有选项名。

        举个例子: 选项为 [Option("体育"), Option("财经")] 时，返回 ["体育", "财经"]。
        """
        return [o.name for o in self.options]

    @property
    def gold_index(self) -> int:
        """返回 target 中概率最大的选项下标，也就是“标准答案”在第几个位置（从 0 开始数）。

        这就是对 target 取 argmax。``max(range(n), key=f)`` 的意思是：在 0..n-1 里找出
        使 f(i) 最大的那个 i；这里 f 是 ``self.target.__getitem__``，即 ``self.target[i]``。

        举个例子: target = [0.1, 0.7, 0.2] → 返回 1（第 2 个选项概率最大）。

        注意：如果有并列最大值（例如 [0.5, 0.5]），``max`` 会返回第一个。
        评测时若 ``meta["gold_label"]`` 存在，会优先使用官方答案名（见 ``ajev.metrics.gold_index``）。
        """
        return max(range(len(self.target)), key=self.target.__getitem__)

    def shuffled(self, rng: random.Random) -> "Decision":
        """返回一份“选项顺序被随机打乱”的副本，target 会跟着一起重新排列。

        为什么需要：很多公开数据集里，正确答案总在固定位置（比如总是第 1 个）。
        模型可能偷懒学会“选第 1 个”，而不是真正理解题目，这叫“位置偏置”。
        打乱顺序后，“选项名 ↔ 概率”的对应关系不变，只有位置变了，模型就没法靠位置作弊。

        score 题的等级是有顺序的（0 < 1 < 2 …），打乱会破坏“等级越高越严重”的含义，
        所以 score 题原样返回、不打乱。

        参数:
            rng: 随机数生成器。由调用方传入（而不是在函数里随便新建），
                 这样只要种子相同，每次打乱的结果都一样，实验可以复现。
        返回:
            新的 Decision；score 题直接返回自身。

        举个例子:
            选项 [A, B, C]、target [0.7, 0.2, 0.1]，若随机排列 perm = [2, 0, 1]，
            则新选项为 [C, A, B]、新 target 为 [0.1, 0.7, 0.2]——A 仍然对应 0.7。
        """
        if self.type == "score":
            return self
        perm = list(range(len(self.options)))  # perm = [0, 1, 2, ...]
        rng.shuffle(perm)  # 原地随机打乱，例如变成 [2, 0, 1]
        # 第 1 步：asdict(self) 把当前对象转成字典（包含所有字段）；
        # 第 2 步：{**原字典, "options": 新值, "target": 新值} 复制字典并覆盖这两个字段；
        # 第 3 步：Decision(**字典) 用这些字段创建一个新对象。
        # 新位置 j 上放原来第 perm[j] 个选项，target 用同一个 perm 重排，保证两者一一对应。
        return Decision(
            **{**asdict(self), "options": [self.options[i] for i in perm], "target": [self.target[i] for i in perm]}
        )

    # ---- 序列化 / 反序列化 -----------------------------------------------
    # “序列化”指把内存里的对象转成可以保存到文件或通过网络传输的格式（这里是 dict → JSON 文本）；
    # “反序列化”是反过来，把文件里的内容还原成对象。
    def to_dict(self) -> dict[str, Any]:
        """把 Decision 转成普通字典，以便写成 JSON。

        target 只保留 6 位小数：例如 0.333333333333 → 0.333333，可以明显减小文件体积，
        而精度对训练没有影响。
        """
        d = asdict(self)
        d["target"] = [round(p, 6) for p in self.target]
        return d

    # @classmethod 表示“类方法”：第一个参数 cls 是类本身（Decision），而不是某个对象。
    # 常用来写“另一种创建对象的方式”，调用时写 Decision.from_dict(...)。
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        """从字典（通常是 JSONL 文件里的一行）还原出 Decision。

        options 从 dict 到 Option 的转换在 ``__post_init__`` 中自动完成。
        """
        return cls(**d)


def one_hot(k: int, i: int) -> list[float]:
    """生成长度为 k、只有第 i 位为 1 的 one-hot 分布，用来把“硬标签”转成 target。

    举个例子: one_hot(4, 2) → [0.0, 0.0, 1.0, 0.0]   （4 个选项，正确答案是第 3 个）
    """
    return [1.0 if j == i else 0.0 for j in range(k)]


def normalize(ps: Iterable[float]) -> list[float]:
    """把一组非负数“归一化”成概率分布：每个数除以总和，使它们加起来等于 1。

    特殊情况处理:
        * 负数会被当成 0（概率不可能为负）；
        * 如果全部是 0，没法除（除以 0 会出错），就退化成均匀分布，保证总能得到合法的分布。

    举个例子:
        normalize([2, 1, 1])   → [0.5, 0.25, 0.25]
        normalize([0, 0])      → [0.5, 0.5]
        normalize([0.3, -0.1]) → [1.0, 0.0]
    """
    ps = [max(0.0, float(p)) for p in ps]
    s = sum(ps)
    # 条件表达式：A if 条件 else B。
    return [p / s for p in ps] if s > 0 else [1.0 / len(ps)] * len(ps)


def jev_confidence(probs: list[float]) -> float:
    """计算 Jev 定义的 confidence（置信度）：``1 - H(p) / ln K``。

    直观理解:
        熵（entropy）H(p) = -Σ p_i · ln(p_i) 衡量一个分布“有多不确定”：
            * 分布越集中（比如 [0.98, 0.01, 0.01]），熵越小，说明模型越确定；
            * 分布越平均（比如 [0.33, 0.33, 0.34]），熵越大，说明模型越拿不准。
        K 个选项时，熵最大的情况是均匀分布，此时 H = ln K。
        所以 H / ln K 是“不确定程度占最大可能值的比例”（0～1），用 1 减去它就得到“确定程度”。

        除以 ln K 的好处是：二选一和十选一的题算出来的置信度都在 [0, 1] 之间，可以互相比较。

    举个例子（K = 2，ln 2 ≈ 0.693）:
        [1.0, 0.0] → H = 0                                  → confidence = 1（完全确定）
        [0.5, 0.5] → H = 0.693                              → confidence = 0（完全拿不准）
        [0.9, 0.1] → H = -(0.9·ln0.9 + 0.1·ln0.1) ≈ 0.325    → confidence ≈ 1 - 0.325/0.693 ≈ 0.53

    参数:
        probs: 预测的概率分布。
    返回:
        [0, 1] 之间的置信度；只有 1 个选项时没有不确定性，直接返回 1。
    """
    k = len(probs)
    if k < 2:
        return 1.0
    # ln(0) 是负无穷，没法计算；但数学上约定 0 · ln(0) = 0，所以直接跳过 p == 0 的项。
    h = -sum(p * math.log(p) for p in probs if p > 0)
    # 浮点数计算有微小误差，结果可能是 -0.0000001 这样略小于 0 的数，用 max 截到 0。
    return max(0.0, 1.0 - h / math.log(k))


def read_jsonl(path: str) -> list[Decision]:
    """读取 JSONL 文件（每行一个 Decision），返回 Decision 列表。空行会被跳过。

    ``with open(...) as f:`` 是 Python 打开文件的标准写法：离开 with 代码块时会自动关闭文件，
    即使中途出错也不会忘记关闭。``encoding="utf-8"`` 保证中文能正确读取。
    """
    with open(path, encoding="utf-8") as f:
        # 逐行遍历文件 → json.loads 把一行文本解析成字典 → from_dict 还原成 Decision。
        # line.strip() 去掉首尾空白，空行去掉后是 ""，在 if 判断中为 False，于是被跳过。
        return [Decision.from_dict(json.loads(line)) for line in f if line.strip()]


def write_jsonl(path: str, decisions: Iterable[Decision]) -> int:
    """把一组 Decision 逐行写成 JSONL 文件，返回写入的条数。

    ``ensure_ascii=False`` 让中文按原样写出（“体育”），而不是转义成 ``\\u4f53\\u80b2`` 这种形式，
    文件更小，也方便直接打开查看。
    """
    n = 0
    with open(path, "w", encoding="utf-8") as f:  # "w" 表示写模式（会覆盖已有文件）
        for d in decisions:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")  # 每条记录占一行
            n += 1
    return n


# ---- Jev 请求/响应格式（wire format）的转换 ------------------------------------
# 下面几个函数负责在“Jev 的 JSON 格式”和“我们内部的 Decision”之间来回转换。


def state_to_text(state: Any) -> str:
    """把 Jev 的 state（材料）统一转成字符串，因为模型只能读文本。

    Jev 允许 state 是字符串、JSON 对象（dict）或 JSON 数组（list）：
        * 字符串：原样返回；
        * 其他：转成紧凑的 JSON 文本。``separators=(",", ":")`` 去掉逗号和冒号后面的空格，
          能少占一些 token（模型输入长度有上限，省下来的空间可以多放材料内容）。

    举个例子:
        state_to_text("你好")                 → "你好"
        state_to_text({"a": 1, "b": "中文"})  → '{"a":1,"b":"中文"}'
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _options_from_jev_question(q: dict[str, Any], lang: str) -> list[Option]:
    """把一个 Jev 问题里的 criteria 转成选项列表。（函数名以下划线开头，表示仅供本模块内部使用。）

    三种题型的 criteria 格式各不相同:
        * noul:   可选的 ``{"true": 说明, "false": 说明}``；没给时用 ``DEFAULT_NOUL_DESC`` 里的默认说明。
                  选项固定按 [true, false] 的顺序生成。
        * choice: ``{选项名: 说明}`` 的字典，按字典里的顺序生成选项。
        * score:  等级说明的列表，第 i 个等级的选项名是字符串 ``"i"``，说明就是该等级的文字。

    举个例子:
        {"type": "score", "criteria": ["不急", "一般", "很急"]}
        → [Option("0", "不急"), Option("1", "一般"), Option("2", "很急")]
    """
    qtype, criteria = q["type"], q.get("criteria")  # dict.get(键) 在键不存在时返回 None，不会报错
    if qtype == "noul":
        criteria = criteria or {}  # 如果 criteria 是 None，就用空字典代替
        return [
            # “A or B”：A 为空（None 或 ""）时取 B，即“有自定义说明就用自定义的，否则用默认的”。
            Option(NOUL_TRUE, criteria.get(NOUL_TRUE) or DEFAULT_NOUL_DESC[lang][NOUL_TRUE]),
            Option(NOUL_FALSE, criteria.get(NOUL_FALSE) or DEFAULT_NOUL_DESC[lang][NOUL_FALSE]),
        ]
    if qtype == "choice":
        # .items() 同时遍历字典的键和值。
        return [Option(name, desc or "") for name, desc in criteria.items()]
    if qtype == "score":
        # enumerate 在遍历时同时给出序号 i（从 0 开始）和元素。
        return [Option(str(i), level) for i, level in enumerate(criteria)]
    raise ValueError(f"unknown question type {qtype!r}")


def decisions_from_jev(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,  # 星号之后的参数必须用“参数名=值”的形式传入，防止传错位置
    group: str = "req",
    source: str = "jev",
    lang: str = "en",
    gold: dict[str, dict[str, Any]] | None = None,
) -> list[Decision]:
    """把一个 Jev 请求（一段 state + 多个具名问题）拆成多个 Decision。

    主要有两个用途:
        1. 部署服务时：收到 Jev 格式的请求，拆成 Decision 交给模型打分；
        2. 构建数据时：typed-decisions 数据集本身就是 Jev 格式，带上 ``gold``（标准答案）
           就能直接得到训练题或测试题。

    参数:
        state:     材料，字符串或 JSON 对象/数组。
        questions: ``{问题id: {"type": ..., "instructions": ..., "criteria": ...}}``。
        group:     这批题共享的 group 名，同时作为 id 的前缀：id = ``"{group}/{问题id}"``。
        source:    数据来源标记。
        lang:      语言，决定 noul 题的默认说明用中文还是英文。
        gold:      可选的标准答案，格式为 ``{问题id: {"probabilities": {选项名: 概率}}}``。
                   给出时，按选项顺序取出概率作为 target；没给时 target 设为均匀分布
                   （推理时用不到 target，只是占个位置，保证 Decision 是合法的）。
    返回:
        Decision 列表，顺序与 questions 一致。每道题的 ``meta["question_id"]`` 记录原问题名，
        方便模型预测完以后再拼回 Jev 响应。

    举个例子:
        decisions_from_jev({"msg": "我的包裹在哪"},
                           {"topic": {"type": "choice", "instructions": "关于什么？",
                                      "criteria": {"billing": "扣费", "delivery": "物流"}}},
                           gold={"topic": {"probabilities": {"billing": 0.2, "delivery": 0.8}}})
        → [Decision(id="req/topic", type="choice", state='{"msg":"我的包裹在哪"}',
                    options=[billing, delivery], target=[0.2, 0.8], ...)]
    """
    text = state_to_text(state)  # 所有问题共用同一段材料，只需转换一次
    out = []
    for qid, q in questions.items():
        options = _options_from_jev_question(q, lang)
        if gold and qid in gold:
            probs = gold[qid]["probabilities"]
            # 按选项顺序依次取概率；gold 里没有的选项记为 0；
            # 最后再归一化一次，消除四舍五入带来的微小误差（比如总和是 0.999999）。
            target = normalize(probs.get(o.name, 0.0) for o in options)
        else:
            # [x] * n 会得到 n 个 x 组成的列表，例如 [0.5] * 2 == [0.5, 0.5]。
            target = [1.0 / len(options)] * len(options)
        out.append(
            Decision(
                id=f"{group}/{qid}",
                source=source,
                type=q["type"],
                state=text,
                instructions=q.get("instructions", ""),  # 没有 instructions 时用空字符串
                options=options,
                target=target,
                lang=lang,
                group=group,
                meta={"question_id": qid},
            )
        )
    return out


def jev_answer(decision: Decision, probs: list[float]) -> dict[str, Any]:
    """把模型对一道题的预测概率，转换成 Jev 响应中该问题的 answer 格式。

    三种题型的输出格式与 Jev 官方保持一致:
        * noul:   ``{"noul": P(true)}``
        * choice: ``{"choice": 概率最大的选项名, "probabilities": {选项名: 概率}, "confidence": 置信度}``
        * score:  ``{"score": 期望等级, "legend": 各等级说明, "probabilities": [...], "confidence": 置信度}``

    参数:
        decision: 被预测的那道题（提供选项名和顺序）。
        probs:    与 decision.options 顺序一致的预测概率。
    返回:
        可以直接放进 Jev 响应 ``answers[问题id]`` 的字典，数值保留 4 位小数。

    举个例子:
        noul 题，probs = [0.9, 0.1]（选项顺序 true, false）     → {"noul": 0.9}
        choice 题，选项 [billing, delivery]，probs = [0.2, 0.8] → {"choice": "delivery", ...}
        score 题，probs = [0.0, 0.5, 0.5]                        → {"score": 1.5, ...}
                  （0×0.0 + 1×0.5 + 2×0.5 = 1.5，表示“在 1 级和 2 级之间”）
    """
    names = decision.option_names
    # zip 把两个列表按位置配对：zip(["a","b"], [0.2,0.8]) → ("a",0.2), ("b",0.8)；
    # 再用字典推导式组成 {选项名: 概率}。
    pmap = {n: p for n, p in zip(names, probs)}
    conf = round(jev_confidence(probs), 4)
    if decision.type == "noul":
        # 按名字取 P(true) 而不是按位置取 probs[0]，所以即使选项顺序被打乱也不会取错。
        return {"noul": round(pmap[NOUL_TRUE], 4)}
    if decision.type == "choice":
        best = names[max(range(len(probs)), key=probs.__getitem__)]  # argmax 对应的选项名
        return {"choice": best, "probabilities": {n: round(p, 4) for n, p in pmap.items()}, "confidence": conf}
    # score 题：计算期望等级 E = Σ i × p_i，是一个连续值（例如 1.37）。
    # 它比直接取 argmax 更能反映不确定性：[0, 0.5, 0.5] 的 argmax 是 1，但期望是 1.5。
    score = sum(i * p for i, p in enumerate(probs))
    return {
        "score": round(score, 4),
        "legend": [o.desc for o in decision.options],
        "probabilities": [round(p, 4) for p in probs],
        "confidence": conf,
    }
