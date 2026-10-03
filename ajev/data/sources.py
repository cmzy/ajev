"""公开 Hugging Face 数据集 → ``Decision`` 记录的转换器。

在 AJev 流程（数据构建 → 训练 → 校准 → 评测）中，本模块处于最前端：它负责把
二十多个结构各异的公开数据集，统一改写成同一种“带类型的决策题”格式（见
``ajev.schema.Decision``），供 ``ajev.data.build`` 混合、去重后写成 JSONL。

核心概念：

* ``Source``：描述一个数据源——数据在 HF Hub 上的位置（path / config）、训练用哪个
  split、评测用哪个 split、数据语言，以及把一行原始数据变成 Decision 的转换函数。
* 转换器（converter）：签名为 ``(row, i, ctx) -> Decision | None``。返回 None 表示
  这一行不可用（例如无标签的测试样本），会被跳过。
* 题型映射：
    - noul（是/否题）：BoolQ、复述判断（QQP / PAWS / AFQMC）、有害内容判断、NLI 的二分类形式；
    - choice（单选题）：NLI 三分类、意图识别、主题 / 情绪分类、常识多选题；
    - score（打分题）：Yelp 星级、STS-B 语义相似度、HelpSteer2 有用程度。

设计取舍：

* 指令（instructions）从多条同义改写的模板里随机抽取，避免模型只记住某一种问法；
* 英文数据源会以 ``zh_instr_prob`` 的概率换成中文指令，覆盖“中文提问 + 英文材料”
  的跨语言场景（中文数据源始终使用中文指令）；
* 所有随机性都来自按 (数据源名, split, seed) 派生的 ``random.Random``，保证同一个
  seed 重新构建时得到完全相同的数据。

------------------------------------------------------------------------------
给初学者的背景知识
------------------------------------------------------------------------------

【Hugging Face datasets 是什么】
Hugging Face Hub（huggingface.co）是一个公开的模型 / 数据集仓库。Python 库 ``datasets``
可以用一行 ``load_dataset("google/boolq")`` 把数据集下载到本地缓存（~/.cache/huggingface），
第二次再调用时直接读缓存，不会重复下载。加载出来的对象可以像列表一样遍历，每一行是一个字典，
例如 BoolQ 的一行是 ``{"question": "...", "passage": "...", "answer": True}``。
有些数据集包含多个子集，用第二个参数 config 指定，例如 GLUE 有 mnli、qqp 等子集。

【split（数据划分）是什么，为什么不能用测试集训练】
一个数据集通常被切成几份：
  * train（训练集）：给模型学习用的题目和答案；
  * validation（验证集）：训练过程中用来“模拟考试”，挑选效果最好的 checkpoint、调整超参数；
  * test（测试集）：最终“正式考试”，只在最后用一次，衡量模型的真实水平。
如果拿测试题训练模型，就像考前偷看了考卷答案，分数会虚高，不能反映模型在新题上的能力。
所以本模块每个数据源都区分 train_split（训练用）和 eval_split（评测用）。

【题型与 target（目标分布）】
每道 Decision 都有若干选项，``target`` 是“各选项正确的概率”，所有数加起来等于 1：
  * one-hot（硬标签）：只有正确答案为 1，其余为 0，例如 3 个选项、答案是第 2 个 → [0, 1, 0]；
  * soft label（软标签）：概率分散在多个选项上，例如 [0.1, 0.7, 0.2]，表示多名标注者意见不一，
    或者分数介于两个等级之间。软标签能让模型学会“不确定时就说不确定”，这正是 Jev 强调的校准能力。

【为什么要用多种问法模板】
如果所有 BoolQ 题都用同一句 “{q}?”，模型可能学会“看到这种句式就怎样回答”的捷径，
换个问法就不会了。真实用户的问法千差万别，所以每类题准备几种同义说法，随机挑一种。

【为什么要打乱选项顺序（位置偏置）】
很多数据集的正确答案位置有规律（例如候选列表里正确答案总放在最后）。模型很聪明，
会偷懒学成“总选最后一个”。这种现象叫位置偏置（position bias）。把选项顺序随机打乱后，
模型只能靠理解选项内容来作答。本模块里的候选项会打乱，build.py 和训练时也会再打乱。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ajev.schema import (
    DEFAULT_NOUL_DESC,
    NOUL_FALSE,
    NOUL_TRUE,
    Decision,
    Option,
    one_hot,
    state_to_text,
)

# 一行原始数据：HF datasets 迭代时返回的字典（字段名 → 值）。
Row = dict[str, Any]


# @dataclass 是 Python 的装饰器：只需写出“字段名: 类型”，它会自动生成 __init__ 等方法，
# 适合用来定义“只存数据”的小类。例如 Ctx(rng=..., source=..., label_names=[...]) 就能直接创建对象。
@dataclass
class Ctx:
    """每次转换调用时传给转换器的上下文。

    为什么需要它：转换器除了原始行，还需要知道“随机数用哪个”“类别名是什么”“当前是哪个数据源”
    等信息。把这些打包成一个对象传进去，比给每个转换器加一长串参数更整洁。

    属性：
        rng: 本数据源 / 本 split 专用的随机数生成器（用于抽模板、抽候选项、决定指令语言等），
            由 ``Source.iter_decisions`` 按 (name, split, seed) 创建，保证可复现。
        source: 当前所属的 ``Source``，转换器用它拿到数据源名称（用于生成 id）和数据语言。
        label_names: 标签名列表，下标与原始数据中的整数标签一一对应；字符串标签的数据集
            （如意图识别）则是 split 中出现过的全部标签（已排序）。
        zh_instr_prob: 英文数据源改用中文指令的概率。
    """

    rng: random.Random
    source: "Source"
    label_names: list[str]
    zh_instr_prob: float = 0.0

    def instr_lang(self) -> str:
        """决定本条样本的指令语言。

        中文数据源一律用中文指令；英文数据源以 ``zh_instr_prob`` 的概率用中文指令，
        其余用英文——这样训练集中会出现“中文问题 + 英文 state”的跨语言样本。
        注意：Decision 的 ``lang`` 字段仍记录数据源（state）的语言，而不是指令语言。
        """
        if self.source.lang == "zh":
            return "zh"
        return "zh" if self.rng.random() < self.zh_instr_prob else "en"

    def pick(self, templates: dict[str, list[str]], lang: str, **kw: str) -> str:
        """从某语言的模板列表中随机选一条，并用关键字参数填充占位符（如 ``{q}``、``{h}``）。

        举个例子：
            templates = {"en": ["{q}?", "Based on the passage: {q}?"], "zh": [...]}
            ctx.pick(templates, "en", q="is the sky blue")
            → 随机得到 "is the sky blue?" 或 "Based on the passage: is the sky blue?"
        ``**kw`` 表示“任意多个关键字参数”，会被传给字符串的 ``format`` 方法替换花括号占位符。
        """
        return self.rng.choice(templates[lang]).format(**kw)


# 转换器类型：输入 (原始行, 行号, 上下文)，输出一个 Decision；返回 None 表示跳过该行。
# Callable[[参数类型...], 返回类型] 是类型注解，表示“一个可以调用的函数”，只用于提示和阅读，不影响运行。
Converter = Callable[[Row, int, Ctx], "Decision | list[Decision] | None"]


@dataclass
class Source:
    """一个公开数据源的描述。

    属性：
        name: 数据源在 AJev 内部的名字（也是 Decision.source 与 id 前缀），如 ``"boolq"``。
        path: HF Hub 上的数据集路径，如 ``"google/boolq"``。
        config: 数据集的 config / 子集名（没有则为 None），如 GLUE 的 ``"mnli"``。
        train_split: 用来生成训练数据的 split。
        eval_split: 用来生成验证 / 测试数据的 split（build 时前一半进 val，后一半进 test_public）。
            有些数据集的 test split 没有公开标签，因此这里通常选 validation。
        lang: 数据（state）本身的语言，``"en"`` 或 ``"zh"``。
        convert: 把一行原始数据转换为 Decision 的函数。可以返回一个 Decision、一个 Decision 列表
            （一行原始数据产出多道题，比如一张客服工单同时问“分到哪个队列”和“优先级多高”），
            或 None（跳过这一行）。
        label_field: 标签所在的字段名，用于推断 ``label_names``（类别名列表）。
        tags: 预留的标签字段，目前未使用。
        data_files: 按文件加载时的 {split 名: 文件名}。个别数据集的各个文件列不一致
            （例如 COLD 的 test.csv 多一列），整体加载会报错，只能逐个文件加载。
        train_cap: 本数据源的训练采样上限；None 表示用 build.py 的全局 ``--train-cap``。
            大数据源（如 bev-decision）想多采一些、小数据源想少采一些时用它。

    有些数据集只有一个 split，需要自己切出训练和评测两部分。这时 split 写成 ``"<原 split>#train"`` /
    ``"<原 split>#eval"``，并用 ``holdout`` 指定评测部分占的比例（例如 0.1）。
    切分前会用**固定种子**整体打乱，所以两部分的分布一致，而且每次构建结果都一样。

    为什么不用 ``"train[:90%]"`` / ``"train[90%:]"`` 这种切片：切片是按原始顺序取的，
    而很多数据集是排好序的。例如 When2Call 的 mcq 按答案类型排序，后 15% 全是 tool_call，
    用它做评测得到的准确率毫无意义（我们第一次就踩了这个坑：When2Call 只有 0.34）。
    """

    name: str
    path: str
    config: str | None
    train_split: str | None  # None = 只用于评测，不产出训练数据（例如被公开排行榜用作测试集的数据）
    eval_split: str
    lang: str
    convert: Converter
    label_field: str = "label"
    # 有些数据源只用于评测（例如 typed-decisions 的测试集在别处单独加载）。
    tags: list[str] = field(default_factory=list)
    data_files: dict[str, str] | None = None
    train_cap: int | None = None
    holdout: float = 0.1
    # "#train" / "#eval" 切分时按什么分组：同一组（例如同一篇材料配的多道题）整组进训练或整组进评测，
    # 否则同一篇材料会同时出现在训练和评测里（RAGTruth 同一则新闻有 6 个模型的回答，MMLU 辅助集同一篇阅读配多道题）。
    # None = 按行随机切分。
    group_fn: Callable[[dict], str] | None = None

    def load(self, split: str):
        """从 HF Hub（或本地缓存）加载指定 split。``datasets`` 延迟导入，避免只用 schema 时也要装它。"""
        from datasets import load_dataset

        base, _, part = split.partition("#")
        if self.data_files:
            # 按文件加载：split 名是 data_files 的键；只加载这一个文件，避免列不一致的报错。
            key = base.split("[")[0]
            ds = load_dataset(self.path, data_files={key: self.data_files[key]}, split=base)
        else:
            ds = load_dataset(self.path, self.config, split=base)
        # 图片列直接去掉：我们只用文字（例如 New Yorker 漫画用的是文字描述），也省得装图片解码库。
        from datasets import Image

        img = [c for c, f in ds.features.items() if isinstance(f, Image)]
        if img:
            ds = ds.remove_columns(img)
        if not part:
            return ds
        # "#train" / "#eval"：按分组切分时，组名的哈希决定整组归属（与顺序无关、可复现）。
        if self.group_fn is not None:
            import hashlib

            cut = int(self.holdout * 10000)
            in_eval = lambda row: int(hashlib.sha1(self.group_fn(row).encode()).hexdigest(), 16) % 10000 < cut
            ds = ds.filter(lambda row: in_eval(row) == (part == "eval"))
            if part not in ("train", "eval"):
                raise ValueError(f"unknown split part {part!r} in {split!r}")
            return ds.shuffle(seed=12345)
        # 否则用固定种子打乱后切分（种子与 build 的 --seed 无关，保证训练和评测永远不重叠）。
        ds = ds.shuffle(seed=12345)
        n_eval = int(len(ds) * self.holdout)
        if part == "eval":
            return ds.select(range(n_eval))
        if part == "train":
            return ds.select(range(n_eval, len(ds)))
        raise ValueError(f"unknown split part {part!r} in {split!r}")

    def iter_decisions(
        self, split: str, limit: int | None, seed: int, zh_instr_prob: float = 0.0
    ) -> Iterator[Decision]:
        """逐条产出某个 split 转换后的 Decision。

        参数：
            split: 要读取的 split 名。
            limit: 最多产出多少条有效 Decision（被转换器跳过的行不计数）；None 表示不限。
            seed: 随机种子，决定打乱顺序以及模板 / 候选项 / 指令语言的抽取。
            zh_instr_prob: 英文数据源改用中文指令的概率。

        每条 Decision 产出前都会调用 ``validate()`` 做格式校验，有问题会直接抛错，
        以便在构建阶段尽早发现转换器的 bug。

        关于 ``yield``（生成器）：这个函数不是一次性返回一个大列表，而是“算出一条交出一条”。
        调用方用 for 循环或 list(...) 逐条取用，好处是省内存，而且调用方取够了就可以停下，
        后面的行根本不会被转换（build.py 里配合 islice 使用）。

        举个例子：
            SOURCES["boolq"].iter_decisions("train", limit=2, seed=0)
            → 依次产出 2 个 Decision，例如 id 为 "boolq/0"、"boolq/1"（编号是打乱后的行号）。

        处理步骤：
            第 1 步：加载 split；
            第 2 步：推断类别名列表 label_names（整数标签 → 名字的对照表）；
            第 3 步：用 seed 打乱数据；
            第 4 步：逐行调用转换器，跳过返回 None 的行，校验后产出，直到达到 limit。
                     转换器返回列表时（一行产出多道题），逐道产出，每道题都计入 limit。
        """
        # 第 1 步：加载数据。
        ds = self.load(split)
        # 第 2 步：推断类别名。ds.features 描述每个字段的类型，
        # 例如 MNLI 的 label 字段是 ClassLabel(names=["entailment", "neutral", "contradiction"])。
        feat = ds.features.get(self.label_field)
        # ClassLabel 类型的字段自带类别名列表（names），下标即整数标签。
        label_names = list(getattr(feat, "names", []) or [])
        if not label_names and getattr(feat, "dtype", None) == "string":
            # 字符串标签（如意图名）：没有预定义类别表，就取该 split 中实际出现过的全部标签，
            # 排序后作为候选池（排序保证不同运行之间顺序一致、可复现）。
            label_names = sorted(set(ds[self.label_field]))
        # 第 3 步：先整体打乱再按 limit 截取：很多数据集是按类别排好序的（如 dbpedia），
        # 直接取前 N 条会导致类别严重失衡（例如前 3000 条全是 "Company" 类）。
        ds = ds.shuffle(seed=seed)
        # 每个 (数据源, split, seed) 一个独立的随机数生成器：增减其他数据源不会影响本数据源的结果。
        rng = random.Random(f"{self.name}/{split}/{seed}")
        ctx = Ctx(rng=rng, source=self, label_names=label_names, zh_instr_prob=zh_instr_prob)
        # 第 4 步：逐行转换。n 统计已产出的有效条数；i 是行号（enumerate 同时给出下标和元素）。
        n = 0
        for i, row in enumerate(ds):
            if limit is not None and n >= limit:
                break
            out = self.convert(row, i, ctx)
            if out is None:
                continue
            # 统一成列表处理：单个 Decision 包成只有一个元素的列表。
            for d in out if isinstance(out, list) else [out]:
                if limit is not None and n >= limit:
                    break
                d.validate()
                n += 1
                yield d


# ---- 通用构造函数：按题型生成 Decision -----------------------------------------------


def _noul(ctx: Ctx, i: int, state: str, instructions: str, answer: bool, lang: str, **meta: Any) -> Decision:
    """构造一道 noul（是/否）题。

    选项固定为 ``true`` / ``false``（描述使用指令语言 ``lang`` 的默认文案），
    target 为硬标签：答案为真时 ``[1, 0]``，否则 ``[0, 1]``。
    ``lang`` 只决定选项描述的语言；Decision 的 ``lang`` 字段取数据源语言。

    举个例子（BoolQ 的一行，行号 i=7）：
        _noul(ctx, 7, state="Persian is spoken in Iran and Afghanistan ...",
              instructions="do iran and afghanistan speak the same language?", answer=True, lang="en")
        → Decision(id="boolq/7", type="noul",
                   options=[Option("true", "Yes, the statement holds."),
                            Option("false", "No, the statement does not hold.")],
                   target=[1.0, 0.0], ...)
    ``**meta`` 收集额外的关键字参数，原样放进 Decision.meta，用于记录附加信息。
    """
    desc = DEFAULT_NOUL_DESC[lang]
    return Decision(
        id=f"{ctx.source.name}/{i}",
        source=ctx.source.name,
        type="noul",
        state=state,
        instructions=instructions,
        options=[Option(NOUL_TRUE, desc[NOUL_TRUE]), Option(NOUL_FALSE, desc[NOUL_FALSE])],
        target=[1.0, 0.0] if answer else [0.0, 1.0],
        lang=ctx.source.lang,
        meta=meta,
    )


def _choice(ctx: Ctx, i: int, state: str, instructions: str, options: list[Option], gold: int) -> Decision:
    """构造一道 choice（单选）题，``gold`` 为正确选项在 ``options`` 中的下标，target 为 one-hot。

    选项顺序在这里保持原样；build 阶段和训练阶段会再随机打乱，防止模型学到“正确答案总在某个位置”。

    举个例子：options 有 4 个（World / Sports / Business / Sci/Tech），gold=2
        → target = one_hot(4, 2) = [0.0, 0.0, 1.0, 0.0]，表示 Business 是正确答案。
    """
    return Decision(
        id=f"{ctx.source.name}/{i}",
        source=ctx.source.name,
        type="choice",
        state=state,
        instructions=instructions,
        options=options,
        target=one_hot(len(options), gold),
        lang=ctx.source.lang,
    )


def _score(ctx: Ctx, i: int, state: str, instructions: str, levels: list[str], target: list[float]) -> Decision:
    """构造一道 score（打分）题。

    ``levels`` 是从低到高的等级描述，选项名依次为 ``"0"``、``"1"``……（等级有序，永不打乱）；
    ``target`` 是各等级上的概率分布（可以是 one-hot，也可以是插值得到的软标签）。

    为什么打分题不打乱：等级之间有大小关系（0 < 1 < 2 ...），打乱后“3 级比 2 级高”的含义就乱了，
    而且训练中针对打分题的 RPS 损失依赖等级顺序。

    举个例子（Yelp 的一条 4 星评论，label=3）：
        levels = REVIEW_LEVELS["en"]（5 条从“非常负面”到“非常正面”的描述）
        → options = [Option("0", "Very negative: ..."), ..., Option("4", "Very positive: ...")]
        → target = [0, 0, 0, 1, 0]
    这里用 enumerate(levels) 同时拿到下标 k 和描述 lvl，下标转成字符串作为选项名。
    """
    return Decision(
        id=f"{ctx.source.name}/{i}",
        source=ctx.source.name,
        type="score",
        state=state,
        instructions=instructions,
        options=[Option(str(k), lvl) for k, lvl in enumerate(levels)],
        target=target,
        lang=ctx.source.lang,
    )


def _interp_target(value: float, k: int) -> list[float]:
    """把 [0, k-1] 区间内的连续分值线性分摊到相邻的两个等级上，得到软标签。

    例：k=6、value=3.25 → 等级 3 得 0.75，等级 4 得 0.25。
    这样既保留了“3.25 比 3 略高”的信息，分布的期望值也恰好等于原始分值。
    超出范围的值会先被截断到 [0, k-1]。

    逐步计算（value=3.25, k=6）：
        第 1 步：截断：3.25 在 [0, 5] 内，不变；
        第 2 步：lo = int(3.25) = 3（向下取整），hi = 4；
        第 3 步：value - lo = 0.25，表示离 3 级“往上走了四分之一”；
                 于是 t[3] = 1 - 0.25 = 0.75，t[4] = 0.25；
        结果：[0, 0, 0, 0.75, 0.25, 0]。验证期望值：3×0.75 + 4×0.25 = 3.25 ✓。
    如果 value 恰好是整数（例如 5.0），lo = hi = 5，t[5] 先加 1 再加 0，结果就是 one-hot。

    为什么不直接四舍五入成 one-hot：3.25 和 3.49 都会变成 3 级，丢失信息；
    软标签告诉模型“这对句子大约在 3 级，略偏 4 级”，学到的分布更细腻、更容易校准。
    """
    value = min(max(value, 0.0), k - 1)
    lo = int(value)
    hi = min(lo + 1, k - 1)  # value 恰好等于 k-1 时，lo 与 hi 是同一等级
    t = [0.0] * k
    t[lo] += 1.0 - (value - lo)
    t[hi] += value - lo
    return t


def _sample_candidates(ctx: Ctx, gold: str, lo: int = 4, hi: int = 20) -> tuple[list[str], int]:
    """从完整标签表中随机抽一组候选项（一定包含正确答案），返回 (候选列表, 正确答案下标)。

    意图识别数据集的类别数很多（Banking77 有 77 个、CLINC 有 151 个），全部放进选项既超出
    选项 token 预算，也不符合真实使用场景。这里每题随机抽 ``lo``~``hi`` 个候选，
    让模型见到各种规模的选项集合；若标签总数不足则取全部。

    为什么要“随机抽候选”而不是固定选项：真实业务里，每次路由的候选意图都不一样（不同产品线、
    不同上下文给的选项不同）。每题换一组候选，能逼模型真正理解“这条消息和这个选项匹不匹配”，
    而不是记住“某些意图总是一起出现”。同时也锻炼模型在 5 个选项和 20 个选项时都能工作。

    举个例子：label_names 有 77 个意图，gold="card_arrival"，随机得到 k=5
        第 1 步：pool = 除 card_arrival 外的 76 个意图；
        第 2 步：从 pool 随机抽 4 个，再加上 gold，凑成 5 个；
        第 3 步：打乱顺序，例如 ["top_up_failed", "card_arrival", "exchange_rate", "pin_blocked", "refund_not_showing_up"]；
        返回 (上面的列表, 1)，1 是 card_arrival 在列表中的下标。
    """
    pool = [n for n in ctx.label_names if n != gold]
    k = min(ctx.rng.randint(lo, hi), len(pool) + 1)
    cands = ctx.rng.sample(pool, k - 1) + [gold]
    ctx.rng.shuffle(cands)  # 打乱，避免正确答案总在最后一个
    return cands, cands.index(gold)


def _humanize(label: str) -> str:
    """把 ``card_arrival`` 这类标签名变成可读文本 ``card arrival``，用作选项描述。

    这样模型看到的选项是 “card_arrival: card arrival”，描述部分是自然语言，更容易理解。
    """
    return label.replace("_", " ").strip()


# ---- 指令模板 ------------------------------------------------------------------------
# 每个模板字典按语言（"en" / "zh"）给出若干条同义问法，转换时随机抽一条。
# 占位符：{q} = 原始问题，{h} = 假设句（hypothesis）。
# 用反引号括起来的 `a`、`b`、`prompt`、`response` 指的是 state（JSON）中的同名字段，
# 与 Jev 用反引号引用 state 字段的写法一致。

# BoolQ：直接把原始问题作为是/否题的指令。
T_BOOLQ = {
    "en": ["{q}?", "Based on the passage: {q}?", "According to the text, {q}?"],
    "zh": ["根据这段文字回答：{q}？", "依据给定材料判断：{q}？"],
}
# NLI 的 noul 形式：判断前提（state）能否推出假设 {h}。
T_ENTAIL = {
    "en": [
        "The text implies that: {h}",
        "Does the passage support the claim \"{h}\"?",
        "Given the text, it is true that {h}",
    ],
    "zh": ["根据上文可以推出：{h}", "上文是否支持这一说法：“{h}”？", "由给定内容可知：{h}"],
}
# NLI 的 choice 形式：在 蕴含 / 中立 / 矛盾 三者中选一。
T_NLI3 = {
    "en": ["What is the relationship between the text and the claim \"{h}\"?",
           "How does the passage relate to this statement: {h}"],
    "zh": ["上文与陈述“{h}”是什么关系？", "给定内容和这句话的逻辑关系是什么：{h}"],
}
# NLI 三分类的选项描述（选项名保持英文标签名，描述随指令语言切换）。
NLI3_OPTS = {
    "en": {
        "entailment": "The text clearly supports the claim.",
        "neutral": "The text neither supports nor contradicts the claim.",
        "contradiction": "The text contradicts the claim.",
    },
    "zh": {
        "entailment": "上文明确支持该陈述。",
        "neutral": "上文既不支持也不否定该陈述。",
        "contradiction": "上文与该陈述矛盾。",
    },
}
# 复述 / 同义判断（QQP、PAWS、AFQMC）：state 为 {"a": ..., "b": ...}。
T_PARAPHRASE = {
    "en": ["The two texts in `a` and `b` mean the same thing.",
           "Do `a` and `b` ask or state the same thing?",
           "`a` is a paraphrase of `b`."],
    "zh": ["`a` 和 `b` 表达的是同一个意思。", "`a` 与 `b` 问的是同一个问题吗？", "`a` 是 `b` 的同义改写。"],
}
# 意图识别 / 请求路由（Banking77、CLINC150、MASSIVE）。
T_INTENT = {
    "en": ["What does the user want?", "Which intent best matches this user message?",
           "Route this request to the right intent."],
    "zh": ["用户想做什么？", "这条用户消息最符合哪个意图？", "把这条请求路由到正确的意图。"],
}
# 主题分类（AG News、DBpedia、TNEWS）。
T_TOPIC = {
    "en": ["What is this text mainly about?", "Which category does this article belong to?",
           "Classify the topic of this text."],
    "zh": ["这段文字主要讲的是什么？", "这篇文章属于哪个类别？", "给这段文本的主题分类。"],
}
# 情绪分类（dair-ai/emotion）。
T_EMOTION = {
    "en": ["Which emotion does the writer express?", "What is the dominant feeling in this text?"],
    "zh": ["作者表达了哪种情绪？", "这段文字的主要情感是什么？"],
}
# 常识 / 科学多选题（ARC、CommonsenseQA）：问题本身就是指令，state 为空。
T_MCQ = {
    "en": ["{q}", "Answer the question: {q}", "Pick the best answer. {q}"],
    "zh": ["{q}", "回答问题：{q}", "选出最合适的答案。{q}"],
}
# Yelp 评论满意度打分。
T_REVIEW = {
    "en": ["How satisfied is the reviewer?", "Rate the sentiment of this review.",
           "How positive is this customer review?"],
    "zh": ["评论者的满意度如何？", "给这条评论的情感打分。", "这条顾客评价有多正面？"],
}
# Yelp 的 5 个等级（对应 1~5 星）。Jev 要求打分等级描述“具体的情形”，而不是抽象的程度词。
REVIEW_LEVELS = {
    "en": ["Very negative: angry or strongly disappointed, would not return.",
           "Negative: more complaints than praise.",
           "Mixed or neutral: balanced good and bad points.",
           "Positive: satisfied with minor reservations.",
           "Very positive: enthusiastic, would strongly recommend."],
    "zh": ["非常负面：愤怒或极度失望，不会再来。", "负面：抱怨多于称赞。", "中性或褒贬参半。",
           "正面：基本满意，有小保留。", "非常正面：热情推荐。"],
}
# STS-B 语义相似度打分。
T_STS = {
    "en": ["How similar in meaning are `a` and `b`?", "Rate the semantic similarity of `a` and `b`."],
    "zh": ["`a` 和 `b` 的意思有多相近？", "给 `a` 与 `b` 的语义相似度打分。"],
}
# STS-B 的 6 个等级（0~5 分），措辞参考了 STS 原始标注指南。
STS_LEVELS = {
    "en": ["Completely unrelated.", "Not equivalent, but on the same topic.",
           "Not equivalent, but share some details.", "Roughly equivalent; some important information differs.",
           "Mostly equivalent; only minor details differ.", "Completely equivalent in meaning."],
    "zh": ["完全无关。", "不等价，但主题相同。", "不等价，但有部分细节相同。", "大致等价，但有重要信息不同。",
           "基本等价，仅细节不同。", "意思完全相同。"],
}
# HelpSteer2：评价回复（response）对提问（prompt）的有用程度。
T_HELPFUL = {
    "en": ["How helpful is the `response` to the `prompt`?", "Rate the overall helpfulness of the response."],
    "zh": ["`response` 对 `prompt` 有多大帮助？", "给这条回复的整体有用程度打分。"],
}
# HelpSteer2 helpfulness 的 5 个等级（0~4 分）。
HELPFUL_LEVELS = {
    "en": ["Not helpful at all: wrong, off-topic or refuses without reason.",
           "Slightly helpful: mostly misses what the user needed.",
           "Partially helpful: addresses the request with notable gaps or errors.",
           "Helpful: addresses the request well with minor issues.",
           "Extremely helpful: complete, correct and well suited to the request."],
    "zh": ["完全没帮助：错误、离题或无理拒答。", "略有帮助：大体没满足需求。", "部分有帮助：回应了请求但有明显缺漏或错误。",
           "有帮助：较好地完成请求，仅有小问题。", "非常有帮助：完整、正确且贴合需求。"],
}
# 内容审核（toxic-chat）：判断用户输入是否有害。
T_TOXIC = {
    "en": ["This user message is toxic, harmful or inappropriate.",
           "The message should be blocked by a content moderation filter.",
           "Is this request unsafe or offensive?"],
    "zh": ["这条用户消息有毒、有害或不当。", "内容审核应当拦截这条消息。", "这个请求是否不安全或冒犯？"],
}

# TNEWS 原始标签是数字编码（"100"~"116"，中间有缺号），这里映射为 (中文类别名, 类别描述)，
# 让选项对模型来说是有意义的文字，而不是无语义的编号。
TNEWS_NAMES = {
    "100": ("故事", "民间故事、生活故事"), "101": ("文化", "文化、艺术、历史"), "102": ("娱乐", "明星、影视、综艺"),
    "103": ("体育", "体育赛事与运动员"), "104": ("财经", "经济、金融、商业"), "106": ("房产", "房地产、楼市"),
    "107": ("汽车", "汽车、驾驶"), "108": ("教育", "学校、考试、教育"), "109": ("科技", "科技、互联网、数码"),
    "110": ("军事", "军事、国防"), "112": ("旅游", "旅行、景点"), "113": ("国际", "国际新闻与外交"),
    "114": ("股票", "股市、证券"), "115": ("农业", "农业、农村、农民"), "116": ("电竞", "电子竞技与游戏"),
}
# AG News 四个类别的选项描述。
AG_NEWS_DESC = {"World": "International news and politics.", "Sports": "Sport events and athletes.",
                "Business": "Companies, markets and the economy.", "Sci/Tech": "Science and technology."}


# ---- 各数据集的转换器 -----------------------------------------------------------------------


def conv_boolq(row: Row, i: int, ctx: Ctx) -> Decision:
    """BoolQ → noul 题。

    state = 段落 ``passage``；instructions = 用模板包装的 ``question``；
    target = ``answer``（True → true，False → false）。

    举个例子：
        原始行：{"question": "do iran and afghanistan speak the same language",
                 "passage": "Persian ... is primarily spoken in Iran, Afghanistan ...", "answer": True}
        → state = "Persian ... is primarily spoken in Iran, Afghanistan ..."
        → instructions = "According to the text, do iran and afghanistan speak the same language?"（随机模板之一）
        → type = "noul"，target = [1.0, 0.0]（true 正确）
    """
    lang = ctx.instr_lang()
    return _noul(ctx, i, row["passage"], ctx.pick(T_BOOLQ, lang, q=row["question"]), bool(row["answer"]), lang)


def _nli(premise_key: str, hyp_key: str) -> Converter:
    """生成 NLI（自然语言推理）数据集的转换器，用于 MNLI、ANLI、OCNLI、CMNLI。

    state = 前提句（``premise_key`` 字段）。每条样本以 50% 概率转成两种题型之一：

    * noul：“上文能否推出 {假设}”，只有 entailment 记为 true，neutral / contradiction 都记为 false；
    * choice：在 entailment / neutral / contradiction 三个选项中选一，选项描述随指令语言切换。

    同一份数据同时产出两种题型，让模型学会同一语义关系在不同问法下的一致回答。
    标签 < 0 的行（无标注样本，常见于测试集）返回 None 跳过。

    举个例子（MNLI 的一行）：
        原始行：{"premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
                 "hypothesis": "Product and geography are what make cream skimming work.", "label": 1}
        label_names = ["entailment", "neutral", "contradiction"]，所以 label=1 → name="neutral"。
        情况 A（50% 概率，noul）：
            state = premise，instructions = "The text implies that: Product and geography are ..."
            target = [0.0, 1.0]（neutral 不是 entailment，所以答案为 false）
        情况 B（50% 概率，choice）：
            state = premise，instructions = "What is the relationship between the text and the claim \"...\"?"
            options = [entailment, neutral, contradiction]（各带一句描述），target = [0, 1, 0]

    关于“函数返回函数”（闭包）：``_nli("premise", "hypothesis")`` 本身不处理数据，
    而是返回一个记住了字段名的 conv 函数。这样 MNLI（字段叫 premise/hypothesis）和
    OCNLI（字段叫 sentence1/sentence2）可以共用同一套逻辑，只是字段名不同。
    注意：不同数据集的标签顺序不同（GLUE 是 [entailment, neutral, contradiction]，
    CLUE 是 [neutral, entailment, contradiction]），所以统一用 ``label_names`` 把下标转成名字再处理。
    """

    def conv(row: Row, i: int, ctx: Ctx) -> Decision | None:
        # 第 1 步：没有标签（-1）的行不能用于训练或评测，跳过。
        if row["label"] < 0:
            return None
        # 第 2 步：整数标签 → 标签名，并决定指令语言。
        name = ctx.label_names[row["label"]]
        lang = ctx.instr_lang()
        premise, hyp = row[premise_key], row[hyp_key]
        # 第 3 步：抛硬币决定题型。rng.random() 返回 [0, 1) 的随机数，小于 0.5 的概率是 50%。
        if ctx.rng.random() < 0.5:
            return _noul(ctx, i, premise, ctx.pick(T_ENTAIL, lang, h=hyp), name == "entailment", lang)
        # choice 形式：选项名固定用统一的英文顺序，描述按指令语言取；列表推导式为每个名字建一个 Option。
        names = ["entailment", "neutral", "contradiction"]
        opts = [Option(n, NLI3_OPTS[lang][n]) for n in names]
        return _choice(ctx, i, premise, ctx.pick(T_NLI3, lang, h=hyp), opts, names.index(name))

    return conv


def _paraphrase(a_key: str, b_key: str, positive: Callable[[Row, Ctx], bool]) -> Converter:
    """生成复述 / 同义判断数据集的转换器，用于 QQP、PAWS、AFQMC。

    state = JSON ``{"a": 文本1, "b": 文本2}``，指令里用 `a`、`b` 引用这两个字段；
    target：``positive(row, ctx)`` 为真（两句同义）记为 true，否则 false。
    ``positive`` 由各数据源注册时传入，因为不同数据集表示“同义”的标签值不同。
    标签 < 0 的行返回 None 跳过。

    为什么把两段文本放进 JSON：Jev 的 state 支持 JSON 对象，问题里可以用反引号 `a`、`b`
    指代字段。这种“结构化材料 + 引用字段提问”的形式，和真实业务（state 是订单、工单等 JSON）一致。

    举个例子（AFQMC 的一行）：
        原始行：{"sentence1": "蚂蚁借呗等额还款可以换成先息后本吗", "sentence2": "借呗有先息到期还本吗", "label": 0}
        → state = '{"a":"蚂蚁借呗等额还款可以换成先息后本吗","b":"借呗有先息到期还本吗"}'
        → instructions = "`a` 与 `b` 问的是同一个问题吗？"
        → target = [0.0, 1.0]（label 0 表示不同义，所以答案 false）
    """

    def conv(row: Row, i: int, ctx: Ctx) -> Decision | None:
        if row["label"] < 0:
            return None
        lang = ctx.instr_lang()
        state = state_to_text({"a": row[a_key], "b": row[b_key]})
        return _noul(ctx, i, state, ctx.pick(T_PARAPHRASE, lang), positive(row, ctx), lang)

    return conv


def _intent(text_key: str, label_key: str) -> Converter:
    """生成意图识别数据集的转换器，用于 Banking77、CLINC150、MASSIVE（中 / 英）。

    state = 用户消息（``text_key`` 字段）；instructions = 意图类模板；
    options = 从全部意图中随机抽 4~20 个候选（必含正确意图），选项名为原始意图标签，
    描述为可读化后的标签名；CLINC 的 ``oos``（超出支持范围）给出专门的描述；
    target = 正确意图的 one-hot。
    标签字段可能是字符串（MASSIVE、Banking77 的 label_text），也可能是整数（CLINC 的 intent），
    整数时通过 ``label_names`` 转成名字。

    举个例子（MASSIVE 中文的一行）：
        原始行：{"text": "星期五早上九点叫醒我", "label": "alarm_set"}
        → state = "星期五早上九点叫醒我"
        → instructions = "用户想做什么？"（中文数据源固定用中文模板）
        → options = 例如 [Option("alarm_set", "alarm set"), Option("music_likeness", "music likeness"), ...]（随机 4~20 个）
        → target = 在 alarm_set 位置为 1 的 one-hot
    """

    def conv(row: Row, i: int, ctx: Ctx) -> Decision:
        # 条件表达式 “A if 条件 else B”：标签是字符串就直接用，是整数就查表转成名字。
        gold = row[label_key] if isinstance(row[label_key], str) else ctx.label_names[row[label_key]]
        cands, gi = _sample_candidates(ctx, gold)
        opts = [
            Option(c, "The request does not match any supported intent." if c == "oos" else _humanize(c))
            for c in cands
        ]
        return _choice(ctx, i, row[text_key], ctx.pick(T_INTENT, ctx.instr_lang()), opts, gi)

    return conv


def _topic(text_fn: Callable[[Row], str], desc: dict[str, str] | None = None, templates=T_TOPIC) -> Converter:
    """生成主题 / 情绪分类数据集的转换器，用于 AG News、DBpedia、emotion。

    参数：
        text_fn: 从原始行取出 state 文本的函数（例如 DBpedia 要把标题和正文拼起来）。
        desc: 可选的“类别名 → 描述”映射；没有则选项描述为空。
        templates: 指令模板，默认主题分类模板，emotion 数据集传入情绪模板。

    options = 数据集全部类别（类别数较少，不需要抽样）；target = ``label`` 的 one-hot。

    举个例子（AG News 的一行）：
        原始行：{"text": "Wall St. Bears Claw Back Into the Black (Reuters) ...", "label": 2}
        → state = 原文；options = [World, Sports, Business, Sci/Tech]（各带 AG_NEWS_DESC 中的描述）
        → target = [0, 0, 1, 0]（label 2 = Business）

    ``text_fn`` 常用 lambda 传入：``lambda r: r["text"]`` 是一个匿名的小函数，等价于
    ``def f(r): return r["text"]``，适合写这种一行就能说清的取值逻辑。
    ``(desc or {}).get(n, "")``：desc 为 None 时用空字典代替，再查不到描述就返回空字符串。
    """

    def conv(row: Row, i: int, ctx: Ctx) -> Decision:
        opts = [Option(n, (desc or {}).get(n, "")) for n in ctx.label_names]
        return _choice(ctx, i, text_fn(row), ctx.pick(templates, ctx.instr_lang()), opts, row["label"])

    return conv


def conv_tnews(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """CLUE TNEWS（中文新闻标题分类，15 类）→ choice 题。

    state = 新闻标题 ``sentence``；指令固定使用中文模板；
    options = 15 个类别，数字编码通过 ``TNEWS_NAMES`` 换成中文类别名和描述；
    target = ``label`` 的 one-hot。无标签的行（label < 0）跳过。

    举个例子：
        原始行：{"sentence": "上课时学生手机响个不停，老师一怒之下把手机摔了……", "label": 7}
        label_names[7] = "108" → TNEWS_NAMES["108"] = ("教育", "学校、考试、教育")
        → options 为 15 个中文类别，例如 Option("教育", "学校、考试、教育")，target 在“教育”处为 1。
    ``Option(*TNEWS_NAMES[code])`` 中的 * 把 (名字, 描述) 这个二元组拆开，作为两个位置参数传入。
    """
    if row["label"] < 0:
        return None
    opts = [Option(*TNEWS_NAMES[code]) for code in ctx.label_names]
    return _choice(ctx, i, row["sentence"], ctx.pick(T_TOPIC, "zh"), opts, row["label"])


def _mcq(row: Row, i: int, ctx: Ctx) -> Decision | None:
    """常识 / 科学多选题（ARC-Easy、ARC-Challenge、CommonsenseQA）→ choice 题。

    这类题没有单独的材料，所以 state 为空字符串，问题本身放进 instructions；
    options = 各选项的文本（选项名直接用答案文本，而不是 A/B/C 字母，因为后续会打乱顺序，
    字母会失去意义）；target = ``answerKey`` 对应选项的 one-hot。

    跳过两类异常数据：answerKey 不在选项标签里（个别样本标签写法不一致，如 "1" 与 "A"），
    以及选项文本有重复（Decision 要求选项名唯一）。

    举个例子（ARC-Easy 的一行）：
        原始行：{"question": "Which factor will most likely cause a person to develop a fever?",
                 "choices": {"text": ["a leg muscle relaxing after exercise", "a bacterial population in the bloodstream",
                                      "several viral particles on the skin", "carbohydrates being digested in the stomach"],
                             "label": ["A", "B", "C", "D"]},
                 "answerKey": "B"}
        → state = ""（没有材料）
        → instructions = "Answer the question: Which factor will most likely cause a person to develop a fever?"
        → options = 4 个答案文本；target = [0, 1, 0, 0]（labels.index("B") = 1）
    ``len(set(texts)) != len(texts)``：set 会去掉重复元素，长度变短就说明有重复选项。
    """
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if row["answerKey"] not in labels or len(set(texts)) != len(texts):
        return None
    opts = [Option(t, "") for t in texts]
    lang = ctx.instr_lang()
    return _choice(ctx, i, "", ctx.pick(T_MCQ, lang, q=row["question"]), opts, labels.index(row["answerKey"]))


def conv_yelp(row: Row, i: int, ctx: Ctx) -> Decision:
    """Yelp 评论（1~5 星）→ score 题。

    state = 评论正文 ``text``；options = 5 个满意度等级（等级 0 对应 1 星）；
    target = ``label``（0~4）的 one-hot。

    举个例子：一条 5 星好评 label=4 → target = [0, 0, 0, 0, 1]，选项 "4" 的描述是
    "Very positive: enthusiastic, would strongly recommend."。
    """
    lang = ctx.instr_lang()
    return _score(ctx, i, row["text"], ctx.pick(T_REVIEW, lang), REVIEW_LEVELS[lang], one_hot(5, row["label"]))


def conv_stsb(row: Row, i: int, ctx: Ctx) -> Decision:
    """STS-B 语义相似度 → score 题（6 个等级，0~5 分）。

    state = JSON ``{"a": sentence1, "b": sentence2}``；
    target = 把连续分值线性插值到相邻两个等级得到的软标签（见 ``_interp_target``）。

    举个例子：
        原始行：{"sentence1": "A plane is taking off.", "sentence2": "An air plane is taking off.", "score": 1.0}
        → 还原到 0~5 分：1.0 × 5 = 5.0 → target = [0, 0, 0, 0, 0, 1]（完全同义）
        若 score = 0.65 → 3.25 分 → target = [0, 0, 0, 0.75, 0.25, 0]
    """
    lang = ctx.instr_lang()
    state = state_to_text({"a": row["sentence1"], "b": row["sentence2"]})
    # sentence-transformers/stsb 的分数已经归一化到 [0, 1]，原始量表是 0~5，所以先乘 5 还原。
    return _score(ctx, i, state, ctx.pick(T_STS, lang), STS_LEVELS[lang], _interp_target(row["score"] * 5, 6))


def conv_helpsteer(row: Row, i: int, ctx: Ctx) -> Decision:
    """HelpSteer2 → score 题（helpfulness，5 个等级，0~4 分）。

    state = JSON ``{"prompt": 提问, "response": 回复}``；
    target = 人工标注的 ``helpfulness`` 分数的 one-hot。
    这类“评判一段回复好不好”的题，对应 Jev 在 LLM 输出质检、路由场景中的用法。

    举个例子：
        原始行：{"prompt": "c#", "response": "C# is a high-level, object-oriented programming language ...",
                 "helpfulness": 3, ...}
        → state = '{"prompt":"c#","response":"C# is a high-level, ..."}'
        → target = [0, 0, 0, 1, 0]（3 分 = “有帮助，仅有小问题”）
    """
    lang = ctx.instr_lang()
    state = state_to_text({"prompt": row["prompt"], "response": row["response"]})
    return _score(ctx, i, state, ctx.pick(T_HELPFUL, lang), HELPFUL_LEVELS[lang], one_hot(5, row["helpfulness"]))


def conv_toxic(row: Row, i: int, ctx: Ctx) -> Decision:
    """toxic-chat（真实用户发给 LLM 的输入）→ noul 题：这条消息是否有害 / 应被拦截。

    state = 用户输入 ``user_input``；target = ``toxicity``（1 → true，0 → false）。

    举个例子：用户输入是一句正常的编程问题，toxicity=0
        → instructions = "Is this request unsafe or offensive?"（随机模板之一）
        → target = [0.0, 1.0]（false：不是有害内容）
    """
    lang = ctx.instr_lang()
    return _noul(ctx, i, row["user_input"], ctx.pick(T_TOXIC, lang), bool(row["toxicity"]), lang)


# ---- 数据源注册表 ---------------------------------------------------------------------------
# build.py 通过名字从这里取数据源。每一项的参数依次是：
#   Source(内部名, HF 数据集路径, config, 训练 split, 评测 split, 数据语言, 转换器, label_field=标签字段)
# 共 23 个数据源，其中中文 5 个（afqmc、ocnli、cmnli、massive_zh、tnews）。
#
# 语法说明：下面是一个“字典推导式” {键: 值 for 元素 in 列表}。
#   第 1 步：先写出一个 Source 对象列表；
#   第 2 步：对列表里每个 s，生成键值对 s.name → s；
#   结果就是 {"boolq": Source(...), "qqp": Source(...), ...}，可以按名字查数据源。
# 注册项里的 lambda r, c: ... 是传给 _paraphrase 的“判断是否同义”的小函数，r 是原始行，c 是 Ctx。

SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # ---- noul（是/否题）----
        # BoolQ：维基段落 + 是非问题；标签字段是布尔值 answer。
        Source("boolq", "google/boolq", None, "train", "validation", "en", conv_boolq, label_field="answer"),
        # QQP：Quora 问题对是否重复；label 1 = duplicate。
        Source("qqp", "nyu-mll/glue", "qqp", "train", "validation", "en",
               _paraphrase("question1", "question2", lambda r, c: r["label"] == 1)),
        # PAWS：词序打乱造成的“字面相近但意思不同”的句对，专门考察不被表面重叠骗到；label 1 = 同义。
        Source("paws", "google-research-datasets/paws", "labeled_final", "train", "validation", "en",
               _paraphrase("sentence1", "sentence2", lambda r, c: r["label"] == 1)),
        # toxic-chat：内容审核；该数据集没有 validation split，评测用 test（有公开标签）。
        Source("toxic_chat", "lmsys/toxic-chat", "toxicchat0124", "train", "test", "en", conv_toxic,
               label_field="toxicity"),
        # AFQMC（中文）：蚂蚁金服问句相似度。ClassLabel 的类别名是字符串 "0"/"1"，
        # 所以先用 label_names 转成名字再判断是否为 "1"（同义）。
        Source("afqmc", "clue/clue", "afqmc", "train", "validation", "zh",
               _paraphrase("sentence1", "sentence2", lambda r, c: c.label_names[r["label"]] == "1")),
        # ---- noul / choice（NLI，每条随机二选一题型）----
        # MNLI：多领域英文 NLI；评测用 matched 验证集。
        Source("mnli", "nyu-mll/glue", "mnli", "train", "validation_matched", "en", _nli("premise", "hypothesis")),
        # ANLI：对抗构造的高难度 NLI，取最难的第 3 轮（r3）。
        Source("anli", "facebook/anli", "plain_text", "train_r3", "dev_r3", "en", _nli("premise", "hypothesis")),
        # OCNLI（中文原生 NLI）与 CMNLI（MNLI 的中文翻译版）。
        Source("ocnli", "clue/clue", "ocnli", "train", "validation", "zh", _nli("sentence1", "sentence2")),
        Source("cmnli", "clue/clue", "cmnli", "train", "validation", "zh", _nli("sentence1", "sentence2")),
        # ---- choice（意图识别 / 路由，候选意图随机抽样）----
        # Banking77：银行客服 77 类意图；mteb 版本直接提供字符串标签 label_text，评测用 test。
        Source("banking77", "mteb/banking77", None, "train", "test", "en", _intent("text", "label_text"),
               label_field="label_text"),
        # CLINC150（plus）：150 类意图 + oos（超出范围），可训练“都不符合”的判断。
        Source("clinc150", "clinc/clinc_oos", "plus", "train", "validation", "en", _intent("text", "intent"),
               label_field="intent"),
        # MASSIVE：亚马逊多语言意图数据集（60 类），这里取中文（zh-CN）与英文两份。
        Source("massive_zh", "mteb/amazon_massive_intent", "zh-CN", "train", "validation", "zh",
               _intent("text", "label"), label_field="label"),
        Source("massive_en", "mteb/amazon_massive_intent", "en", "train", "validation", "en",
               _intent("text", "label"), label_field="label"),
        # ---- choice（主题 / 情绪分类，类别全部作为选项）----
        # AG News：新闻 4 分类，带类别描述；没有 validation，评测用 test。
        Source("ag_news", "fancyzhx/ag_news", None, "train", "test", "en", _topic(lambda r: r["text"], AG_NEWS_DESC)),
        # DBpedia：14 类实体百科，state = "标题: 正文"。
        Source("dbpedia", "fancyzhx/dbpedia_14", None, "train", "test", "en",
               _topic(lambda r: f"{r['title']}: {r['content'].strip()}")),
        # emotion：推特短文本 6 类情绪，使用情绪专用指令模板。
        Source("emotion", "dair-ai/emotion", "split", "train", "validation", "en",
               _topic(lambda r: r["text"], templates=T_EMOTION)),
        # TNEWS（中文）：今日头条新闻标题 15 类，类别编码映射为中文名。
        Source("tnews", "clue/clue", "tnews", "train", "validation", "zh", conv_tnews),
        # ---- choice（常识 / 推理多选题，state 为空）----
        # answerKey 是字符串（"A"/"B"/...），label_field 只用于推断 label_names，转换器本身不依赖它。
        Source("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "validation", "en", _mcq, label_field="answerKey"),
        Source("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "validation", "en", _mcq,
               label_field="answerKey"),
        Source("commonsense_qa", "tau/commonsense_qa", None, "train", "validation", "en", _mcq,
               label_field="answerKey"),
        # ---- score（打分题）----
        # Yelp：评论 1~5 星 → 5 级满意度；没有 validation，评测用 test。
        Source("yelp", "Yelp/yelp_review_full", None, "train", "test", "en", conv_yelp),
        # STS-B：句对相似度 0~5 → 6 级，连续分值转软标签。
        Source("stsb", "sentence-transformers/stsb", None, "train", "validation", "en", conv_stsb,
               label_field="score"),
        # HelpSteer2：回复有用程度 0~4 → 5 级。
        Source("helpsteer2", "nvidia/HelpSteer2", None, "train", "validation", "en", conv_helpsteer,
               label_field="helpfulness"),
    ]
}
