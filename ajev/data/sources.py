"""Converters from public Hugging Face datasets to ``Decision`` records.

Each ``Source`` knows where its data lives, which split is used for training and which
for evaluation, and how to turn one row into a Decision. Instructions are drawn from
several paraphrased templates so the model does not latch onto one wording; English
sources sometimes get Chinese instructions (``zh_instr_prob``) so cross-lingual
questions are covered too.
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

Row = dict[str, Any]


@dataclass
class Ctx:
    """Per-call context handed to converters."""

    rng: random.Random
    source: "Source"
    label_names: list[str]
    zh_instr_prob: float = 0.0

    def instr_lang(self) -> str:
        if self.source.lang == "zh":
            return "zh"
        return "zh" if self.rng.random() < self.zh_instr_prob else "en"

    def pick(self, templates: dict[str, list[str]], lang: str, **kw: str) -> str:
        return self.rng.choice(templates[lang]).format(**kw)


Converter = Callable[[Row, int, Ctx], "Decision | None"]


@dataclass
class Source:
    name: str
    path: str
    config: str | None
    train_split: str
    eval_split: str
    lang: str
    convert: Converter
    label_field: str = "label"
    # Some sources are only used for evaluation (e.g. typed-decisions test lives elsewhere).
    tags: list[str] = field(default_factory=list)

    def load(self, split: str):
        from datasets import load_dataset

        return load_dataset(self.path, self.config, split=split)

    def iter_decisions(
        self, split: str, limit: int | None, seed: int, zh_instr_prob: float = 0.0
    ) -> Iterator[Decision]:
        ds = self.load(split)
        feat = ds.features.get(self.label_field)
        label_names = list(getattr(feat, "names", []) or [])
        if not label_names and getattr(feat, "dtype", None) == "string":
            # String labels (e.g. intents): the label set is whatever occurs in the split.
            label_names = sorted(set(ds[self.label_field]))
        ds = ds.shuffle(seed=seed)
        rng = random.Random(f"{self.name}/{split}/{seed}")
        ctx = Ctx(rng=rng, source=self, label_names=label_names, zh_instr_prob=zh_instr_prob)
        n = 0
        for i, row in enumerate(ds):
            if limit is not None and n >= limit:
                break
            d = self.convert(row, i, ctx)
            if d is None:
                continue
            d.validate()
            n += 1
            yield d


# ---- shared builders ---------------------------------------------------------


def _noul(ctx: Ctx, i: int, state: str, instructions: str, answer: bool, lang: str, **meta: Any) -> Decision:
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
    """Spread a continuous value in [0, k-1] over its two neighbouring levels."""
    value = min(max(value, 0.0), k - 1)
    lo = int(value)
    hi = min(lo + 1, k - 1)
    t = [0.0] * k
    t[lo] += 1.0 - (value - lo)
    t[hi] += value - lo
    return t


def _sample_candidates(ctx: Ctx, gold: str, lo: int = 4, hi: int = 20) -> tuple[list[str], int]:
    """Pick a random candidate set from the full label list, always containing gold."""
    pool = [n for n in ctx.label_names if n != gold]
    k = min(ctx.rng.randint(lo, hi), len(pool) + 1)
    cands = ctx.rng.sample(pool, k - 1) + [gold]
    ctx.rng.shuffle(cands)
    return cands, cands.index(gold)


def _humanize(label: str) -> str:
    return label.replace("_", " ").strip()


# ---- instruction templates ---------------------------------------------------------

T_BOOLQ = {
    "en": ["{q}?", "Based on the passage: {q}?", "According to the text, {q}?"],
    "zh": ["根据这段文字回答：{q}？", "依据给定材料判断：{q}？"],
}
T_ENTAIL = {
    "en": [
        "The text implies that: {h}",
        "Does the passage support the claim \"{h}\"?",
        "Given the text, it is true that {h}",
    ],
    "zh": ["根据上文可以推出：{h}", "上文是否支持这一说法：“{h}”？", "由给定内容可知：{h}"],
}
T_NLI3 = {
    "en": ["What is the relationship between the text and the claim \"{h}\"?",
           "How does the passage relate to this statement: {h}"],
    "zh": ["上文与陈述“{h}”是什么关系？", "给定内容和这句话的逻辑关系是什么：{h}"],
}
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
T_PARAPHRASE = {
    "en": ["The two texts in `a` and `b` mean the same thing.",
           "Do `a` and `b` ask or state the same thing?",
           "`a` is a paraphrase of `b`."],
    "zh": ["`a` 和 `b` 表达的是同一个意思。", "`a` 与 `b` 问的是同一个问题吗？", "`a` 是 `b` 的同义改写。"],
}
T_INTENT = {
    "en": ["What does the user want?", "Which intent best matches this user message?",
           "Route this request to the right intent."],
    "zh": ["用户想做什么？", "这条用户消息最符合哪个意图？", "把这条请求路由到正确的意图。"],
}
T_TOPIC = {
    "en": ["What is this text mainly about?", "Which category does this article belong to?",
           "Classify the topic of this text."],
    "zh": ["这段文字主要讲的是什么？", "这篇文章属于哪个类别？", "给这段文本的主题分类。"],
}
T_EMOTION = {
    "en": ["Which emotion does the writer express?", "What is the dominant feeling in this text?"],
    "zh": ["作者表达了哪种情绪？", "这段文字的主要情感是什么？"],
}
T_MCQ = {
    "en": ["{q}", "Answer the question: {q}", "Pick the best answer. {q}"],
    "zh": ["{q}", "回答问题：{q}", "选出最合适的答案。{q}"],
}
T_REVIEW = {
    "en": ["How satisfied is the reviewer?", "Rate the sentiment of this review.",
           "How positive is this customer review?"],
    "zh": ["评论者的满意度如何？", "给这条评论的情感打分。", "这条顾客评价有多正面？"],
}
REVIEW_LEVELS = {
    "en": ["Very negative: angry or strongly disappointed, would not return.",
           "Negative: more complaints than praise.",
           "Mixed or neutral: balanced good and bad points.",
           "Positive: satisfied with minor reservations.",
           "Very positive: enthusiastic, would strongly recommend."],
    "zh": ["非常负面：愤怒或极度失望，不会再来。", "负面：抱怨多于称赞。", "中性或褒贬参半。",
           "正面：基本满意，有小保留。", "非常正面：热情推荐。"],
}
T_STS = {
    "en": ["How similar in meaning are `a` and `b`?", "Rate the semantic similarity of `a` and `b`."],
    "zh": ["`a` 和 `b` 的意思有多相近？", "给 `a` 与 `b` 的语义相似度打分。"],
}
STS_LEVELS = {
    "en": ["Completely unrelated.", "Not equivalent, but on the same topic.",
           "Not equivalent, but share some details.", "Roughly equivalent; some important information differs.",
           "Mostly equivalent; only minor details differ.", "Completely equivalent in meaning."],
    "zh": ["完全无关。", "不等价，但主题相同。", "不等价，但有部分细节相同。", "大致等价，但有重要信息不同。",
           "基本等价，仅细节不同。", "意思完全相同。"],
}
T_HELPFUL = {
    "en": ["How helpful is the `response` to the `prompt`?", "Rate the overall helpfulness of the response."],
    "zh": ["`response` 对 `prompt` 有多大帮助？", "给这条回复的整体有用程度打分。"],
}
HELPFUL_LEVELS = {
    "en": ["Not helpful at all: wrong, off-topic or refuses without reason.",
           "Slightly helpful: mostly misses what the user needed.",
           "Partially helpful: addresses the request with notable gaps or errors.",
           "Helpful: addresses the request well with minor issues.",
           "Extremely helpful: complete, correct and well suited to the request."],
    "zh": ["完全没帮助：错误、离题或无理拒答。", "略有帮助：大体没满足需求。", "部分有帮助：回应了请求但有明显缺漏或错误。",
           "有帮助：较好地完成请求，仅有小问题。", "非常有帮助：完整、正确且贴合需求。"],
}
T_TOXIC = {
    "en": ["This user message is toxic, harmful or inappropriate.",
           "The message should be blocked by a content moderation filter.",
           "Is this request unsafe or offensive?"],
    "zh": ["这条用户消息有毒、有害或不当。", "内容审核应当拦截这条消息。", "这个请求是否不安全或冒犯？"],
}

TNEWS_NAMES = {
    "100": ("故事", "民间故事、生活故事"), "101": ("文化", "文化、艺术、历史"), "102": ("娱乐", "明星、影视、综艺"),
    "103": ("体育", "体育赛事与运动员"), "104": ("财经", "经济、金融、商业"), "106": ("房产", "房地产、楼市"),
    "107": ("汽车", "汽车、驾驶"), "108": ("教育", "学校、考试、教育"), "109": ("科技", "科技、互联网、数码"),
    "110": ("军事", "军事、国防"), "112": ("旅游", "旅行、景点"), "113": ("国际", "国际新闻与外交"),
    "114": ("股票", "股市、证券"), "115": ("农业", "农业、农村、农民"), "116": ("电竞", "电子竞技与游戏"),
}
AG_NEWS_DESC = {"World": "International news and politics.", "Sports": "Sport events and athletes.",
                "Business": "Companies, markets and the economy.", "Sci/Tech": "Science and technology."}


# ---- converters ----------------------------------------------------------------------


def conv_boolq(row: Row, i: int, ctx: Ctx) -> Decision:
    lang = ctx.instr_lang()
    return _noul(ctx, i, row["passage"], ctx.pick(T_BOOLQ, lang, q=row["question"]), bool(row["answer"]), lang)


def _nli(premise_key: str, hyp_key: str) -> Converter:
    def conv(row: Row, i: int, ctx: Ctx) -> Decision | None:
        if row["label"] < 0:
            return None
        name = ctx.label_names[row["label"]]
        lang = ctx.instr_lang()
        premise, hyp = row[premise_key], row[hyp_key]
        if ctx.rng.random() < 0.5:
            return _noul(ctx, i, premise, ctx.pick(T_ENTAIL, lang, h=hyp), name == "entailment", lang)
        names = ["entailment", "neutral", "contradiction"]
        opts = [Option(n, NLI3_OPTS[lang][n]) for n in names]
        return _choice(ctx, i, premise, ctx.pick(T_NLI3, lang, h=hyp), opts, names.index(name))

    return conv


def _paraphrase(a_key: str, b_key: str, positive: Callable[[Row, Ctx], bool]) -> Converter:
    def conv(row: Row, i: int, ctx: Ctx) -> Decision | None:
        if row["label"] < 0:
            return None
        lang = ctx.instr_lang()
        state = state_to_text({"a": row[a_key], "b": row[b_key]})
        return _noul(ctx, i, state, ctx.pick(T_PARAPHRASE, lang), positive(row, ctx), lang)

    return conv


def _intent(text_key: str, label_key: str) -> Converter:
    def conv(row: Row, i: int, ctx: Ctx) -> Decision:
        gold = row[label_key] if isinstance(row[label_key], str) else ctx.label_names[row[label_key]]
        cands, gi = _sample_candidates(ctx, gold)
        opts = [
            Option(c, "The request does not match any supported intent." if c == "oos" else _humanize(c))
            for c in cands
        ]
        return _choice(ctx, i, row[text_key], ctx.pick(T_INTENT, ctx.instr_lang()), opts, gi)

    return conv


def _topic(text_fn: Callable[[Row], str], desc: dict[str, str] | None = None, templates=T_TOPIC) -> Converter:
    def conv(row: Row, i: int, ctx: Ctx) -> Decision:
        opts = [Option(n, (desc or {}).get(n, "")) for n in ctx.label_names]
        return _choice(ctx, i, text_fn(row), ctx.pick(templates, ctx.instr_lang()), opts, row["label"])

    return conv


def conv_tnews(row: Row, i: int, ctx: Ctx) -> Decision | None:
    if row["label"] < 0:
        return None
    opts = [Option(*TNEWS_NAMES[code]) for code in ctx.label_names]
    return _choice(ctx, i, row["sentence"], ctx.pick(T_TOPIC, "zh"), opts, row["label"])


def _mcq(row: Row, i: int, ctx: Ctx) -> Decision | None:
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if row["answerKey"] not in labels or len(set(texts)) != len(texts):
        return None
    opts = [Option(t, "") for t in texts]
    lang = ctx.instr_lang()
    return _choice(ctx, i, "", ctx.pick(T_MCQ, lang, q=row["question"]), opts, labels.index(row["answerKey"]))


def conv_yelp(row: Row, i: int, ctx: Ctx) -> Decision:
    lang = ctx.instr_lang()
    return _score(ctx, i, row["text"], ctx.pick(T_REVIEW, lang), REVIEW_LEVELS[lang], one_hot(5, row["label"]))


def conv_stsb(row: Row, i: int, ctx: Ctx) -> Decision:
    lang = ctx.instr_lang()
    state = state_to_text({"a": row["sentence1"], "b": row["sentence2"]})
    # sentence-transformers/stsb scores are normalised to [0, 1]; the original scale is 0-5.
    return _score(ctx, i, state, ctx.pick(T_STS, lang), STS_LEVELS[lang], _interp_target(row["score"] * 5, 6))


def conv_helpsteer(row: Row, i: int, ctx: Ctx) -> Decision:
    lang = ctx.instr_lang()
    state = state_to_text({"prompt": row["prompt"], "response": row["response"]})
    return _score(ctx, i, state, ctx.pick(T_HELPFUL, lang), HELPFUL_LEVELS[lang], one_hot(5, row["helpfulness"]))


def conv_toxic(row: Row, i: int, ctx: Ctx) -> Decision:
    lang = ctx.instr_lang()
    return _noul(ctx, i, row["user_input"], ctx.pick(T_TOXIC, lang), bool(row["toxicity"]), lang)


# ---- registry ---------------------------------------------------------------------------

SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # noul
        Source("boolq", "google/boolq", None, "train", "validation", "en", conv_boolq, label_field="answer"),
        Source("qqp", "nyu-mll/glue", "qqp", "train", "validation", "en",
               _paraphrase("question1", "question2", lambda r, c: r["label"] == 1)),
        Source("paws", "google-research-datasets/paws", "labeled_final", "train", "validation", "en",
               _paraphrase("sentence1", "sentence2", lambda r, c: r["label"] == 1)),
        Source("toxic_chat", "lmsys/toxic-chat", "toxicchat0124", "train", "test", "en", conv_toxic,
               label_field="toxicity"),
        Source("afqmc", "clue/clue", "afqmc", "train", "validation", "zh",
               _paraphrase("sentence1", "sentence2", lambda r, c: c.label_names[r["label"]] == "1")),
        # noul / choice (NLI)
        Source("mnli", "nyu-mll/glue", "mnli", "train", "validation_matched", "en", _nli("premise", "hypothesis")),
        Source("anli", "facebook/anli", "plain_text", "train_r3", "dev_r3", "en", _nli("premise", "hypothesis")),
        Source("ocnli", "clue/clue", "ocnli", "train", "validation", "zh", _nli("sentence1", "sentence2")),
        Source("cmnli", "clue/clue", "cmnli", "train", "validation", "zh", _nli("sentence1", "sentence2")),
        # choice (intent / routing)
        Source("banking77", "mteb/banking77", None, "train", "test", "en", _intent("text", "label_text"),
               label_field="label_text"),
        Source("clinc150", "clinc/clinc_oos", "plus", "train", "validation", "en", _intent("text", "intent"),
               label_field="intent"),
        Source("massive_zh", "mteb/amazon_massive_intent", "zh-CN", "train", "validation", "zh",
               _intent("text", "label"), label_field="label"),
        Source("massive_en", "mteb/amazon_massive_intent", "en", "train", "validation", "en",
               _intent("text", "label"), label_field="label"),
        # choice (topic / emotion)
        Source("ag_news", "fancyzhx/ag_news", None, "train", "test", "en", _topic(lambda r: r["text"], AG_NEWS_DESC)),
        Source("dbpedia", "fancyzhx/dbpedia_14", None, "train", "test", "en",
               _topic(lambda r: f"{r['title']}: {r['content'].strip()}")),
        Source("emotion", "dair-ai/emotion", "split", "train", "validation", "en",
               _topic(lambda r: r["text"], templates=T_EMOTION)),
        Source("tnews", "clue/clue", "tnews", "train", "validation", "zh", conv_tnews),
        # choice (reasoning)
        Source("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "validation", "en", _mcq, label_field="answerKey"),
        Source("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "validation", "en", _mcq,
               label_field="answerKey"),
        Source("commonsense_qa", "tau/commonsense_qa", None, "train", "validation", "en", _mcq,
               label_field="answerKey"),
        # score
        Source("yelp", "Yelp/yelp_review_full", None, "train", "test", "en", conv_yelp),
        Source("stsb", "sentence-transformers/stsb", None, "train", "validation", "en", conv_stsb,
               label_field="score"),
        Source("helpsteer2", "nvidia/HelpSteer2", None, "train", "validation", "en", conv_helpsteer,
               label_field="helpfulness"),
    ]
}

