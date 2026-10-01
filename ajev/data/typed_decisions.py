"""LocalLLaMA/typed-decisions 数据集加载器：4 个业务流程，每个 state 配 5 道 Jev 格式的问题。

这是 AJev 的主测试集（headline benchmark）。在流程中的位置：``ajev.data.build`` 调用本模块，
把 train split 加入训练集（并按 state 留出 10% 作为 val_typed），把 test split 写成
test_typed.jsonl，供 ``ajev.eval.evaluate`` 计算主指标。

数据特点：

* 4 个业务流程（workflow）：agent 执行轨迹观测、客服、发票处理、安全事件，各自有固定的 5 个问题，
  三种题型 noul（是/否题）/ choice（单选题）/ score（打分题）都有；
* 每行原始数据的 state、questions、gold 都是 JSON 字符串，questions 直接就是 Jev 请求格式，
  所以可以复用 ``decisions_from_jev`` 展开成 Decision；
* test split 有 400 个 state × 5 道题 = 2,000 道题，Laya（0.766）、Verdict 2.0（0.771）和
  TypeSafe Jev（0.727）报告的都是这个测试集上的准确率，我们的分数可以直接与之比较；
* gold 标签是多名标注者给出的软分布（soft label），既是训练目标，也天然可以用来评估校准。

原始数据长什么样（customer_service 流程的一行，已简化）::

    state     = '{"account": {"tier": "free", ...}, "thread": [{"role": "customer", "text": "...拿到的商品不对..."}, ...]}'
    questions = '{"category": {"type": "choice", "instructions": "What is this customer conversation primarily about?",
                               "criteria": {"billing": "...", "delivery": "...", "refund": "...", ...}},
                  "needs_human": {"type": "noul", "instructions": "This conversation requires a human agent ...",
                                  "criteria": {"true": "...", "false": "..."}},
                  "urgency": {"type": "score", "instructions": "How time-sensitive is this conversation?",
                              "criteria": ["No time pressure; ...", "Routine; ...", "Elevated; ...", "Critical; ..."]},
                  ...}'
    gold      = '{"category": {"label": "delivery", "probabilities": {"delivery": 0.92, "refund": 0.05, ...}}, ...}'

“软分布”的意思：例如 category 题有若干名标注者，92% 认为是 delivery、5% 认为是 refund……
这比简单的“答案是 delivery”包含更多信息：它告诉模型这道题有多确定。
"""

from __future__ import annotations

import json
from typing import Iterator

from ajev.schema import Decision, decisions_from_jev

# HF Hub 上的数据集路径。
DATASET = "LocalLLaMA/typed-decisions"
# 数据集包含的 4 个业务流程，同时也是可单独加载的 config 名。
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def iter_typed_decisions(split: str, config: str = "all") -> Iterator[Decision]:
    """逐条产出 typed-decisions 某个 split 展开后的 Decision。

    参数：
        split: ``"train"``（1,200 个 state）或 ``"test"``（400 个 state）。
        config: ``"all"`` 表示全部 4 个流程，也可以传 ``WORKFLOWS`` 中的某一个只加载单个流程。

    每个 state 的 5 道题展开为 5 个 Decision：
        * id = ``"<行 id>/<问题 id>"``，group = 行 id（同一 state 的题共享 group，
          build 阶段按 group 整组留出验证集）；
        * source = ``"typed_decisions/<流程名>"``，评测报告可以按流程拆分；
        * target = gold 中该问题的概率分布（按选项顺序对齐）。

    举个例子：上面那行 customer_service 数据（id="customer_service_000000"）会产出 5 个 Decision，其中之一是
        Decision(id="customer_service_000000/category", group="customer_service_000000",
                 source="typed_decisions/customer_service", type="choice",
                 state='{"account": ..., "thread": [...]}',
                 instructions="What is this customer conversation primarily about?",
                 options=[account, billing, delivery, refund, technical],
                 target=[0.0067, 0.01, 0.9233, 0.0467, 0.0133],
                 meta={"question_id": "category", "gold_label": "delivery", "annotators_agree": True})

    处理步骤（对每一行原始数据）：
        第 1 步：把 gold、label_agreement 两个 JSON 字符串解析成字典；
        第 2 步：把 state 和 questions 也解析出来，交给 decisions_from_jev 展开成多个 Decision，
                 target 从 gold 的 probabilities 中按选项顺序取出；
        第 3 步：为每个 Decision 补充 meta 信息并校验，然后用 yield 逐个交出。

    另外在 meta 里补充两项：
        * ``gold_label``：数据集给出的标准答案。评测时以它为准计算准确率，
          而不是对 target 取 argmax——软分布可能出现并列最高，这样能与其他项目的口径保持一致；
        * ``annotators_agree``：标注者之间的 argmax 是否一致，可用于分析“难题 / 有争议的题”。
    """
    # 延迟导入 datasets，只用到 schema 的场景不需要安装它。
    from datasets import load_dataset

    for row in load_dataset(DATASET, config, split=split):
        # 第 1 步：json.loads 把 JSON 格式的字符串转换成 Python 字典。
        gold = json.loads(row["gold"])
        agreement = json.loads(row["label_agreement"])
        # 第 2 步：展开为每个问题一个 Decision。
        for d in decisions_from_jev(
            json.loads(row["state"]),
            json.loads(row["questions"]),
            group=row["id"],
            source=f"typed_decisions/{row['workflow']}",
            gold=gold,
        ):
            # 第 3 步：补充 meta。agreement.get(qid, {}) 在找不到该问题时返回空字典，避免 KeyError。
            qid = d.meta["question_id"]
            d.meta["gold_label"] = gold[qid]["label"]
            d.meta["annotators_agree"] = agreement.get(qid, {}).get("argmax_agree")
            d.validate()  # 校验选项与 target 是否对齐、分布是否合法，出错立即暴露
            yield d
