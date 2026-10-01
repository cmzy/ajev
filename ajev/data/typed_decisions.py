"""LocalLLaMA/typed-decisions: 4 business workflows, 5 Jev-format questions per state.

The test split (400 states x 5 questions = 2,000 decisions) is the main benchmark used by
Laya (0.766), Verdict 2.0 (0.771) and TypeSafe Jev (0.727). Gold labels are soft
distributions from several annotators, so they double as calibration targets.
"""

from __future__ import annotations

import json
from typing import Iterator

from ajev.schema import Decision, decisions_from_jev

DATASET = "LocalLLaMA/typed-decisions"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def iter_typed_decisions(split: str, config: str = "all") -> Iterator[Decision]:
    from datasets import load_dataset

    for row in load_dataset(DATASET, config, split=split):
        gold = json.loads(row["gold"])
        agreement = json.loads(row["label_agreement"])
        for d in decisions_from_jev(
            json.loads(row["state"]),
            json.loads(row["questions"]),
            group=row["id"],
            source=f"typed_decisions/{row['workflow']}",
            gold=gold,
        ):
            qid = d.meta["question_id"]
            d.meta["gold_label"] = gold[qid]["label"]
            d.meta["annotators_agree"] = agreement.get(qid, {}).get("argmax_agree")
            d.validate()
            yield d
