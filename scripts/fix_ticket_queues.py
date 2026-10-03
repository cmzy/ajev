#!/usr/bin/env python3
"""把已经构建好的数据文件里的客服工单“队列”题改成新格式：10 个团队队列 + 说明（原来是 52 个选项，
其中 42 个是永远不会正确的话题标签，见 ajev/data/more_sources.py 的 TICKET_QUEUES）。

    python scripts/fix_ticket_queues.py data/lm2/train.jsonl data/lm3/train.jsonl

不用重新下载、重新构建整套数据。标准答案不变，只换选项列表；答案不在 10 个队列里的题删除。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ajev.data.more_sources import TICKET_QUEUES  # noqa: E402
from ajev.schema import Option, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    out, fixed, dropped = [], 0, 0
    for d in read_jsonl(src):
        if d.source == "support_tickets" and d.id.endswith("/queue"):
            gold = d.options[d.gold_index].name
            if gold not in TICKET_QUEUES:
                dropped += 1
                continue
            d.options = [Option(q, desc) for q, desc in TICKET_QUEUES.items()]
            d.target = [1.0 if q == gold else 0.0 for q in TICKET_QUEUES]
            d.validate()
            fixed += 1
        out.append(d)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    write_jsonl(dst, out)
    print(f"{fixed} queue decisions fixed, {dropped} dropped, {len(out)} decisions -> {dst}")


if __name__ == "__main__":
    main()
