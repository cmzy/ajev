"""ajev/data/hard_gen.py 的测试：日期计算、工作时长计算、生成结果合法且可复现。"""

import datetime as dt

from ajev.data.hard_gen import add_months, business_hours, generate


def test_add_months_clamps_to_month_end():
    assert add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert add_months(dt.date(2023, 1, 31), 1) == dt.date(2023, 2, 28)
    assert add_months(dt.date(2023, 9, 5), 36) == dt.date(2026, 9, 5)
    assert add_months(dt.date(2023, 11, 15), 2) == dt.date(2024, 1, 15)


def test_business_hours():
    mon9 = dt.datetime(2026, 3, 2, 9)  # 2026-03-02 是周一
    assert business_hours(mon9, dt.datetime(2026, 3, 2, 12), set()) == 3
    # 周五 17:00 → 下周一 10:00：周五 1 小时 + 周一 1 小时
    assert business_hours(dt.datetime(2026, 3, 6, 17), dt.datetime(2026, 3, 9, 10), set()) == 2
    # 周一是节假日：只算周二的 2 小时
    assert business_hours(dt.datetime(2026, 3, 1, 20), dt.datetime(2026, 3, 3, 11), {dt.date(2026, 3, 2)}) == 2
    # 晚上创建、第二天早上 8 点前回复：0 小时
    assert business_hours(dt.datetime(2026, 3, 2, 19), dt.datetime(2026, 3, 3, 8), set()) == 0


def test_generate_valid_and_deterministic():
    a = generate(100, seed=3)
    b = generate(100, seed=3)
    assert [d.id for d in a] == [d.id for d in b] and [d.target for d in a] == [d.target for d in b]
    assert len(a) == 100
    for d in a:
        d.validate()
        assert abs(sum(d.target) - 1) < 1e-9 and max(d.target) == 1.0
    assert len({d.id for d in a}) == len(a)
