"""ajev/lm/train_utils.py 的测试：不依赖 torch，uv 环境也能跑。"""

import math

from ajev.lm.train_utils import SpikeGuard, fixed_count_steps, flatten_steps, ordinal_smooth


def test_fixed_count_steps_exact_count_and_budget():
    lengths = [50 + (i * 37) % 900 for i in range(100)]
    steps = fixed_count_steps(lengths, per_step=32, max_tokens=4000, max_batch=16, seed=1)
    flat = [i for st in steps for mb in st for i in mb]
    assert sorted(flat) == list(range(100))  # 每道题恰好出现一次
    assert [sum(len(mb) for mb in st) for st in steps] == [32, 32, 32, 4]  # 每步正好 32 道（最后一步是余数）
    for st in steps:
        for mb in st:
            assert len(mb) <= 16 and len(mb) * max(lengths[i] for i in mb) <= 4000 or len(mb) == 1
    assert steps == fixed_count_steps(lengths, 32, 4000, 16, seed=1)  # 同一种子结果确定（断点续训依赖这一点）


def test_flatten_steps_marks_step_ends():
    micro, ends = flatten_steps([[[0], [1]], [[3, 2]]])
    assert micro == [[0], [1], [3, 2]] and ends == [False, True, True]


def test_spike_guard_relative_threshold():
    g = SpikeGuard(ratio=10, abs_limit=1000, window=50, min_history=5)
    for x in [3.0, 4.0, 5.0, 4.0, 3.0]:
        assert not g.should_skip(x)
    assert g.median() == 4.0
    assert g.should_skip(41.0)        # > 10 × 中位数 4
    assert not g.should_skip(39.0)    # 没超过：照常更新，并计入历史
    assert g.should_skip(float("nan")) and g.should_skip(math.inf)
    g2 = SpikeGuard(ratio=10, abs_limit=1000, min_history=20)
    assert not g2.should_skip(500.0)  # 历史不够时只看绝对上限
    assert g2.should_skip(1500.0)
    g3 = SpikeGuard(min_history=5)
    g3.load_state_dict(g.state_dict())
    assert g3.median() == g.median()  # 续训后状态一致


def test_ordinal_smooth():
    assert ordinal_smooth([0, 0, 1, 0, 0], 0.2) == [0, 0.1, 0.8, 0.1, 0]
    assert ordinal_smooth([1, 0, 0, 0, 0], 0.2) == [0.8, 0.2, 0, 0, 0]
    assert ordinal_smooth([0, 0.5, 0.5], 0.2) == [0, 0.5, 0.5]  # 已经是软标签，不处理
    out = ordinal_smooth([0, 0, 0, 0, 0, 0, 1], 0.3)
    assert math.isclose(sum(out), 1.0) and math.isclose(out[5], 0.3)
