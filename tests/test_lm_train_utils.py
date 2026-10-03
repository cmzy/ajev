"""ajev/lm/train_utils.py 的测试：不依赖 torch，uv 环境也能跑。"""

import math

from ajev.lm.train_utils import SpikeGuard, fixed_count_steps, flatten_steps, ordinal_smooth


def test_fixed_count_steps_exact_count_and_budget():
    lengths = [50 + (i * 37) % 900 for i in range(100)]
    steps = fixed_count_steps(lengths, per_step=32, max_tokens=4000, max_batch=16, seed=1, sort_block=0)
    flat = [i for st in steps for mb in st for i in mb]
    assert sorted(flat) == list(range(100))  # 每道题恰好出现一次
    assert [sum(len(mb) for mb in st) for st in steps] == [32, 32, 32, 4]  # 每步正好 32 道（最后一步是余数）
    for st in steps:
        for mb in st:
            assert len(mb) <= 16 and len(mb) * max(lengths[i] for i in mb) <= 4000 or len(mb) == 1
    assert steps == fixed_count_steps(lengths, 32, 4000, 16, seed=1, sort_block=0)  # 同一种子结果确定（断点续训依赖这一点）


def test_fixed_count_steps_sorted_blocks():
    lengths = [50 + (i * 37) % 900 for i in range(1000)]
    steps = fixed_count_steps(lengths, per_step=32, max_tokens=8000, max_batch=32, seed=2, sort_block=256)
    flat = sorted(i for st in steps for mb in st for i in mb)
    assert flat == list(range(1000))
    counts = [sum(len(mb) for mb in st) for st in steps]
    assert counts.count(32) == len(counts) - 1  # 每块 256 道 = 8 步 × 32；最后一块 232 道 → 7 步 + 一步 8 道
    assert sorted(counts)[0] == 8


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


def test_step_mean_lengths_and_edges():
    from ajev.lm.train_utils import quantile_edges, step_mean_lengths
    micro, ends = [[0, 1], [2], [3]], [False, True, True]
    assert step_mean_lengths(micro, ends, [100, 300, 200, 50]) == [200.0, 50.0]
    assert step_mean_lengths([[0], [1], [2], [3]], None, [100, 300, 200, 50], grad_accum=2) == [200.0, 125.0]
    assert quantile_edges([float(x) for x in range(100)], 4) == [25.0, 50.0, 75.0]


def test_bucketed_guard_compares_long_steps_with_long_steps():
    from ajev.lm.train_utils import BucketedSpikeGuard
    g = BucketedSpikeGuard(edges=[500, 1500], ratio=30, abs_limit=2000, min_history=5)
    for _ in range(5):
        assert not g.should_skip(3.0, mean_len=200)     # 短题步：中位数 3
        assert not g.should_skip(150.0, mean_len=2500)  # 长题步：中位数 150
    assert not g.should_skip(400.0, mean_len=2600)      # 长题步 400 只有同档中位数的 2.7 倍：不跳过
    assert g.should_skip(400.0, mean_len=250)           # 短题步 400 是同档中位数的 133 倍：跳过
    assert g.should_skip(2500.0, mean_len=2600)         # 超过绝对上限：跳过
    assert g.bucket(100) == 0 and g.bucket(800) == 1 and g.bucket(5000) == 2
    g2 = BucketedSpikeGuard(edges=[500, 1500], min_history=5)
    g2.load_state_dict(g.state_dict())
    assert g2.median(2500) == g.median(2500)
