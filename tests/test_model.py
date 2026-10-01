"""模型侧单元测试：输入编码、batch 补齐、unpermute、各损失函数和温度拟合。

什么是单元测试：用一些已知答案的小例子去调用函数，检查输出是否符合预期。改代码后跑一遍测试，
就能马上发现有没有把原来对的东西改坏。这里用的是 pytest 框架：所有以 ``test_`` 开头的函数都会被
自动执行，函数里的 ``assert 条件`` 不成立时，该测试就算失败。

运行方法（在仓库根目录）：

    PYTHONPATH=$PWD python -m pytest -q tests/test_model.py

需要 torch 和 transformers；在没有这两个包的环境（例如本地只装了数据与评测依赖的 uv 环境）中
会被整体跳过。本地可用 Python 3.11 + torch 2.2 的 CPU 环境运行，Colab 上自带依赖。
"""

import random

import pytest

# pytest.importorskip：能导入就返回该模块，导入不了就把整个测试文件标记为“跳过”，而不是报错。
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ajev.calibrate import fit_temperature  # noqa: E402
from ajev.model.batching import collate  # noqa: E402
from ajev.model.encoding import DecisionEncoder, Encoded  # noqa: E402
from ajev.schema import Decision, Option  # noqa: E402
from ajev.train.losses import rps, soft_ce, symmetric_kl, unpermute  # noqa: E402


class FakeTok:
    """字符级的假 tokenizer：每个字符对应一个 token，并提供 DecisionEncoder 需要的特殊 token id。

    用它代替真实 tokenizer，测试不用联网下载，而且 token 数 = 字符数，长度预算一目了然。
    特殊 token：pad=0，cls(<bos>)=1，sep(<eos>)=2，mask(标记位)=3；普通字符从 10 开始编号。
    """

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    bos_token_id = eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 1000) for c in text]}


def decision(n_opts=3, state="s" * 50, desc="d" * 20):
    """构造一道测试用的 choice 题：n_opts 个选项，标准答案是第一个。"""
    return Decision(id="x", source="t", type="choice", state=state, instructions="q?",
                    options=[Option(f"o{i}", desc) for i in range(n_opts)], target=[1.0] + [0.0] * (n_opts - 1))


def test_encoder_markers_and_truncation():
    """序列不超过 max_len、state 超长时被截断、每个选项的标记位都是 <mask>(=3)、首尾是 <bos>/<eos>。

    这道题：头部 "choice: q?" 约 12 个 token，3 个选项各 1+2+22 个 token 但预算只有 40，
    描述会被压缩；材料有 50 个字符，而总长度上限只有 64，所以 state 一定被截断。
    """
    # max_instr=16：max_len 只有 64，问题预算必须相应调小（否则构造时就会被拒绝，见下一个测试）。
    enc = DecisionEncoder(FakeTok(), max_len=64, max_instr=16, max_options=40, min_desc=4)
    e = enc.encode(decision())
    assert len(e.input_ids) <= 64 and e.truncated_state
    assert [e.input_ids[p] for p in e.marker_pos] == [3, 3, 3]
    assert e.input_ids[0] == 1 and e.input_ids[-1] == 2


def test_encoder_shrinks_descriptions_then_fails():
    """选项区超预算时先压缩描述（仍保留全部 5 个标记位）；连选项名都放不下整条序列时才报错。

    - max_options=60：描述被压缩，5 个标记位都在；
    - max_options=10：5 个选项光选项名就要 5 ×（1 个标记位 + 2）= 15 > 10。以前这里会直接报错，
      现在预算自动放宽到 15（序列里还有足够空间），编码成功、描述全部去掉；
    - 100 个选项、选项名各 4 个字符：光选项名就要 100 × 5 = 500 个 token，超过 max_len=200 能给的空间，
      这道题在 200 个 token 内根本放不下，``pytest.raises(ValueError)`` 期望这里报错。
    """
    enc = DecisionEncoder(FakeTok(), max_len=512, max_options=60, min_desc=4)
    e = enc.encode(decision(n_opts=5, state=""))
    assert len(e.marker_pos) == 5
    e = DecisionEncoder(FakeTok(), max_options=10).encode(decision(n_opts=5))
    assert len(e.marker_pos) == 5
    many = Decision(id="m", source="t", type="choice", state="", instructions="q?",
                    options=[Option(f"o{i:03d}") for i in range(100)], target=[1.0] + [0.0] * 99)
    with pytest.raises(ValueError):
        DecisionEncoder(FakeTok(), max_len=200, max_instr=16).encode(many)


def test_encoder_never_exceeds_max_len():
    """问题很长、选项很多时，序列长度也绝不超过 max_len（以前头部和选项区加起来可能超出）。

    max_len=200，问题 300 个字符（截到 max_instr=128），8 个选项各带 60 字符描述：
    头部约 1 + 8 + 128 + 1 = 138，选项区只剩约 60 个 token，描述被压缩；材料一个字都放不下。
    """
    long_q = Decision(id="l", source="t", type="choice", state="s" * 100, instructions="q" * 300,
                      options=[Option(f"o{i}", "d" * 60) for i in range(8)], target=[1.0] + [0.0] * 7)
    e = DecisionEncoder(FakeTok(), max_len=200, max_options=384).encode(long_q)
    assert len(e.input_ids) <= 200 and len(e.marker_pos) == 8 and e.truncated_state


def test_encoder_rejects_too_small_max_len():
    """max_len 连头部（最长 1 + 8 + max_instr + 1）都放不下时，构造时就报错，而不是生成超长序列。"""
    with pytest.raises(ValueError):
        DecisionEncoder(FakeTok(), max_len=100, max_instr=128)


def test_collate_and_unpermute():
    """collate 正确补齐 input_ids 和 option_mask；unpermute 按 out[perm[j]] = logits[j] 放回原始顺序，
    且补齐位（perm 映射到自身）仍然是 -inf。

    第 2 行的验证过程：logits=[1, 2, 3]，perm=[2, 0, 1]
        out[2]=1，out[0]=2，out[1]=3 → out=[2, 3, 1]
    """
    b = collate([Encoded([1, 3, 5, 3, 6, 2], [1, 3], False), Encoded([1, 3, 3, 3, 2], [1, 2, 3], False)], 0)
    assert b["input_ids"].shape == (2, 6) and b["option_mask"].tolist() == [[True, True, False], [True, True, True]]
    logits = torch.tensor([[1.0, 2.0, float("-inf")], [1.0, 2.0, 3.0]])
    perm = torch.tensor([[1, 0, 2], [2, 0, 1]])
    out = unpermute(logits, perm, b["option_mask"])
    assert out[0].tolist()[:2] == [2.0, 1.0] and out[0, 2] == float("-inf")
    assert out[1].tolist() == [2.0, 3.0, 1.0]


def test_losses():
    """预测正确时 soft CE 和 RPS 都更小；相同分布的对称 KL 为 0、相反分布的对称 KL 很大；
    带 -inf 补齐位时反向传播不会产生 NaN。

    good 的 logits [5, -5] 对应概率约 [0.99995, 0.00005]，几乎全押正确答案；bad 正好相反。
    requires_grad_() 告诉 PyTorch 要为这个张量计算梯度，之后 backward() 才能执行。
    """
    mask = torch.tensor([[True, True, False]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    good = torch.tensor([[5.0, -5.0, float("-inf")]])
    bad = torch.tensor([[-5.0, 5.0, float("-inf")]])
    assert soft_ce(good, target, mask) < soft_ce(bad, target, mask)
    assert rps(good, target, mask) < rps(bad, target, mask)
    assert symmetric_kl(good, good, mask).item() == pytest.approx(0, abs=1e-6)
    assert symmetric_kl(good, bad, mask).item() > 1
    loss = soft_ce(good.clone().requires_grad_(), target, mask)
    loss.sum().backward()  # -inf 补齐位不能产生 NaN 梯度


def test_temperature_recovers_scale():
    """构造“过度自信 3 倍”的 logits（真实分布按 z 采样，但模型输出 3z），拟合出的温度应接近 3。

    原理：真实答案是按 softmax(z) 的概率抽出来的，所以 softmax(z) 才是“说真话”的概率。
    模型却输出 3z，softmax(3z) 比真实分布尖锐得多（过度自信）。除以温度 T=3 后正好变回 z，
    所以最优温度应约等于 3。400 个样本有随机误差，因此允许 35% 的相对误差（rel=0.35）。
    """
    rng = random.Random(0)
    logits, targets = [], []
    for _ in range(400):
        z = [rng.gauss(0, 1) for _ in range(4)]
        p = torch.softmax(torch.tensor(z), -1).tolist()
        k = rng.choices(range(4), weights=p)[0]
        logits.append([3 * x for x in z])  # 模型的 logits 放大了 3 倍，即过度自信 3 倍
        targets.append([1.0 if i == k else 0.0 for i in range(4)])
    assert fit_temperature(logits, targets) == pytest.approx(3, rel=0.35)


def test_collate_view_b_only_for_non_score():
    """训练 collate：视图 B 只包含 choice/noul 题，b_index 记录它们在本批中的位置，perm_b 宽度按视图 B 自己的最大选项数。

    本批 3 道题：choice（2 个选项）、score（3 个等级）、noul（2 个选项）。
    视图 B 只有第 0、2 道，所以 b_index = [0, 2]，perm_b 是 2 行 × 2 列（视图 B 中最多 2 个选项）。
    """
    from ajev.schema import Option as O
    from ajev.train.train import TrainSet, make_collate

    ds = [
        Decision(id="c", source="t", type="choice", state="x", instructions="q", options=[O("a"), O("b")], target=[1, 0]),
        Decision(id="s", source="t", type="score", state="x", instructions="q",
                 options=[O("0"), O("1"), O("2")], target=[0, 1, 0]),
        Decision(id="n", source="t", type="noul", state="x", instructions="q",
                 options=[O("true"), O("false")], target=[0, 1]),
    ]
    ts = TrainSet(ds, DecisionEncoder(FakeTok(), max_len=64, max_instr=16), seed=0)
    batch = make_collate(0)([ts[i] for i in range(3)])
    assert batch["b_index"].tolist() == [0, 2]
    assert batch["b"]["input_ids"].size(0) == 2 and tuple(batch["perm_b"].shape) == (2, 2)
    assert tuple(batch["perm_a"].shape) == (3, 3)
    # 一致性权重为 0 时完全不生成视图 B。
    ts1 = TrainSet(ds, DecisionEncoder(FakeTok(), max_len=64, max_instr=16), seed=0, two_views=False)
    batch1 = make_collate(0)([ts1[i] for i in range(3)])
    assert batch1["b"] is None and batch1["b_index"].numel() == 0


def test_checkpoint_integrity_via_meta(tmp_path):
    """续训前的完整性检查：清单里记录的文件大小与磁盘一致才算完整；文件被截断就判为不完整。"""
    import json

    from ajev.train.train import META_FILE, STATE_FILE, checkpoint_step

    d = tmp_path / "last"
    d.mkdir()
    files = {"model.safetensors": b"w" * 100, "decision_head.pt": b"h" * 10, STATE_FILE: b"s" * 50,
             "ajev_config.json": json.dumps({"step": 7}).encode()}
    for name, data in files.items():
        (d / name).write_bytes(data)
    (d / META_FILE).write_text(json.dumps({"step": 7, "sizes": {k: len(v) for k, v in files.items()}}))
    assert checkpoint_step(str(d)) == 7
    (d / "model.safetensors").write_bytes(b"w" * 40)  # 模拟写到一半被打断
    assert checkpoint_step(str(d)) is None
