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
    enc = DecisionEncoder(FakeTok(), max_len=64, max_options=40, min_desc=4)
    e = enc.encode(decision())
    assert len(e.input_ids) <= 64 and e.truncated_state
    assert [e.input_ids[p] for p in e.marker_pos] == [3, 3, 3]
    assert e.input_ids[0] == 1 and e.input_ids[-1] == 2


def test_encoder_shrinks_descriptions_then_fails():
    """选项区超预算时先压缩描述（仍保留全部 5 个标记位）；连选项名都放不下时报错。

    max_options=10 时：5 个选项每个至少要 1（标记位）+ 2（选项名 "o0"）= 3 个 token，共 15 > 10，
    所以 ``pytest.raises(ValueError)`` 期望这里抛出 ValueError。
    """
    enc = DecisionEncoder(FakeTok(), max_len=512, max_options=60, min_desc=4)
    e = enc.encode(decision(n_opts=5, state=""))
    assert len(e.marker_pos) == 5
    with pytest.raises(ValueError):
        DecisionEncoder(FakeTok(), max_options=10).encode(decision(n_opts=5))


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
