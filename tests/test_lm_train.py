"""大模型 LoRA 训练（ajev/lm/train.py）的单元测试：LoRA 目标层选择、batch 拼接。需要 torch / transformers。"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ajev.schema import Decision, Option  # noqa: E402


def test_lora_targets_language_model_only():
    """有 language_model 子模块时只选它下面的投影层（正则）；没有时按层名选（列表）。"""
    from ajev.lm.train import LINEAR_NAMES, lora_targets

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.up_proj, self.other = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)

    class Multi(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model, self.vision_tower = Block(), Block()

    assert lora_targets(Block()) == list(LINEAR_NAMES)
    rx = lora_targets(Multi())
    import re
    assert re.fullmatch(rx, "language_model.q_proj") and not re.fullmatch(rx, "vision_tower.q_proj")


def test_collate_left_pad_and_view_b():
    """拼 batch：左侧补齐；视图 B 只含 choice/noul 题；option_mask 按各题选项数。"""
    from ajev.lm.train import make_collate

    c = Decision(id="c", source="s", type="choice", state="x", instructions="q",
                 options=[Option("a"), Option("b"), Option("c")], target=[1, 0, 0])
    s = Decision(id="s", source="s", type="score", state="x", instructions="q",
                 options=[Option("0"), Option("1")], target=[0, 1])
    batch = make_collate(0)([(c, [5, 6, 7], [2, 0, 1], [7, 8], [1, 2, 0]), (s, [9], [0, 1], None, None)])
    assert batch["ids"].tolist() == [[5, 6, 7], [0, 0, 9]] and batch["mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch["option_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["b_index"].tolist() == [0] and batch["b"]["ids"].tolist() == [[7, 8]] and batch["b"]["k"] == 3
    assert batch["is_score"].tolist() == [False, True]
