"""大模型提示词（ajev/lm/prompt.py）的测试：字母选项、noul 的 Yes/No、选项超过 26 个时报错。"""

import pytest

from ajev.lm.prompt import build_user_message, option_lines
from ajev.schema import Decision, Option


def d(type_, names, descs=None):
    descs = descs or [""] * len(names)
    return Decision(id="x", source="s", type=type_, state="STATE", instructions="Q?",
                    options=[Option(n, ds) for n, ds in zip(names, descs)], target=[1.0] + [0.0] * (len(names) - 1))


def test_option_lines_letters_and_noul_yes_no():
    assert option_lines(d("choice", ["refund", "delivery"], ["money back", ""])) == ["A. refund: money back", "B. delivery"]
    # noul 的选项顺序可能被打乱过：字母跟着当前顺序走，名字写成 Yes / No。
    assert option_lines(d("noul", ["false", "true"])) == ["A. No", "B. Yes"]


def test_message_layout_state_first():
    msg = build_user_message(d("score", ["0", "1", "2"], ["low", "mid", "high"]))
    assert msg.index("STATE") < msg.index("Q?") < msg.index("A. 0: low") < msg.index("C. 2: high")
    assert msg.endswith("Reply with the letter of the best option only.")


def test_too_many_options():
    with pytest.raises(ValueError):
        build_user_message(d("choice", [f"o{i}" for i in range(27)]))
