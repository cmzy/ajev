"""推理封装：加载训练好的 checkpoint，实现 ``Predictor`` 协议（输入一批 Decision，输出每题的概率分布）。

====================================================================
基础概念
====================================================================

1. 推理（inference）与训练的区别
   训练要算梯度、更新参数；推理只做前向计算拿结果。推理时要做两件事：
   - ``model.eval()``：切到评估模式，关闭 Dropout 等“训练专用的随机行为”，保证同样输入得到同样输出；
   - ``torch.no_grad()``：不记录计算图、不算梯度，省显存、更快。
   （对应地，训练时用 ``model.train()`` 打开 Dropout。注意：eval() 不会关闭梯度，no_grad 也不会关闭
   Dropout，两者作用不同，推理时通常两个都要。）

2. temperature（温度）
   模型输出的概率可能“过度自信”（说 95% 但实际只对 80%）或“不够自信”。温度是一个事后校正：
       p = softmax(logits / T)
   - T > 1：logits 被缩小，概率分布变平，降低自信；
   - T < 1：logits 被放大，分布变尖，提高自信；
   - T = 1：不变。
   举例：logits = [2.0, 0.0]
       T=1 → [0.881, 0.119]
       T=2 → logits/2 = [1.0, 0.0] → [0.731, 0.269]
   注意温度不改变 logits 的大小顺序，所以**不影响准确率**，只影响概率是否可信。
   温度按题型（noul / choice / score）分别拟合，保存在 checkpoint 的 ``ajev_config.json`` 里；
   未校准时 T=1。

3. 混合精度（mixed precision）
   默认情况下 GPU 用 32 位浮点数（float32）计算。改用 16 位浮点数（fp16 或 bf16）可以
   让显存减半、速度明显提升，代价是精度变低。``torch.autocast`` 会自动挑选适合用 16 位的运算
   （如矩阵乘法），其余运算仍保持 32 位，所以叫“混合”精度。
   - fp16：精度较好但数值范围小（最大约 65504），很小的数会变成 0（下溢），训练时需要 GradScaler 帮忙；
   - bf16：数值范围和 float32 一样大，不易溢出，但只有较新的 GPU 支持（A100、L4 支持，T4 不支持）。

====================================================================
在 AJev 流程中的位置
====================================================================

训练产出 checkpoint → **本模块推理** → 校准（ajev.calibrate 用原始 logits 拟合温度）
→ 评测（ajev.eval.evaluate 用 ``--predictor model``）→ 以后的 API 服务也会复用它。
训练过程中的验证（ajev.train.train.evaluate）也调用这里的 ``predict_logits``。
"""

from __future__ import annotations

import sys

import torch

from ajev.model.batching import collate, length_sorted_batches
from ajev.model.encoder import DecisionModel, load_ajev_config, load_tokenizer
from ajev.model.encoding import DecisionEncoder
from ajev.schema import Decision


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """选择混合精度的数据类型（训练和推理共用）。

    - CPU：返回 None，不开混合精度（CPU 上 16 位计算通常没有加速效果）；
    - 支持 bf16 的 GPU（如 A100 / L4）：用 bfloat16，数值范围大，不需要 GradScaler；
    - 不支持 bf16 的 GPU（如 Colab 免费的 T4）：用 float16，训练时需配合 GradScaler 防止梯度下溢。
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# @torch.no_grad() 是装饰器写法：整个函数体都在“不算梯度”的模式下运行。
@torch.no_grad()
def predict_logits(model: DecisionModel, encoder: DecisionEncoder, decisions: list[Decision],
                   device: torch.device, batch_size: int = 32) -> list[list[float]]:
    """对一批题做前向，返回每道题的原始 logits（未除以温度、未做 softmax）。

    训练中的验证、温度校准和推理都调用这个函数。

    Args:
        model: 决策模型。
        encoder: 与训练时相同配置的 DecisionEncoder。
        decisions: 要预测的题。
        device: 运行设备（torch.device("cuda") 或 torch.device("cpu")）。
        batch_size: 每批最多多少道题。

    Returns:
        与 decisions 一一对应的 logits 列表，每个列表长度等于该题的选项数，顺序与选项顺序一致。

    举个例子：3 道题，选项数分别是 2、5、3，batch_size=2
        第 1 步：逐题编码，得到 3 个 Encoded；
        第 2 步：按长度排序分批，例如 [[0, 2], [1]]；
        第 3 步：第一批 2 道题补齐成张量，前向得到 logits [2, 3]（K=3，第 0 题的第 3 列是 -inf）；
                 第二批 1 道题，logits [1, 5]；
        第 4 步：去掉补齐位、按原下标放回：
                 out = [[l00, l01], [l10, ..., l14], [l20, l21, l22]]
    """
    # 切换到评估模式：关闭 Dropout，保证结果确定。
    model.eval()
    encs = [encoder.encode(d) for d in decisions]
    # 预先建好与输入等长的结果列表，之后按下标填入，这样分批时打乱顺序也没关系。
    out: list[list[float]] = [[] for _ in decisions]
    dtype = autocast_dtype(device)
    # 按长度排序分批以减少补齐；结果再按原下标 i 放回，保证输出顺序与输入一致。
    for idx in length_sorted_batches([len(e.input_ids) for e in encs], batch_size):
        # .to(device)：把张量从 CPU 内存搬到 GPU 显存（模型在哪，数据就要在哪）。
        batch = {k: v.to(device) for k, v in collate([encs[i] for i in idx], encoder.tok.pad_token_id).items()}
        # enabled=False 时 autocast 什么也不做，所以 CPU 上同一段代码也能跑。
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(**batch)
        # .cpu() 搬回 CPU，.tolist() 转成普通 Python 列表。
        for row, i in zip(logits.float().cpu(), idx):
            # 只取真实选项部分，去掉补齐位（-inf）。
            out[i] = row[: len(decisions[i].options)].tolist()
    return out


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """带温度的 softmax：T>1 让分布更平（降低过度自信），T<1 让分布更尖。

    举个例子：softmax([2.0, 0.0]) ≈ [0.881, 0.119]；softmax([2.0, 0.0], temperature=2) ≈ [0.731, 0.269]。
    """
    t = torch.tensor(logits) / temperature
    return torch.softmax(t, dim=-1).tolist()


class EncoderPredictor:
    """加载 checkpoint 目录并提供 ``predict`` / ``predict_logits`` 的推理器。

    Args:
        path: checkpoint 目录（包含主干权重、decision_head.pt、tokenizer、ajev_config.json）。
        device: "cuda" / "cpu"；不传时有 GPU 就用 GPU。
        batch_size: 推理时每批最多多少道题（越大越快，但越占显存）。
        temperatures: 每种题型的温度，例如 ``{"noul": 1.2, "choice": 0.9}``。
            传 None 时读取 ajev_config.json 中的值；传 ``{}`` 表示强制不校准（拟合温度时就这样用，
            因为拟合需要的是未校准的原始输出）。

    举个例子：
        pred = EncoderPredictor("runs/sft1/best")
        pred.predict([decision])   # → [[0.92, 0.08]]，一道 noul 题：P(true)=0.92
    """

    def __init__(self, path: str, device: str | None = None, batch_size: int = 32,
                 temperatures: dict[str, float] | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = load_ajev_config(path)
        # 编码参数（如 max_len）沿用训练时保存的配置，保证推理与训练输入一致。
        # **cfg.get("encoding", {}) 把字典展开成关键字参数，例如 max_len=1024。
        self.encoder = DecisionEncoder(load_tokenizer(path), **cfg.get("encoding", {}))
        self.model = DecisionModel.from_pretrained(path).to(self.device)
        self.batch_size = batch_size
        # 每种题型的温度由 ajev.calibrate 拟合；没有校准时默认 1.0。
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        # 温度只对拟合它时的那份权重有效。没有温度，或者权重在校准之后又被训练改过
        # （训练保存 checkpoint 时会重写配置、清掉温度，并更新 step），都要明确提示，
        # 而不是悄悄用 T=1 输出未校准的概率。传入 temperatures={} 表示有意不用温度，不提示。
        if temperatures is None:
            if not self.temperatures:
                print(f"[ajev] warning: {path} has no calibrated temperatures; probabilities are "
                      "uncalibrated (run `python -m ajev.calibrate`)", file=sys.stderr)
            elif cfg.get("calibrated_at_step") != cfg.get("step"):
                print(f"[ajev] warning: temperatures in {path} were fitted at step "
                      f"{cfg.get('calibrated_at_step')} but the weights are from step {cfg.get('step')}; "
                      "re-run `python -m ajev.calibrate`", file=sys.stderr)

    def predict_logits(self, decisions: list[Decision]) -> list[list[float]]:
        """返回原始 logits（不做温度缩放），供温度校准使用。"""
        return predict_logits(self.model, self.encoder, decisions, self.device, self.batch_size)

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        """返回校准后的概率分布：每道题按其题型的温度做 softmax。"""
        logits = self.predict_logits(decisions)
        return [softmax(lg, self.temperatures.get(d.type, 1.0)) for d, lg in zip(decisions, logits)]
