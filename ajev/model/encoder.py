"""Encoder 决策模型：双向 Transformer 主干（backbone）+ 选项标记位打分头（decision head）。

====================================================================
基础概念
====================================================================

1. encoder 与双向注意力
   Transformer 模型分两大类：
   - decoder（如 GPT、Qwen）：从左到右逐个生成文字，每个位置只能看到它前面的内容（单向）；
   - encoder（如 BERT、mmBERT）：一次读完整条序列，每个位置都能同时看到前后所有内容（双向）。
   我们不需要生成文字，只需要“读懂题目后给选项打分”，所以 encoder 更合适：速度快，
   而且每个选项的标记位能同时看到问题、其他选项和材料。

2. hidden state（隐藏向量）
   encoder 读完序列后，会为**每个 token 位置**输出一个长度为 H 的数字向量（mmBERT-base 中 H=768），
   叫隐藏向量。可以把它理解为“这个位置结合了全文上下文之后的含义”。
   输出张量的形状是 [B, T, H]：B 道题 × 每题 T 个位置 × 每个位置 H 个数。

3. logits 与 softmax
   模型给每个选项打的原始分数叫 logit，可以是任意实数（比如 2.0、-1.5）。
   softmax 把一组 logits 变成概率：p_i = exp(z_i) / Σ_j exp(z_j)，结果都在 0~1 之间且加起来等于 1。
   举例：logits = [2.0, 1.0, 0.0]
         exp 之后 ≈ [7.39, 2.72, 1.00]，总和 ≈ 11.11
         softmax ≈ [0.665, 0.245, 0.090]
   logit 越大，概率越高；logits 之间的差值决定了概率的“尖锐程度”。

4. 为什么补齐位的 logit 填 -inf
   一个 batch 里有的题 2 个选项、有的题 5 个，logits 要补齐成同样的 K 列。
   补出来的“假选项”必须不分走任何概率：exp(-inf) = 0，所以 softmax 后它的概率正好是 0，
   真实选项的概率不受影响。
   举例：logits = [2.0, 1.0, -inf] → softmax = [0.731, 0.269, 0.0]，和只有两个选项时完全一样。

====================================================================
在 AJev 流程中的位置
====================================================================

``ajev.model.encoding`` 把题目编码成带标记位的序列 → **本模块前向** 得到每个选项的 logit
→ 训练时算损失（ajev.train）、推理时做 softmax 得到概率（ajev.model.predictor）。

工作方式：
- 输入中每个选项前都有一个 ``<mask>`` 标记位。主干（默认 mmBERT-base）做一次双向编码后，
  每个标记位上的隐藏向量已经“看过”问题、所有选项和材料。
- 决策头把“标记位向量”与整条序列的 ``<bos>`` 摘要向量组合起来，映射成一个 logit。
- 对同一道题的所有选项 logit 做 softmax，就得到这道题的概率分布。

保存格式（一个目录就是一个 checkpoint，即模型的“存档”）：
- 主干权重与 config：HuggingFace 标准格式（``save_pretrained``），可被 ``AutoModel`` 直接加载；
- 决策头权重：``decision_head.pt``；
- tokenizer 文件；
- AJev 自己的配置：``ajev_config.json``（编码参数、基座模型名、每种题型的校准温度等）。
"""

from __future__ import annotations

import json
import os

import torch
# nn 是 PyTorch 的神经网络模块库：nn.Module 是所有模型/层的基类，nn.Linear 是全连接层等。
from torch import nn
# AutoModel / AutoTokenizer 会根据模型目录里的 config 自动选择正确的模型类和分词器类。
from transformers import AutoModel, AutoTokenizer

# checkpoint 目录中决策头权重和 AJev 配置的文件名。
HEAD_FILE = "decision_head.pt"
AJEV_CONFIG = "ajev_config.json"


class DecisionHead(nn.Module):
    """选项打分头：把每个选项标记位的隐藏向量打成一个 logit。

    输入特征 = [opt, ctx, opt * ctx]（三段拼接，维度 3H）：
    - opt：选项标记位的向量，代表“这个选项在当前上下文中的含义”；
    - ctx：``<bos>`` 位置的向量，代表整道题（问题 + 材料）的摘要；
    - opt * ctx：逐元素乘积。两个向量在同一维度上都大时乘积就大，相当于显式提供
      “选项与题目是否匹配”的交互特征，让小小的打分头更容易学。

    网络结构：Linear(3H→H) → GELU → Dropout → Linear(H→1)
    - Linear：全连接层，y = xW + b，W 和 b 是要学习的参数；
    - GELU：非线性激活函数（类似 ReLU 但更平滑）。没有非线性，多层 Linear 叠起来仍等价于一层；
    - Dropout：训练时随机把一部分数置 0，防止过拟合；推理（eval 模式）时自动关闭。
    """

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        # 继承 nn.Module 的类必须先调用父类的 __init__，PyTorch 才能登记这个模块里的参数。
        super().__init__()
        # nn.Sequential：把几层按顺序串起来，调用时数据依次流过每一层。
        self.net = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, opt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opt: [B, K, H]，每道题每个选项标记位的隐藏向量（B=题数，K=最大选项数，H=隐藏维度）。
            ctx: [B, H]，每道题 ``<bos>`` 位置的隐藏向量。

        Returns:
            [B, K]，每个选项一个 logit（尚未做掩码）。

        举个例子：B=2 道题，K=3，H=768
            opt: [2, 3, 768]，ctx: [2, 768]
            ctx 广播后: [2, 3, 768]（同一道题的 3 个选项共用同一个摘要向量）
            拼接后: [2, 3, 2304] → Linear → [2, 3, 768] → GELU/Dropout → Linear → [2, 3, 1]
            squeeze(-1) 去掉最后那个长度为 1 的维度 → [2, 3]
        """
        # unsqueeze(1)：在第 1 维插入长度为 1 的维度，[B, H] → [B, 1, H]；
        # expand_as(opt)：把这个维度“复制”成 K 份（不真正占新内存），→ [B, K, H]。
        ctx = ctx.unsqueeze(1).expand_as(opt)
        # torch.cat(..., dim=-1)：在最后一维拼接，[B,K,H] ×3 → [B,K,3H]。
        return self.net(torch.cat([opt, ctx, opt * ctx], dim=-1)).squeeze(-1)


class DecisionModel(nn.Module):
    """完整的决策模型 = 主干 backbone（mmBERT，约 3 亿参数）+ 决策头 DecisionHead（约 180 万参数）。

    主干负责“读懂”，决策头负责“打分”。主干是预训练好的，决策头是新加的、随机初始化的。
    """

    def __init__(self, backbone: nn.Module, head_dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        # 决策头的输入维度要和主干的隐藏维度 H 一致，所以从主干的 config 里读。
        self.head = DecisionHead(backbone.config.hidden_size, head_dropout)

    def forward(self, input_ids, attention_mask, marker_pos, option_mask) -> torch.Tensor:
        """前向计算，返回每个选项的 logit。

        在 PyTorch 中，写 ``model(**batch)`` 实际上就是调用这个 forward 方法
        （``**batch`` 把字典展开成关键字参数）。

        Args:
            input_ids: [B, T]，补齐后的 token id（T=batch 内最长序列长度）。
            attention_mask: [B, T]，1 表示真实 token，0 表示补齐位。
            marker_pos: [B, K]，每个选项标记位在序列中的下标；补齐的选项位填 0（指向 <bos>，结果会被掩掉）。
            option_mask: [B, K]，True 表示真实选项，False 表示补齐出来的选项位。

        Returns:
            [B, K] 的 float32 logits；补齐的选项位为 -inf，softmax 后概率为 0。

        举个例子：B=2，T=6，K=3，H=4（为了好写把 H 设得很小）
            hidden:     [2, 6, 4]  每道题 6 个位置，每个位置 4 个数
            marker_pos: [[1, 3, 0],
                         [1, 2, 3]]
            gather 后 opt: [2, 3, 4]，其中 opt[0, 1] = hidden[0, 3]（第 1 题第 2 个选项的标记位在第 3 位）
            head 输出: [2, 3]，例如 [[0.8, -0.3, 0.1], [1.2, 0.4, -0.5]]
            option_mask = [[T, T, F], [T, T, T]] → 最终 [[0.8, -0.3, -inf], [1.2, 0.4, -0.5]]
        """
        # 主干前向：.last_hidden_state 是最后一层的隐藏向量，[B, T, H]。
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # torch.gather：按下标“挑出”指定位置的元素。这里要在第 1 维（序列位置 T）上挑，
        # 所以先把 marker_pos 从 [B, K] 扩展成 [B, K, H]（每个位置的 H 个数都用同一个下标），
        # gather 的结果 opt[b, k, :] = hidden[b, marker_pos[b, k], :]，形状 [B, K, H]。
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        opt = torch.gather(hidden, 1, idx)
        # hidden[:, 0] 取每道题第 0 个位置（<bos>）的向量作为整道题的摘要，[B, H]。
        # .float() 转成 float32：混合精度下主干输出可能是 fp16，后面的 softmax/损失在 float32 下更稳定。
        logits = self.head(opt, hidden[:, 0]).float()
        # masked_fill(条件, 值)：条件为 True 的位置填入该值。~option_mask 是“取反”，即补齐位。
        return logits.masked_fill(~option_mask, float("-inf"))

    # ---- 保存与加载 ------------------------------------------------------------
    def save(self, path: str, tokenizer=None, extra: dict | None = None) -> None:
        """把模型保存为一个 checkpoint 目录。

        Args:
            path: 目标目录（不存在会自动创建）。
            tokenizer: 若提供，一并保存，保证推理时用的分词与训练时完全一致。
            extra: 写入 ``ajev_config.json`` 的额外信息，例如编码参数、基座模型名、训练步数。

        state_dict() 是 PyTorch 模块的“参数字典”（参数名 → 张量），保存它就保存了所有可学习参数。
        """
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, HEAD_FILE))
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        with open(os.path.join(path, AJEV_CONFIG), "w") as f:
            json.dump(extra or {}, f, indent=2)

    # @classmethod：类方法，用 DecisionModel.from_pretrained(...) 直接调用，返回一个新建的模型对象。
    @classmethod
    def from_pretrained(cls, path: str, **backbone_kw) -> "DecisionModel":
        """加载模型：既可以加载已保存的 AJev checkpoint，也可以从 HF 基座模型 id 开始全新训练。

        - 如果 ``path`` 下有 ``decision_head.pt``，说明是 AJev checkpoint，会同时加载决策头；
        - 否则（例如 ``jhu-clsp/mmBERT-base``）只加载主干，决策头保持随机初始化，等待训练。

        加载基座模型时日志里出现 “UNEXPECTED: decoder.weight / head.* ...” 是正常的：
        那是 mmBERT 预训练用的“猜 <mask> 词”输出层，我们用不到，所以被丢弃。
        """
        model = cls(AutoModel.from_pretrained(path, **backbone_kw))
        head = os.path.join(path, HEAD_FILE)
        if os.path.exists(head):
            # map_location="cpu"：先把权重读到 CPU 内存，之后再随模型 .to(device) 搬到 GPU，避免设备不匹配。
            model.head.load_state_dict(torch.load(head, map_location="cpu"))
        return model


def load_tokenizer(path: str):
    """加载 tokenizer（HF 模型 id 如 "jhu-clsp/mmBERT-base"，或本地 checkpoint 目录均可）。

    版本兼容：transformers 5.x 保存的 tokenizer_config.json 把类名写成 ``TokenizersBackend``，
    旧版（4.x，例如 Intel Mac 上只能用的版本）不认识这个类名，``AutoTokenizer`` 会报错。
    这时退回到直接读 ``tokenizer.json``（分词规则本身在新旧版本之间是通用的），
    再从 tokenizer_config.json 里补上特殊 token（<bos>、<eos>、<mask>、<pad> 等）。
    两种方式切出来的 token id 完全相同。
    """
    try:
        return AutoTokenizer.from_pretrained(path)
    except (ValueError, AttributeError, ImportError):
        cfg_path = os.path.join(path, "tokenizer_config.json")
        tok_path = os.path.join(path, "tokenizer.json")
        if not (os.path.exists(cfg_path) and os.path.exists(tok_path)):
            raise
        from transformers import PreTrainedTokenizerFast

        with open(cfg_path) as f:
            cfg = json.load(f)
        special = {k: cfg[k] for k in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token",
                                       "cls_token", "sep_token") if isinstance(cfg.get(k), str)}
        return PreTrainedTokenizerFast(tokenizer_file=tok_path, **special)


def load_ajev_config(path: str) -> dict:
    """读取 checkpoint 目录下的 ``ajev_config.json``；不存在时返回空字典（例如直接用基座模型）。"""
    p = os.path.join(path, AJEV_CONFIG)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)
