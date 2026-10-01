"""Encoder 决策模型的监督训练（软标签 + 选项打乱一致性）。

在 AJev 流程中的位置：数据构建（ajev.data.build）→ **本模块训练** → 温度校准（ajev.calibrate）
→ 评测（ajev.eval.evaluate）。在 Colab 上通常由 scripts/colab_job.py 的 ``train`` 子命令在后台启动。

用法：

    python -m ajev.train.train --train data/build/train.jsonl --val data/build/val.jsonl --out runs/sft1

每道题的训练目标（视图 A = 选项随机打乱后的一种顺序）：

    soft_ce(A, 平滑后的 target)                   所有题型：软标签交叉熵
  + rps_weight * RPS(A)                          仅 score（打分题）：有序等级的排序概率评分
  + consistency_weight * symKL(A, B)             仅 choice/noul：B 是同一道题换一种选项顺序，
                                                 两个视图的分布要一致（压制位置偏置）

checkpoint：
- ``{out}/last``：可续训，包含模型权重 + 优化器 / 学习率调度器 / GradScaler 状态 + 训练进度；
- ``{out}/best``：验证集平均准确率最高的模型（只有权重，不含优化器状态）。
再次运行同一条命令会自动从 ``last`` 续训。
``--hub-repo`` 会额外把 checkpoint 上传到 Hugging Face Hub（需要 HF_TOKEN），
这样 Colab VM 被回收也不会丢进度。

====================================================================
训练的基础概念（初学者先读这一段）
====================================================================

1. 一次训练 step 在做什么
   前向（forward）：把一批题喂给模型，得到预测；
   算损失（loss）：比较预测和标准答案，得到一个“错得有多离谱”的数；
   反向（backward）：loss.backward() 自动算出每个参数的梯度（往哪个方向改能让损失变小）；
   更新（optimizer.step）：优化器按梯度把参数改一点点；然后清空梯度，进入下一步。

2. 学习率（learning rate）与 warmup + cosine 调度
   学习率决定每次参数改多大一步。太大会训崩，太小学得慢。
   - warmup（预热）：刚开始几百步把学习率从接近 0 线性升到目标值。因为决策头是随机初始化的，
     一开始梯度又大又乱，直接用大学习率容易把预训练好的主干参数“冲坏”；
   - cosine（余弦衰减）：之后学习率按余弦曲线平滑地降到 0，后期小步微调，更容易收敛到好的位置。
   例如总共 1000 步、warmup=6%：第 0~59 步从 1/60 倍升到 1 倍，第 60 步后从 1 倍逐渐降到 0。

3. AdamW 与 weight decay（权重衰减）
   AdamW 是目前最常用的优化器：它为每个参数自适应地调整步长（梯度一直很大的参数步子放小，
   一直很小的放大）。weight decay 每步把参数往 0 拉一点点，防止参数变得过大，起正则化（防过拟合）作用。
   代价：Adam 要为每个参数额外存 2 个数（动量和方差），所以优化器状态占的显存约是参数本身的 2 倍。

4. 梯度累积（gradient accumulation）
   显存只够一次算 8 道题，但我们想要 16 道题的批大小（大 batch 梯度更稳）。做法：连续算 2 个
   小批（micro-batch），每次只 backward 不更新，梯度会自动累加；累积够了再做一次更新。
   为了让累加结果等于“一个大批的平均梯度”，每个小批的损失要先除以累积次数 grad_accum。

5. 梯度裁剪（gradient clipping）
   偶尔某个 batch 会产生特别大的梯度，一步就把参数改坏。裁剪规则：如果所有梯度合起来的长度
   （范数）超过 1.0，就整体按比例缩小到 1.0。方向不变，只限制步子大小。

6. 混合精度与 GradScaler
   T4 用 fp16 计算以节省显存、提高速度（见 ajev/model/predictor.py 的说明）。但 fp16 能表示的
   最小正数约 6e-8，很多梯度比这还小，会直接变成 0（下溢），参数就学不动了。
   GradScaler 的办法：先把损失乘以一个大数（如 65536），梯度也就跟着放大，不会下溢；
   更新参数前再除回去（unscale）。如果放大后出现 inf/NaN（上溢），就跳过这一步并把倍数调小。
   bf16 数值范围够大，不需要 GradScaler，此时它处于禁用状态，所有调用都是空操作。

7. 梯度检查点（gradient checkpointing）
   反向传播需要用到前向时每一层的中间结果（激活值），默认全部存着，非常占显存。
   开启后只保存少量“检查点”，反向时需要哪段就重新算一遍那段前向。用约 30% 的额外计算换大量显存。

8. 冻结 embedding 为什么省显存
   mmBERT 的词嵌入矩阵是 25.6 万（词表大小）× 768 ≈ 1.97 亿个参数，占全模型的 2/3。
   训练一个参数要存：参数本身 + 梯度 + Adam 的 2 个状态 = 4 份（float32 每份 4 字节）。
   冻结后就不用存它的梯度和 Adam 状态，约省 1.97 亿 × 3 × 4 字节 ≈ 2.4 GB 显存。
   同时也保护了预训练好的多语言（含中文）词向量，不被少量训练数据带偏。

9. 为什么按 token 预算分批
   显存占用主要取决于“这一批补齐后一共有多少个 token”（题数 × 最长序列长度）。如果固定每批 16 道题，
   碰上 16 道都是 1000 token 的长题，就是 16000 个 token，T4 直接显存溢出（我们实际遇到过）；
   而 16 道短题只有几百个 token，又很浪费。按 token 预算分批：短题一批放很多，长题一批放几道，
   每批 token 数大致恒定，显存稳定、利用率也高。

10. 断点续训需要保存什么
    Colab 免费 VM 随时可能被回收，所以要能从中断处精确接着训。只存模型权重不够：
    - 优化器状态：Adam 的动量和方差，丢了相当于优化器“失忆”，训练会抖一下；
    - 学习率调度器状态：否则 warmup 会重新来一遍；
    - GradScaler 状态：当前的放大倍数；
    - 训练进度：step / micro_total / epoch / epoch_micro，用来跳过已经训练过的数据；
    - best：目前最好的验证分数，避免续训后把更差的模型当成 best 覆盖。

11. DataLoader 与 num_workers
    DataLoader 负责按批取数据、调用 collate 拼成张量。num_workers=2 表示开 2 个子进程在后台
    准备下一批数据（分词、补齐），GPU 训练当前批的同时 CPU 准备下一批，GPU 不用干等。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time

import torch
# Dataset：定义“第 i 条数据是什么”；DataLoader：负责按批取数据、多进程预取。
from torch.utils.data import DataLoader, Dataset

from ajev import metrics
from ajev.model.batching import collate, pad_targets
from ajev.model.encoder import DecisionModel, load_tokenizer
from ajev.model.encoding import DecisionEncoder
from ajev.model.predictor import autocast_dtype, predict_logits, softmax
from ajev.schema import Decision, read_jsonl
from ajev.train.losses import rps, smooth, soft_ce, symmetric_kl, unpermute

# last checkpoint 中保存优化器、调度器、GradScaler 和训练进度的文件名。
STATE_FILE = "trainer_state.pt"


def checkpoint_step(path: str) -> int | None:
    """检查一个可续训的 checkpoint 目录是否完整；完整则返回它的步数，否则返回 None。

    为什么需要：Colab 的 VM 可能在任意时刻被回收。如果恰好在写 checkpoint 时被打断，
    目录里可能只有一半文件，或者文件被截断。续训时如果直接加载这种目录，要么报错，
    要么（更糟）权重和优化器状态来自不同的步数。

    判断标准（全部满足才算完整）：
    1. 权重 model.safetensors、决策头 decision_head.pt、配置 ajev_config.json、
       训练状态 trainer_state.pt 都存在；
    2. trainer_state.pt 能被完整读出（文件被截断时 torch.load 会抛异常）；
    3. 配置里记录的 step 与训练状态里的 step 一致（说明权重和状态是同一次保存写的）。
    """
    from ajev.model.encoder import AJEV_CONFIG, HEAD_FILE

    needed = ["model.safetensors", HEAD_FILE, AJEV_CONFIG, STATE_FILE]
    if not all(os.path.exists(os.path.join(path, f)) for f in needed):
        return None
    try:
        st = torch.load(os.path.join(path, STATE_FILE), map_location="cpu", weights_only=False)
        with open(os.path.join(path, AJEV_CONFIG)) as f:
            cfg_step = json.load(f).get("step")
    except Exception:  # 文件被截断 / 损坏
        return None
    return st["step"] if st.get("step") == cfg_step else None


def find_resumable(last_dir: str) -> str | None:
    """在 ``last`` 和它的备份 ``last.prev`` 中，找出最新的完整 checkpoint。

    返回可用的目录；两个都不存在时返回 None（表示从头训练）；
    目录存在但都不完整时直接报错退出，避免悄悄从头开始、浪费已有进度。
    """
    candidates = [(checkpoint_step(p), p) for p in (last_dir, last_dir + ".prev") if os.path.isdir(p)]
    if not candidates:
        return None
    valid = [(s, p) for s, p in candidates if s is not None]
    if not valid:
        raise SystemExit(f"checkpoints under {last_dir}(.prev) are all incomplete; "
                         "move them away to start fresh")
    return max(valid)[1]


class TrainSet(Dataset):
    """训练集：每道题产出两个独立打乱选项顺序的“视图”（score 题的等级顺序保持不变）。

    继承 torch 的 Dataset 只需要实现两个方法：``__len__``（共多少条）和 ``__getitem__``（第 i 条是什么）。

    为什么要两个视图：
    - 计算选项打乱一致性损失（两个视图的预测应该相同）；
    - 每个 epoch 的打乱都不同，相当于免费的数据增强，模型没法靠“记住答案在第几个位置”来作弊。
    随机数种子由 (seed, epoch, 题目下标) 决定，所以断点续训后得到的视图与中断前完全一致。

    举个例子：一道 choice 题，原始选项 [退款, 物流, 账户]，标准答案“退款”
        视图 A 可能是 [物流, 退款, 账户]，perm_a=[1, 0, 2]
        视图 B 可能是 [账户, 物流, 退款]，perm_b=[2, 1, 0]
        target 始终按原始顺序存：[1, 0, 0]
    """

    def __init__(self, decisions: list[Decision], encoder: DecisionEncoder, seed: int) -> None:
        self.decisions = decisions
        self.encoder = encoder
        self.seed = seed
        # 当前 epoch，由训练循环每轮开始时设置，影响打乱方式。
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.decisions)

    def _view(self, d: Decision, rng: random.Random) -> tuple:
        """生成一个视图：返回 (编码结果, perm)，其中视图的第 j 个选项 = 原始第 perm[j] 个选项。"""
        perm = list(range(len(d.options)))
        # score 题的选项是有序等级（0 < 1 < 2 ...），打乱会破坏语义，所以不打乱。
        if d.type != "score":
            rng.shuffle(perm)
        # {**d.__dict__, "options": ...}：复制题目的所有字段，只替换 options，得到一道“换了选项顺序”的新题。
        view = Decision(**{**d.__dict__, "options": [d.options[i] for i in perm]})
        return self.encoder.encode(view), perm

    def __getitem__(self, i: int):
        """返回 (原始题目, 视图A编码, 视图A的perm, 视图B编码, 视图B的perm)。"""
        d = self.decisions[i]
        # 用字符串当种子：同一个 (seed, epoch, i) 永远得到同样的随机序列，结果可复现。
        rng = random.Random(f"{self.seed}/{self.epoch}/{i}")
        enc_a, perm_a = self._view(d, rng)
        enc_b, perm_b = self._view(d, rng)
        return d, enc_a, perm_a, enc_b, perm_b


def make_collate(pad_id: int):
    """构造 DataLoader 的 collate 函数，把 TrainSet 的若干条输出拼成一个训练 batch。

    collate 函数：DataLoader 每次取出一批 __getitem__ 的结果（一个列表），交给它拼成张量。
    这里用“函数里返回函数”（闭包）的写法，是为了把 pad_id 这个参数“记住”在 fn 里。

    返回的 batch 字典：
    - ``a`` / ``b``: 两个视图各自的模型输入（见 ajev.model.batching.collate）；
    - ``perm_a`` / ``perm_b``: [B, K]，两个视图的选项排列，用于把 logits 放回原始顺序；
    - ``target``: [B, K]，原始选项顺序下的目标分布；
    - ``is_score``: [B]，是否为 score 题（决定用不用 RPS、算不算一致性损失）。

    举个例子：一批 2 道题，第 1 道 2 个选项（choice），第 2 道 3 个选项（score）
        K = 3
        perm_a = [[1, 0, 2],      ← 第 1 道真实 perm 是 [1, 0]，补齐位 2 映射到自身
                  [0, 1, 2]]      ← score 题不打乱
        target = [[1.0, 0.0, 0.0],
                  [0.0, 0.3, 0.7]]
        is_score = [False, True]
    """

    def fn(items):
        # zip(*items)：把“每条一个元组”的列表转置成“每个字段一个元组”。
        ds, enc_a, perm_a, enc_b, perm_b = zip(*items)
        a, b = collate(list(enc_a), pad_id), collate(list(enc_b), pad_id)
        # 两个视图是同一批题，选项数相同，所以 K 也相同。
        k = a["option_mask"].size(1)

        def perm_tensor(perms):
            # 补齐位映射到自身，这样 unpermute 之后 -inf 仍然留在补齐位上。
            return torch.tensor([p + list(range(len(p), k)) for p in perms])

        return {
            "a": a,
            "b": b,
            "perm_a": perm_tensor(perm_a),
            "perm_b": perm_tensor(perm_b),
            "target": pad_targets([d.target for d in ds], k),  # 原始（未打乱）的选项顺序
            "is_score": torch.tensor([d.type == "score" for d in ds]),
        }

    return fn


def token_budget_batches(lengths: list[int], max_tokens: int, max_batch: int, seed: int,
                         mega: int = 2048) -> list[list[int]]:
    """按 token 预算分批（训练用；为什么要这样做见模块说明第 9 条）。

    步骤：
    第 1 步：先整体打乱（保证每批题目来源多样）；
    第 2 步：每 ``mega`` 条为一个大块，块内按长度排序（长度相近的题放在一起，减少补齐浪费）；
    第 3 步：顺序切批：一旦再加一条会让“批大小 × 本批最长序列”超过 ``max_tokens``，
            或题数超过 ``max_batch``，就另起一批（因为块内已按长度升序，新加入的题就是本批最长的）；
    第 4 步：最后再打乱各批的顺序，避免训练时先全是短题、后全是长题。

    为什么不直接全局排序：那样每批里的题都来自相近长度，而长度又和数据来源强相关（比如长题几乎都是
    HelpSteer2），一批全是同一种题会让梯度很偏。分块排序是“减少补齐”和“保持随机性”的折中。

    Args:
        lengths: 每道题编码后的 token 长度。
        max_tokens: 每批（单个视图）补齐后的最大 token 数。
        max_batch: 每批最多题数。
        seed: 随机种子；每个 epoch 用不同种子，同一 epoch 结果确定，便于断点续训时精确跳过已训练的批。
        mega: 排序块大小。

    Returns:
        批列表，每批是题目下标列表。

    举个例子：lengths=[100, 900, 120, 80, 1000]，max_tokens=2000，max_batch=32（假设打乱后顺序不变）
        块内按长度排序：下标 [3, 0, 2, 1, 4]，长度 [80, 100, 120, 900, 1000]
        依次加入：[3] → [3,0] → [3,0,2]（3×120=360）→ 加下标 1 会变成 4×900=3600 > 2000，另起一批
                  [1] → 加下标 4 会变成 2×1000=2000，不超过，于是 [1, 4]
        结果（打乱批顺序前）：[[3, 0, 2], [1, 4]]——3 道短题一批，2 道长题一批。
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    # 第 1 步：整体打乱。
    rng.shuffle(idx)
    batches: list[list[int]] = []
    for s in range(0, len(idx), mega):
        cur: list[int] = []
        # 第 2 步：块内按长度升序排列。
        for i in sorted(idx[s : s + mega], key=lengths.__getitem__):
            # 第 3 步：加入第 i 题后，补齐 token 数 = (题数) × (第 i 题的长度，即本批最长)。
            if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * lengths[i] > max_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    # 第 4 步：打乱批的顺序。
    rng.shuffle(batches)
    return batches


def evaluate(model, encoder, decisions, device, batch_size) -> dict:
    """在一个验证集上评估：返回整体指标，以及按题型（acc_noul 等）和语言（acc_en / acc_zh）的准确率。

    验证集是模型训练时没见过的题，用来判断模型是真学会了还是只背下了训练集（过拟合）。
    这里不做温度校准（温度在训练结束后由 ajev.calibrate 单独拟合）。
    predict_logits 内部会调用 model.eval()，所以评估完要用 model.train() 切回训练模式，
    否则后续训练时 Dropout 会一直处于关闭状态。

    返回示例：{"n": 1500, "accuracy": 0.71, "brier": 0.05, ..., "acc_noul": 0.80, "acc_zh": 0.68}
    """
    logits = predict_logits(model, encoder, decisions, device, batch_size)
    preds = [softmax(lg) for lg in logits]
    out = metrics.compute(decisions, preds)
    for t, m in metrics.breakdown(decisions, preds, lambda d: d.type).items():
        out[f"acc_{t}"] = m["accuracy"]
    for lang, m in metrics.breakdown(decisions, preds, lambda d: d.lang).items():
        out[f"acc_{lang}"] = m["accuracy"]
    model.train()
    return out


def main(argv: list[str] | None = None) -> None:
    """命令行入口：解析参数 → 加载模型和数据 → 训练循环（定期打印、评估、保存）→ 收尾。

    整体流程：
    第 1 步：解析命令行参数；
    第 2 步：判断是否续训，加载 tokenizer 和模型，设置冻结 embedding / 梯度检查点；
    第 3 步：读训练集和验证集，预先算出每道题的 token 长度（用于分批）；
    第 4 步：创建优化器、学习率调度器、GradScaler；续训时恢复它们的状态；
    第 5 步：训练主循环（每个 step 的细节见循环上方的注释）；
    第 6 步：收尾，做最后一次评估和保存。
    """
    # ---- 第 1 步：解析命令行参数 ----
    # argparse 把 "--lr 3e-5" 这样的命令行参数解析成 args.lr = 3e-5（参数名里的 - 会变成 _）。
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", nargs="+", default=[], help="one or more val JSONL files (metrics are reported per file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32, help="max decisions per micro-batch")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="max padded tokens per micro-batch and view (two views are run per step)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides --epochs when > 0")
    ap.add_argument("--lr", type=float, default=3e-5, help="backbone learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--rps-weight", type=float, default=0.5)
    ap.add_argument("--consistency-weight", type=float, default=0.5)
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also train the 197M-param token embedding matrix (frozen by default to save memory)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500, help="optimizer steps")
    ap.add_argument("--save-every", type=int, default=500, help="optimizer steps")
    ap.add_argument("--val-limit", type=int, default=0, help="evaluate on a fixed random N decisions of each val file")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--hub-repo", default="", help="push checkpoints to this HF model repo (private)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # ---- 第 2 步：设备、是否续训、模型与 tokenizer ----
    # 固定随机种子，让决策头的随机初始化等结果可复现。
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_dir, best_dir = os.path.join(args.out, "last"), os.path.join(args.out, "best")
    # 在 last 和备份 last.prev 中找最新的完整 checkpoint；找到就续训，模型和 tokenizer 都从那里加载。
    resume_dir = find_resumable(last_dir)
    resume = resume_dir is not None
    src = resume_dir if resume else args.model

    tokenizer = load_tokenizer(src)
    encoding_cfg = {"max_len": args.max_len}
    encoder = DecisionEncoder(tokenizer, **encoding_cfg)
    # .to(device)：把模型所有参数搬到 GPU 上。
    model = DecisionModel.from_pretrained(src).to(device)
    # 冻结词嵌入矩阵（原理见模块说明第 8 条）。requires_grad_(False) 表示这些参数不需要梯度、不被更新。
    if not args.train_embeddings:
        model.backbone.get_input_embeddings().requires_grad_(False)
    # 梯度检查点（原理见模块说明第 7 条）。
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()

    # ---- 第 3 步：读数据 ----
    train = read_jsonl(args.train)
    # 验证集按文件名区分，例如 {"val": [...], "val_typed": [...]}，日志里分别报告。
    # removesuffix(".jsonl")：去掉字符串末尾的 ".jsonl"（Python 3.9+ 的字符串方法）。
    vals = {os.path.basename(p).removesuffix(".jsonl"): read_jsonl(p) for p in args.val}
    # 验证集太大时固定随机抽 N 条（种子固定，每次评估用的是同一批题，结果可比）。
    if args.val_limit:
        vals = {k: random.Random(args.seed).sample(v, min(args.val_limit, len(v))) for k, v in vals.items()}

    print(f"[train] tokenizing {len(train)} decisions for length bucketing", flush=True)
    # 预先算出每道题的真实 token 长度，供按 token 预算分批使用（8.5 万道题约需 1 分钟）。
    lengths = [len(encoder.encode(d).input_ids) for d in train]

    def epoch_batches(epoch: int) -> list[list[int]]:
        """第 epoch 轮的分批结果；对同一个 epoch 是确定的，续训时可以据此跳过已训练的批。"""
        return token_budget_batches(lengths, args.max_tokens, args.batch_size, seed=args.seed * 1000 + epoch)

    # ---- 第 4 步：优化器、学习率调度、GradScaler ----
    # 一个优化器 step = grad_accum 个 micro-batch（梯度累积，见模块说明第 4 条）。
    steps_per_epoch = math.ceil(len(epoch_batches(0)) / args.grad_accum)
    # 指定了 --max-steps 就用它，否则按 epoch 数换算总步数（`a or b`：a 为 0 时取 b）。
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    # 两组学习率：主干是预训练好的，用小学习率微调，避免破坏已有知识；
    # 决策头是随机初始化的，需要较大学习率才能尽快学起来。
    head_params = list(model.head.parameters())
    # 只把需要梯度的参数交给优化器（被冻结的 embedding 不在其中，也就不占 Adam 状态的显存）。
    body_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        [{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    warmup = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        """学习率倍率（乘在各组的基础学习率上）：前 warmup 步线性升温，之后按余弦曲线衰减到 0。

        例如 total_steps=1000，warmup=60：
            step=0   → 1/60 ≈ 0.017
            step=59  → 1.0
            step=530 → 余弦走了一半，0.5
            step=1000 → 0.0
        """
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))

    # LambdaLR：每次 sched.step() 后，学习率 = 基础学习率 × lr_lambda(当前步数)。
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    # 混合精度（见模块说明第 6 条）：T4 用 fp16，需要 GradScaler；bf16 或 CPU 时 GradScaler 处于禁用状态。
    # 新版 torch 用 torch.amp.GradScaler，旧版（如本地测试用的 2.2）用 torch.cuda.amp.GradScaler。
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    # 训练进度（全部保存在 trainer_state.pt 中，续训时恢复）：
    #   step        已完成的优化器步数（参数更新了多少次）
    #   micro_total 已完成的 micro-batch 总数（每满 grad_accum 个做一次更新）
    #   epoch       当前所在 epoch（第几遍过训练集）
    #   epoch_micro 当前 epoch 内已完成的 micro-batch 数（续训时跳过这么多批）
    #   best        目前最好的验证集平均准确率（初始 -1，第一次评估一定会刷新）
    step, micro_total, epoch, epoch_micro, best = 0, 0, 0, 0, -1.0
    if resume:
        # weights_only=False：trainer_state.pt 里不只是张量，还有普通的 Python 数字，需要完整反序列化。
        st = torch.load(os.path.join(resume_dir, STATE_FILE), map_location="cpu", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, micro_total, best = st["step"], st["micro_total"], st["best"]
        epoch, epoch_micro = st["epoch"], st["epoch_micro"]
        print(f"[train] resumed from {resume_dir} at step {step}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    # 保存本次运行的参数，以后查看结果时知道是用什么配置训练的。
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    # "a" 追加模式：续训时日志接在原来的后面，不会覆盖之前的记录。
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(rec: dict) -> None:
        """一条日志同时打印到 stdout 并追加写入 {out}/log.jsonl（每行一个 JSON）。"""
        rec = {"step": step, "time": round(time.time()), **rec}
        # flush=True：立刻输出，不在缓冲区里攒着，后台运行时才能实时看到日志。
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    hub = None
    if args.hub_repo:
        # 只有用到时才导入 huggingface_hub，不用 Hub 的人不需要关心它。
        from huggingface_hub import HfApi

        hub = HfApi()
        hub.create_repo(args.hub_repo, private=True, exist_ok=True)

    def save(path: str, with_state: bool) -> None:
        """保存 checkpoint。

        with_state=True 时额外保存优化器等状态（用于 last，可续训，原因见模块说明第 10 条）；
        best 只存权重，因为它只用来推理，不需要续训，省空间（优化器状态比权重本身还大）。
        配置了 --hub-repo 时在后台线程上传到 HF Hub（run_as_future=True，不阻塞训练）。

        “原子”保存，防止写到一半被打断导致 checkpoint 损坏：
        1. 先把所有文件写进临时目录 ``{path}.tmp``；
        2. 全部写完后，把旧的 ``{path}`` 改名为 ``{path}.prev``（覆盖更早的备份）；
        3. 再把 ``{path}.tmp`` 改名为 ``{path}``。
        改名几乎是瞬间完成的，所以任何时刻被打断，``{path}`` 或 ``{path}.prev`` 至少有一个是完整的。
        续训时 find_resumable 会自动挑出最新的完整那一个。
        best 只用于推理，不需要保留备份，替换成功后删掉 .prev 以节省空间。
        """
        tmp, prev = path + ".tmp", path + ".prev"
        if os.path.exists(tmp):  # 上次写到一半留下的残骸
            shutil.rmtree(tmp)
        model.save(tmp, tokenizer, extra={"encoding": encoding_cfg, "base_model": args.model, "step": step})
        if with_state:
            torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "micro_total": micro_total, "epoch": epoch, "epoch_micro": epoch_micro,
                        "best": best}, os.path.join(tmp, STATE_FILE))
        if os.path.exists(path):
            if os.path.exists(prev):
                shutil.rmtree(prev)
            os.rename(path, prev)
        os.rename(tmp, path)
        if not with_state and os.path.exists(prev):
            shutil.rmtree(prev)
        if hub:
            hub.upload_folder(repo_id=args.hub_repo, folder_path=path, path_in_repo=os.path.basename(path),
                              run_as_future=True)

    def run_eval() -> None:
        """在所有验证集上评估；若平均准确率创新高，就保存为 best。"""
        # nonlocal：声明 best 指的是外层 main 函数里的变量，这样赋值才会修改它，而不是新建局部变量。
        nonlocal best
        if not vals:
            return
        scores = {}
        for name, ds in vals.items():
            m = evaluate(model, encoder, ds, device, args.eval_batch_size)
            scores[name] = m
            log({"eval": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        # 用各验证集准确率的平均值来选模型（公开数据 val 与 typed-decisions val 各占一半权重）。
        score = sum(m["accuracy"] for m in scores.values()) / len(scores)
        if score > best:
            best = score
            save(best_dir, with_state=False)
            log({"new_best": round(best, 4)})

    trainset = TrainSet(train, encoder, args.seed)
    collate_fn = make_collate(tokenizer.pad_token_id)

    # ---- 第 5 步：训练主循环 ----
    # 外层 while：每次循环跑一个 epoch（续训时从中断的 epoch 中间开始），直到达到总步数。
    # 内层 for：一次取一个 micro-batch。一次完整的训练 step 内部发生的事情：
    #   第 1 步：把这一批两个视图的数据搬到 GPU；
    #   第 2 步：在 autocast（混合精度）下做前向，得到两个视图的 logits，并用 unpermute 还原成原始选项顺序；
    #   第 3 步：在 float32 下算损失 = 软标签 CE + RPS（只算 score 题）+ 一致性 KL（只算 choice/noul 题）；
    #   第 4 步：损失除以 grad_accum，经 GradScaler 放大后 backward，梯度累加到参数的 .grad 上；
    #   第 5 步：累积满 grad_accum 个 micro-batch 后：unscale 还原梯度 → 梯度裁剪 → 优化器更新参数
    #           → 更新 GradScaler 倍数 → 清空梯度 → 学习率调度前进一步 → step 加 1；
    #   第 6 步：按需打印日志（每 20 步）、评估（每 eval_every 步）、保存（每 save_every 步）。
    # model.train()：切换到训练模式，打开 Dropout。
    model.train()
    # running 累计最近若干个 micro-batch 的各项损失，用于每 20 步打印一次平均值。
    t0, running = time.time(), {"loss": 0.0, "ce": 0.0, "rps": 0.0, "cons": 0.0, "n": 0}
    while step < total_steps:
        trainset.epoch = epoch
        batches = epoch_batches(epoch)[epoch_micro:]  # 续训时跳过本 epoch 已训练过的批（分批是确定的）
        # batch_sampler=batches：直接告诉 DataLoader 每一批取哪些下标（我们自己按 token 预算分好了）。
        loader = DataLoader(trainset, batch_sampler=batches, collate_fn=collate_fn,
                            num_workers=args.num_workers, persistent_workers=False)
        for batch in loader:
            # 第 1 步：数据搬到 GPU。
            a = {k: v.to(device) for k, v in batch["a"].items()}
            b = {k: v.to(device) for k, v in batch["b"].items()}
            target = batch["target"].to(device)
            is_score = batch["is_score"].to(device)
            mask = a["option_mask"]
            # 第 2 步：前向在 autocast 下进行（混合精度）；两个视图的 logits 都先 unpermute 回原始选项顺序。
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits_a = unpermute(model(**a), batch["perm_a"].to(device), mask)
                # 只有存在 choice/noul 题时才需要跑视图 B（score 题不打乱、不算一致性），省一次前向。
                use_b = args.consistency_weight > 0 and bool((~is_score).any())
                logits_b = unpermute(model(**b), batch["perm_b"].to(device), mask) if use_b else None
            # 第 3 步：算损失。损失在 autocast 之外用 float32 计算，数值更稳定。
            # 交叉熵用平滑后的 target；RPS 用原始 target，且只对 score 题生效（乘以 is_score，非 score 题乘 0）。
            tgt = smooth(target, mask, args.label_smoothing)
            ce = soft_ce(logits_a, tgt, mask)
            r = rps(logits_a, target, mask) * is_score
            loss = ce + args.rps_weight * r
            cons = torch.zeros_like(ce)
            if logits_b is not None:
                # 一致性损失只对 choice/noul 题生效（score 题两个视图本来就一样）。
                cons = symmetric_kl(logits_a, logits_b, mask) * (~is_score)
                loss = loss + args.consistency_weight * cons
            # 第 4 步：梯度累积：每个 micro-batch 的损失除以 grad_accum，累积 grad_accum 次后等价于一个大 batch。
            # .mean()：把每道题的损失 [B] 平均成一个数，backward 需要一个标量。
            loss = loss.mean() / args.grad_accum
            # scaler.scale(loss)：fp16 时把损失乘以放大倍数再 backward；禁用时原样返回。
            scaler.scale(loss).backward()

            # 记录日志用的损失（.item() 把只有一个数的张量取成 Python 数字）。
            running["loss"] += loss.item() * args.grad_accum
            running["ce"] += ce.mean().item()
            running["rps"] += r.mean().item()
            running["cons"] += cons.mean().item()
            running["n"] += 1
            micro_total += 1
            epoch_micro += 1
            # 还没累积满 grad_accum 个 micro-batch，继续累积梯度，不更新参数。
            if micro_total % args.grad_accum:
                continue
            # 第 5 步：更新参数。先把梯度从 GradScaler 的放大倍数还原，再做梯度裁剪（范数上限 1.0）。
            # 必须先 unscale 再裁剪，否则裁剪的是被放大过的梯度，阈值就没有意义了。
            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], 1.0)
            # 若出现 inf/NaN 梯度，scaler.step 会跳过本次更新，并在 update 时调小放大倍数。
            scaler.step(optim)
            scaler.update()
            # 清空梯度，为下一次累积做准备（set_to_none=True 直接释放梯度张量，比填 0 更省显存）。
            optim.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            # 第 6 步：日志 / 评估 / 保存。
            if step % 20 == 0 or step == 1:
                n = running.pop("n")
                # gnorm 是裁剪前的梯度范数，持续很大或突然飙升说明训练可能不稳定。
                log({"epoch": round(step / steps_per_epoch, 3), "lr": sched.get_last_lr()[0],
                     "gnorm": round(float(gnorm), 3), "sec_per_step": round((time.time() - t0) / 20, 2),
                     **{k: round(v / n, 4) for k, v in running.items()}})
                running = {"loss": 0.0, "ce": 0.0, "rps": 0.0, "cons": 0.0, "n": 0}
                t0 = time.time()
            if step % args.eval_every == 0:
                run_eval()
            if step % args.save_every == 0:
                save(last_dir, with_state=True)
            if step >= total_steps:
                break
        else:
            # for-else 语法：for 循环正常跑完（没有被 break 打断）时才执行 else。
            # 这里表示本 epoch 的数据全部训练完，进入下一个 epoch，epoch 内计数清零。
            epoch, epoch_micro = epoch + 1, 0

    # ---- 第 6 步：收尾 ----
    # 训练结束：如果最后一步恰好已经评估 / 保存过，就不重复做。
    if step % args.eval_every:
        run_eval()
    if step % args.save_every:
        save(last_dir, with_state=True)
    log({"done": True, "best": round(best, 4)})


if __name__ == "__main__":
    main()
