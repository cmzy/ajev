"""大模型（Gemma 4 12B 等）LoRA 微调：让模型在“读字母选项的打分”这件事上学会我们的题型、业务约定和概率。

    python -m ajev.lm.train --model google/gemma-4-12B-it --train data/lm1/train.jsonl \\
        --val data/lm1/val.jsonl data/lm1/val_typed.jsonl --out runs/gemma_lora1

在 AJev 中的位置：数据（ajev/lm/subset.py 挑出的约 3 万道题）→ **本模块训练 LoRA 适配器** →
校准（``python -m ajev.calibrate --lm-model ... --lm-adapter ...``）→ 评测（``--predictor lm --lm-adapter ...``）。

和 mmBERT 训练（ajev/train/train.py）的关系：**整体结构、损失函数、分批、checkpoint 全部复用**，
只有“模型怎么给选项打分”不同：
- mmBERT：每个选项前放一个 <mask> 标记位，决策头给每个标记位打分；
- 大模型：把题目写成带 A/B/C… 字母选项的对话，读模型“下一个 token 是哪个字母”的打分（见 ajev/lm/predictor.py）。
所以训练时打分的方式和推理时完全是同一个函数（``letter_logits``），不会出现训练和推理不一致。

给初学者的几个新概念：

1. **LoRA（Low-Rank Adaptation，低秩适配）**
   120 亿个参数全部训练，光 Adam 状态就要上百 GB 显存，不现实。LoRA 的做法：原模型参数全部冻结不动，
   只在一部分线性层旁边加一条“小支路”：输入先乘一个 d×r 的矩阵 A（降到 r 维），再乘一个 r×d 的矩阵 B（升回 d 维），
   结果加到原输出上。r 很小（这里是 32），所以每层只多出 2×d×r 个参数——全模型加起来约 6 千万，
   只占 0.5%。训练只更新这些小矩阵；B 初始化为 0，所以刚开始时模型和原来一模一样，之后慢慢学出“调整量”。
   ``lora_alpha`` 是支路输出的缩放系数（实际缩放 alpha / r = 2）。

2. **为什么只给语言模型部分加 LoRA**
   Gemma 4 是图文统一模型，里面还有处理图像、音频的部分。我们只训练文本决策，给那些部分加 LoRA 没有意义，
   还会浪费显存。``lora_targets`` 会挑出语言模型里的 7 种线性层：注意力的 q/k/v/o 投影和 MLP 的 gate/up/down。

3. **冻结主体 + 梯度检查点的坑**
   梯度检查点（见 mmBERT 训练脚本的说明）在反向时会重新计算前向。如果输入的 embedding 不需要梯度
   （主体冻结时正是这样），某些实现下这段重算会“断开”，导致 LoRA 层拿不到梯度、模型根本没在学。
   我们用两个办法防住：``use_reentrant=False`` 的检查点实现 + ``enable_input_require_grads()``；
   并且在第一次反向后**断言** LoRA 参数确实有非零梯度，不满足就立刻报错，而不是白训几个小时。

4. **保存的只是适配器**
   checkpoint 里只有 LoRA 的小矩阵（几百 MB）和它的优化器状态，基座模型不用存（推理时从 HF 下载原版，再把适配器加上去）。

其余（软标签 CE、打分题 RPS、打乱选项一致性、每道题权重相同、按 token 预算分批、原子保存与断点续训）
与 mmBERT 训练完全相同，原理见 ajev/train/train.py 的说明。
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
from torch.utils.data import DataLoader, Dataset

from ajev import metrics
from ajev.lm.predictor import (
    LM_CONFIG,
    LMPredictor,
    left_pad,
    letter_logits,
    letter_token_table,
    load_base_model,
    load_tokenizer,
    prompt_ids,
)
from ajev.lm.prompt import MAX_OPTIONS
from ajev.schema import Decision, read_jsonl
from ajev.train.losses import rps, smooth, soft_ce, symmetric_kl, unpermute
from ajev.train.train import META_FILE, STATE_FILE, find_resumable, token_budget_batches

# 一个完整的 LoRA checkpoint 必须包含的文件（配置文件放第一个，完整性检查从它读取 step）。
NEEDED = [LM_CONFIG, "adapter_model.safetensors", "adapter_config.json", STATE_FILE]
LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_targets(model) -> str | list[str]:
    """要加 LoRA 的线性层。模型里有 language_model 子模块时（图文统一模型），只选它下面的层。

    返回 PEFT 认识的写法：正则字符串（匹配完整模块名）或层名列表（按名字结尾匹配）。
    """
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in LINEAR_NAMES]
    if not names:
        raise ValueError("no q/k/v/o/gate/up/down projections found for LoRA")
    if any("language_model" in n for n in names):
        return r".*language_model.*\.(" + "|".join(LINEAR_NAMES) + r")$"
    return list(LINEAR_NAMES)


class LMTrainSet(Dataset):
    """每道题产出视图 A（选项随机打乱后写成提示词）；choice / noul 题再产出一个不同打乱顺序的视图 B。

    和 mmBERT 的 TrainSet 一样：随机种子由 (seed, epoch, 题目下标) 决定，续训后视图完全一致。
    返回 (原题, 视图 A 的 token id, perm_a, 视图 B 的 token id 或 None, perm_b 或 None)。
    perm 的含义：视图里第 j 个选项 = 原题的第 perm[j] 个选项（用于把打分放回原始顺序，见 losses.unpermute）。
    """

    def __init__(self, decisions: list[Decision], tok, max_state_tokens: int, seed: int, two_views: bool) -> None:
        self.decisions, self.tok, self.max_state_tokens = decisions, tok, max_state_tokens
        self.seed, self.two_views, self.epoch = seed, two_views, 0

    def __len__(self) -> int:
        return len(self.decisions)

    def _view(self, d: Decision, rng: random.Random) -> tuple[list[int], list[int]]:
        perm = list(range(len(d.options)))
        if d.type != "score":  # score 的等级有顺序，不打乱
            rng.shuffle(perm)
        view = Decision(**{**d.__dict__, "options": [d.options[i] for i in perm]})
        return prompt_ids(self.tok, view, self.max_state_tokens), perm

    def __getitem__(self, i: int):
        d = self.decisions[i]
        rng = random.Random(f"{self.seed}/{self.epoch}/{i}")
        ids_a, perm_a = self._view(d, rng)
        if not self.two_views or d.type == "score":
            return d, ids_a, perm_a, None, None
        ids_b, perm_b = self._view(d, rng)
        return d, ids_a, perm_a, ids_b, perm_b


def make_collate(pad_id: int):
    """拼 batch：两个视图各自左侧补齐；视图 B 只包含有视图 B 的题（b_index 记录它们在本批中的位置）。"""
    from ajev.model.batching import pad_targets

    def perm_tensor(perms, width):
        return torch.tensor([p + list(range(len(p), width)) for p in perms])

    def fn(items):
        ds, ids_a, perm_a, ids_b, perm_b = zip(*items)
        k = max(len(d.options) for d in ds)
        a_ids, a_mask = left_pad(list(ids_a), pad_id)
        option_mask = torch.tensor([[j < len(d.options) for j in range(k)] for d in ds])
        b_index = [i for i, x in enumerate(ids_b) if x is not None]
        b = None
        if b_index:
            kb = max(len(ds[i].options) for i in b_index)
            b_ids, b_mask = left_pad([ids_b[i] for i in b_index], pad_id)
            b = {"ids": b_ids, "mask": b_mask, "k": kb, "perm": perm_tensor([perm_b[i] for i in b_index], kb),
                 "option_mask": torch.tensor([[j < len(ds[i].options) for j in range(kb)] for i in b_index])}
        return {"ids": a_ids, "mask": a_mask, "k": k, "perm_a": perm_tensor(list(perm_a), k), "option_mask": option_mask,
                "b": b, "b_index": torch.tensor(b_index, dtype=torch.long),
                "target": pad_targets([d.target for d in ds], k), "is_score": torch.tensor([d.type == "score" for d in ds])}

    return fn


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides --epochs when > 0")
    ap.add_argument("--batch-size", type=int, default=16, help="max decisions per micro-batch")
    ap.add_argument("--max-tokens", type=int, default=12288, help="max padded tokens per micro-batch and view")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-state-tokens", type=int, default=1500)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--rps-weight", type=float, default=0.5)
    ap.add_argument("--consistency-weight", type=float, default=0.5)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000000)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-limit", type=int, default=1500)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", help="resume from this checkpoint directory instead of {out}/last "
                                          "(e.g. an earlier, healthy checkpoint)")
    ap.add_argument("--skip-gnorm", type=float, default=100.0,
                    help="skip the update when the pre-clip gradient norm exceeds this (0 = never skip)")
    args = ap.parse_args(argv)

    # ---- 第 1 步：设备、模型、LoRA（或从 last 续训）----
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    last_dir, best_dir = os.path.join(args.out, "last"), os.path.join(args.out, "best")
    # --resume-from：从指定的（例如更早、更健康的）checkpoint 续训，而不是 {out}/last。
    resume_dir = args.resume_from or find_resumable(last_dir, NEEDED)
    tok = load_tokenizer(args.model)
    model = load_base_model(args.model, dtype)
    model.config.use_cache = False  # 训练时不需要生成用的 KV 缓存
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    from peft import LoraConfig, PeftModel, get_peft_model

    if resume_dir:
        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=args.lora_dropout, target_modules=lora_targets(model)))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train, n_all = sum(p.numel() for p in params), sum(p.numel() for p in model.parameters())
    print(f"[lm-train] trainable {n_train / 1e6:.1f}M / {n_all / 1e9:.2f}B params ({100 * n_train / n_all:.2f}%)",
          flush=True)
    table, valid = letter_token_table(tok)

    # ---- 第 2 步：数据与按 token 预算分批 ----
    train = [d for d in read_jsonl(args.train) if len(d.options) <= MAX_OPTIONS]
    vals = {os.path.basename(p).removesuffix(".jsonl"): read_jsonl(p) for p in args.val}
    if args.val_limit:
        vals = {k: random.Random(args.seed).sample(v, min(args.val_limit, len(v))) for k, v in vals.items()}
    print(f"[lm-train] tokenizing {len(train)} prompts for length bucketing", flush=True)
    lengths = [len(prompt_ids(tok, d, args.max_state_tokens)) for d in train]
    cache: dict[int, list[list[int]]] = {}

    def epoch_batches(epoch: int) -> list[list[int]]:
        if epoch not in cache:
            cache[epoch] = token_budget_batches(lengths, args.max_tokens, args.batch_size, seed=args.seed * 1000 + epoch)
        return cache[epoch]

    full, frac = int(args.epochs), args.epochs - int(args.epochs)
    total_micro = sum(len(epoch_batches(e)) for e in range(full)) + (int(len(epoch_batches(full)) * frac) if frac else 0)
    total_steps = args.max_steps or max(1, total_micro // args.grad_accum)

    # ---- 第 3 步：优化器与学习率调度（与 mmBERT 相同的 warmup + 余弦）----
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    step, micro_total, epoch, epoch_micro, best = 0, 0, 0, 0, -1.0
    if resume_dir:
        st = torch.load(os.path.join(resume_dir, STATE_FILE), map_location="cpu", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        step, micro_total, best = st["step"], st["micro_total"], st["best"]
        epoch, epoch_micro = st["epoch"], st["epoch_micro"]
        print(f"[lm-train] resumed from {resume_dir} at step {step}", flush=True)
        # 续训时允许修改峰值学习率：调度器的状态里存着旧的“基础学习率”，这里换成 --lr，
        # 当前学习率按新基础学习率 × 当前步的调度系数重新计算。
        # （第一次正式训练在 1e-4 下出现梯度尖峰、loss 抬升，就是靠这个降到 3e-5 后从健康的 checkpoint 接着训。）
        if abs(sched.base_lrs[0] - args.lr) > 1e-12:
            print(f"[lm-train] peak lr {sched.base_lrs[0]:.2e} -> {args.lr:.2e}", flush=True)
            sched.base_lrs = [args.lr] * len(sched.base_lrs)
            for g in optim.param_groups:
                g["initial_lr"], g["lr"] = args.lr, args.lr * lr_lambda(step)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(rec: dict) -> None:
        rec = {"step": step, "time": round(time.time()), **rec}
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    def save(path: str, with_state: bool) -> None:
        """原子保存（先写 .tmp，再把旧目录改名为 .prev、.tmp 改名为正式目录），原理同 mmBERT 训练。"""
        tmp, prev = path + ".tmp", path + ".prev"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        model.save_pretrained(tmp)  # PEFT 模型只保存适配器：adapter_model.safetensors + adapter_config.json
        with open(os.path.join(tmp, LM_CONFIG), "w") as f:
            json.dump({"base_model": args.model, "max_state_tokens": args.max_state_tokens, "step": step}, f, indent=2)
        if with_state:
            torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(), "step": step,
                        "micro_total": micro_total, "epoch": epoch, "epoch_micro": epoch_micro, "best": best},
                       os.path.join(tmp, STATE_FILE))
            sizes = {f: os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)}
            with open(os.path.join(tmp, META_FILE), "w") as f:
                json.dump({"step": step, "sizes": sizes}, f)
        if os.path.exists(path):
            if os.path.exists(prev):
                shutil.rmtree(prev)
            os.rename(path, prev)
        os.rename(tmp, path)
        if not with_state and os.path.exists(prev):
            shutil.rmtree(prev)

    predictor = LMPredictor(args.model, model=model, tok=tok, device=str(device),
                            max_state_tokens=args.max_state_tokens, temperatures={})

    def run_eval() -> None:
        nonlocal best
        if not vals:
            return
        scores = {}
        for name, ds in vals.items():
            probs = [torch.softmax(torch.tensor(lg), -1).tolist() for lg in predictor.predict_logits(ds)]
            m = metrics.compute(ds, probs)
            scores[name] = m
            log({"eval": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        score = sum(m["accuracy"] for m in scores.values()) / len(scores)
        if score > best:
            best = score
            save(best_dir, with_state=False)
            log({"new_best": round(best, 4)})
        model.train()

    # ---- 第 4 步：训练主循环（结构与 mmBERT 训练相同，见 ajev/train/train.py）----
    trainset = LMTrainSet(train, tok, args.max_state_tokens, args.seed, two_views=args.consistency_weight > 0)
    collate_fn = make_collate(tok.pad_token_id if tok.pad_token_id is not None else 0)
    ref_n = args.batch_size * args.grad_accum
    model.train()

    def fresh() -> dict:
        return {"loss": 0.0, "ce": 0.0, "n": 0, "rps": 0.0, "n_score": 0, "cons": 0.0, "n_cons": 0, "steps": 0}

    t0, running, window_n, checked_grads, skipped = time.time(), fresh(), 0, False, 0
    while step < total_steps:
        trainset.epoch = epoch
        loader = DataLoader(trainset, batch_sampler=epoch_batches(epoch)[epoch_micro:], collate_fn=collate_fn,
                            num_workers=args.num_workers)
        for batch in loader:
            mask = batch["option_mask"].to(device)
            target, is_score = batch["target"].to(device), batch["is_score"].to(device)
            # 第 1 步：视图 A 前向，字母打分放回原始选项顺序。
            raw = letter_logits(model, batch["ids"].to(device), batch["mask"].to(device), table, valid, batch["k"])
            logits_a = unpermute(raw.masked_fill(~mask, float("-inf")), batch["perm_a"].to(device), mask)
            # 第 2 步：每道题的损失 = 软标签 CE + RPS（score 题）+ 一致性 KL（有视图 B 的题）。
            ce = soft_ce(logits_a, smooth(target, mask, args.label_smoothing), mask)
            r = rps(logits_a, target, mask) * is_score
            loss = ce + args.rps_weight * r
            cons = torch.zeros_like(ce)
            if batch["b"] is not None and args.consistency_weight > 0:
                b = batch["b"]
                bmask = b["option_mask"].to(device)
                bi = batch["b_index"].to(device)
                raw_b = letter_logits(model, b["ids"].to(device), b["mask"].to(device), table, valid, b["k"])
                logits_b = unpermute(raw_b.masked_fill(~bmask, float("-inf")), b["perm"].to(device), bmask)
                cons = cons.index_copy(0, bi, symmetric_kl(logits_a[bi, : b["k"]], logits_b, bmask))
                loss = loss + args.consistency_weight * cons
            # 第 3 步：损失求和 / ref_n 后反向；累计窗口题数（每道题权重相同，原理见 mmBERT 训练）。
            (loss.sum() / ref_n).backward()
            window_n += loss.size(0)
            if not checked_grads:
                # 第一次反向后确认 LoRA 参数真的拿到了梯度（防止“冻结主体 + 梯度检查点”导致白训）。
                if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
                    raise RuntimeError("LoRA parameters received no gradient — check gradient checkpointing setup")
                checked_grads = True
            n_score = int(is_score.sum())
            running["loss"] += loss.sum().item(); running["ce"] += ce.sum().item(); running["n"] += loss.size(0)
            running["rps"] += r.sum().item(); running["n_score"] += n_score
            running["cons"] += cons.sum().item(); running["n_cons"] += int(batch["b_index"].numel())
            micro_total += 1
            epoch_micro += 1
            if micro_total % args.grad_accum:
                continue
            # 第 4 步：梯度归一化到“每道题平均” → 裁剪 → 更新。
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(ref_n / window_n)
            window_n = 0
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            # 梯度尖峰保护：裁剪前的梯度范数异常大，说明这一批把模型推向了损失曲面很陡的地方。
            # 裁剪虽然限制了步长，但方向往往不可靠，连续几次就可能把模型推坏（第一次训练就是这样）。
            # 这里直接放弃这一步的更新（学习率调度照常前进），并计数记入日志。
            if args.skip_gnorm and float(gnorm) > args.skip_gnorm:
                skipped += 1
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            running["steps"] += 1
            if step % 10 == 0 or step == 1:
                rn = running
                log({"epoch": round(epoch + epoch_micro / len(epoch_batches(epoch)), 3), "lr": sched.get_last_lr()[0],
                     "gnorm": round(float(gnorm), 3), "sec_per_step": round((time.time() - t0) / rn["steps"], 2),
                     "loss": round(rn["loss"] / max(1, rn["n"]), 4), "ce": round(rn["ce"] / max(1, rn["n"]), 4),
                     "rps": round(rn["rps"] / max(1, rn["n_score"]), 4),
                     "cons": round(rn["cons"] / max(1, rn["n_cons"]), 4),
                     "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else 0,
                     "skipped": skipped})
                running, t0 = fresh(), time.time()
            if step % args.eval_every == 0:
                run_eval()
            if step % args.save_every == 0:
                save(last_dir, with_state=True)
            if step >= total_steps:
                break
        else:
            epoch, epoch_micro = epoch + 1, 0

    if step % args.eval_every:
        run_eval()
    if step % args.save_every:
        save(last_dir, with_state=True)
    if not vals:  # 没有验证集时，把最后的适配器当作 best
        save(best_dir, with_state=False)
    log({"done": True, "best": round(best, 4)})


if __name__ == "__main__":
    main()
