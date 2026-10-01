"""Supervised training of the encoder decision model (soft labels + option-order consistency).

    python -m ajev.train.train --train data/build/train.jsonl --val data/build/val.jsonl --out runs/sft1

Objective per decision (view A = options in a random order):
    soft_ce(A, smoothed target)
  + rps_weight * RPS(A)                         for score questions
  + consistency_weight * symKL(A, B)            choice/noul: B = same decision, another option order

Checkpoints: ``{out}/last`` (resumable: weights + optimizer/scheduler/scaler/step) and
``{out}/best`` (best val accuracy). Re-running the same command resumes from ``last``.
``--hub-repo`` additionally uploads checkpoints to the Hugging Face Hub (needs HF_TOKEN), so a
reclaimed Colab VM loses nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from ajev import metrics
from ajev.model.batching import collate, pad_targets
from ajev.model.encoder import DecisionModel, load_tokenizer
from ajev.model.encoding import DecisionEncoder
from ajev.model.predictor import autocast_dtype, predict_logits, softmax
from ajev.schema import Decision, read_jsonl
from ajev.train.losses import rps, smooth, soft_ce, symmetric_kl, unpermute

STATE_FILE = "trainer_state.pt"


class TrainSet(Dataset):
    """Yields two independently option-shuffled views of each decision (score order is kept)."""

    def __init__(self, decisions: list[Decision], encoder: DecisionEncoder, seed: int) -> None:
        self.decisions = decisions
        self.encoder = encoder
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.decisions)

    def _view(self, d: Decision, rng: random.Random) -> tuple:
        perm = list(range(len(d.options)))
        if d.type != "score":
            rng.shuffle(perm)
        view = Decision(**{**d.__dict__, "options": [d.options[i] for i in perm]})
        return self.encoder.encode(view), perm

    def __getitem__(self, i: int):
        d = self.decisions[i]
        rng = random.Random(f"{self.seed}/{self.epoch}/{i}")
        enc_a, perm_a = self._view(d, rng)
        enc_b, perm_b = self._view(d, rng)
        return d, enc_a, perm_a, enc_b, perm_b


def make_collate(pad_id: int):
    def fn(items):
        ds, enc_a, perm_a, enc_b, perm_b = zip(*items)
        a, b = collate(list(enc_a), pad_id), collate(list(enc_b), pad_id)
        k = a["option_mask"].size(1)

        def perm_tensor(perms):
            # Padded slots map to themselves so -inf stays in padding after unpermute.
            return torch.tensor([p + list(range(len(p), k)) for p in perms])

        return {
            "a": a,
            "b": b,
            "perm_a": perm_tensor(perm_a),
            "perm_b": perm_tensor(perm_b),
            "target": pad_targets([d.target for d in ds], k),  # canonical option order
            "is_score": torch.tensor([d.type == "score" for d in ds]),
        }

    return fn


def bucketed_order(lengths: list[int], batch_size: int, seed: int, mega: int = 64) -> list[int]:
    """Shuffle, then sort inside mega-batches by length, then shuffle batch order."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    span = batch_size * mega
    for s in range(0, len(idx), span):
        chunk = sorted(idx[s : s + span], key=lengths.__getitem__)
        batches += [chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return [i for b in batches for i in b]


def evaluate(model, encoder, decisions, device, batch_size) -> dict:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", nargs="+", default=[], help="one or more val JSONL files (metrics are reported per file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_dir, best_dir = os.path.join(args.out, "last"), os.path.join(args.out, "best")
    resume = os.path.exists(os.path.join(last_dir, STATE_FILE))
    src = last_dir if resume else args.model

    tokenizer = load_tokenizer(src)
    encoding_cfg = {"max_len": args.max_len}
    encoder = DecisionEncoder(tokenizer, **encoding_cfg)
    model = DecisionModel.from_pretrained(src).to(device)
    if not args.train_embeddings:
        model.backbone.get_input_embeddings().requires_grad_(False)
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()

    train = read_jsonl(args.train)
    vals = {os.path.basename(p).removesuffix(".jsonl"): read_jsonl(p) for p in args.val}
    if args.val_limit:
        vals = {k: random.Random(args.seed).sample(v, min(args.val_limit, len(v))) for k, v in vals.items()}

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    head_params = list(model.head.parameters())
    body_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        [{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    warmup = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    step, micro_total, best = 0, 0, -1.0
    if resume:
        st = torch.load(os.path.join(last_dir, STATE_FILE), map_location="cpu", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, micro_total, best = st["step"], st["micro_total"], st["best"]
        print(f"[train] resumed from {last_dir} at step {step}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(rec: dict) -> None:
        rec = {"step": step, "time": round(time.time()), **rec}
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    hub = None
    if args.hub_repo:
        from huggingface_hub import HfApi

        hub = HfApi()
        hub.create_repo(args.hub_repo, private=True, exist_ok=True)

    def save(path: str, with_state: bool) -> None:
        model.save(path, tokenizer, extra={"encoding": encoding_cfg, "base_model": args.model, "step": step})
        if with_state:
            torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "micro_total": micro_total, "best": best}, os.path.join(path, STATE_FILE))
        if hub:
            hub.upload_folder(repo_id=args.hub_repo, folder_path=path, path_in_repo=os.path.basename(path),
                              run_as_future=True)

    def run_eval() -> None:
        nonlocal best
        if not vals:
            return
        scores = {}
        for name, ds in vals.items():
            m = evaluate(model, encoder, ds, device, args.eval_batch_size)
            scores[name] = m
            log({"eval": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        # Model selection on the mean accuracy over val files.
        score = sum(m["accuracy"] for m in scores.values()) / len(scores)
        if score > best:
            best = score
            save(best_dir, with_state=False)
            log({"new_best": round(best, 4)})

    trainset = TrainSet(train, encoder, args.seed)
    lengths = [len(d.state) + len(d.instructions) + sum(len(o.name) + len(o.desc) for o in d.options) for d in train]
    collate_fn = make_collate(tokenizer.pad_token_id)
    micro_per_epoch = math.ceil(len(train) / args.batch_size)

    model.train()
    t0, running = time.time(), {"loss": 0.0, "ce": 0.0, "rps": 0.0, "cons": 0.0, "n": 0}
    while step < total_steps:
        epoch, skip = divmod(micro_total, micro_per_epoch)
        trainset.epoch = epoch
        order = bucketed_order(lengths, args.batch_size, seed=args.seed * 1000 + epoch)
        order = order[skip * args.batch_size :]  # resume mid-epoch deterministically
        batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        loader = DataLoader(trainset, batch_sampler=batches, collate_fn=collate_fn,
                            num_workers=args.num_workers, persistent_workers=False)
        for batch in loader:
            a = {k: v.to(device) for k, v in batch["a"].items()}
            b = {k: v.to(device) for k, v in batch["b"].items()}
            target = batch["target"].to(device)
            is_score = batch["is_score"].to(device)
            mask = a["option_mask"]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits_a = unpermute(model(**a), batch["perm_a"].to(device), mask)
                use_b = args.consistency_weight > 0 and bool((~is_score).any())
                logits_b = unpermute(model(**b), batch["perm_b"].to(device), mask) if use_b else None
            tgt = smooth(target, mask, args.label_smoothing)
            ce = soft_ce(logits_a, tgt, mask)
            r = rps(logits_a, target, mask) * is_score
            loss = ce + args.rps_weight * r
            cons = torch.zeros_like(ce)
            if logits_b is not None:
                cons = symmetric_kl(logits_a, logits_b, mask) * (~is_score)
                loss = loss + args.consistency_weight * cons
            loss = loss.mean() / args.grad_accum
            scaler.scale(loss).backward()

            running["loss"] += loss.item() * args.grad_accum
            running["ce"] += ce.mean().item()
            running["rps"] += r.mean().item()
            running["cons"] += cons.mean().item()
            running["n"] += 1
            micro_total += 1
            if micro_total % args.grad_accum:
                continue
            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % 20 == 0 or step == 1:
                n = running.pop("n")
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

    if step % args.eval_every:
        run_eval()
    if step % args.save_every:
        save(last_dir, with_state=True)
    log({"done": True, "best": round(best, 4)})


if __name__ == "__main__":
    main()
