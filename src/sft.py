import argparse
import json
import math
import os
import random
import time

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core import build_sft_batch, response_logprobs_from_batch
from src.data import PROMPT_TEMPLATE, format_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(split, n, seed):
    """UltraChat -> single-turn (first user msg, first assistant msg) pairs."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=seed)
    pairs = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        instr, resp = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not instr or not resp:
            continue
        pairs.append({"instruction": instr, "response": resp})
        if len(pairs) == n:
            break
    return pairs


def make_collate(tokenizer, max_length):
    def collate(items):
        return build_sft_batch(
            tokenizer,
            [format_prompt(x["instruction"]) for x in items],
            [x["response"] for x in items],
            max_length,
        )
    return collate


def log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


@torch.no_grad()
def evaluate(model, tokenizer, pairs, max_length, bs, device):
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(pairs), bs):
        batch = make_collate(tokenizer, max_length)(pairs[i : i + bs])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logps, ntok = response_logprobs_from_batch(model, batch)
        total_nll += -logps.sum().item()
        total_tok += ntok.sum().item()
    model.train()
    return total_nll / max(1, total_tok)


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out_dir", default="outputs/sft")
    p.add_argument("--ckpt_dir", default="outputs/checkpoints/sft")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--micro_bs", type=int, default=2)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0")
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    tok = AutoTokenizer.from_pretrained(args.model)
    train_pairs = load_pairs("train_sft", args.n_train, args.seed)
    val_pairs = load_pairs("test_sft", args.n_val, args.seed)
    print(f"train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    loader = DataLoader(
        train_pairs, batch_size=args.micro_bs, shuffle=True,
        collate_fn=make_collate(tok, args.max_length),
        generator=torch.Generator().manual_seed(args.seed),
    )
    steps_per_epoch = len(loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_ratio * total_steps))

    decay = [q for q in model.parameters() if q.requires_grad and q.ndim >= 2]
    no_decay = [q for q in model.parameters() if q.requires_grad and q.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    cfg = dict(vars(args))
    cfg.update(
        torch=torch.__version__, transformers=transformers.__version__,
        datasets=datasets.__version__, prompt_template=PROMPT_TEMPLATE,
        n_train_actual=len(train_pairs), n_val_actual=len(val_pairs),
        optimizer_steps=total_steps, effective_batch=args.micro_bs * args.accum,
        loss_reduction="mean over all unmasked response tokens in the micro-batch",
        precision="fp32 weights + fp16 autocast + GradScaler",
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"optimizer steps: {total_steps} | warmup: {warmup}")

    best_val, step, micro, win_loss, win_n = float("inf"), 0, 0, 0.0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logps, ntok = response_logprobs_from_batch(model, batch)
            loss = -logps.sum() / ntok.sum().clamp(min=1)
            scaler.scale(loss / args.accum).backward()
            win_loss += loss.item()
            win_n += 1
            micro += 1
            if micro % args.accum != 0:
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0:
                rec = {"type": "train", "step": step, "loss": win_loss / win_n,
                       "lr": sched.get_last_lr()[0], "grad_norm": gnorm,
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                       "elapsed_min": (time.time() - t0) / 60}
                log(metrics_path, rec)
                print(f"step {step}/{total_steps} | loss {rec['loss']:.4f} | lr {rec['lr']:.2e} "
                      f"| gnorm {gnorm:.2f} | peak {rec['peak_mem_gb']:.1f}GB | {rec['elapsed_min']:.1f}min")
                win_loss, win_n = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(model, tok, val_pairs, args.max_length, 4, device)
                log(metrics_path, {"type": "val", "step": step, "val_loss": val})
                print(f"   >>> step {step} | val loss {val:.4f}")
                if val < best_val:
                    best_val = val
                    save_model(model, tok, os.path.join(args.ckpt_dir, "best"))
                    print("   >>> new best, saved")

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    save_model(model, tok, os.path.join(args.ckpt_dir, "final"))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"best_val_loss": best_val, "steps": step,
                   "minutes": (time.time() - t0) / 60,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}, f, indent=2)
    print("DONE. best val loss:", best_val)


if __name__ == "__main__":
    main()
