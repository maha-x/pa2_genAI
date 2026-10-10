import argparse
import json
import os
import random
import time

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core import build_dpo_batch, compute_dpo_loss_from_logps, response_logprobs_from_batch
from src.data import PROMPT_TEMPLATE, format_prompt
from src.pref_data import DATASETS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate(tokenizer, max_length, max_prompt_length):
    def collate(items):
        return build_dpo_batch(
            tokenizer,
            [format_prompt(x["prompt"]) for x in items],
            [x["chosen"] for x in items],
            [x["rejected"] for x in items],
            max_length,
            max_prompt_length,
        )
    return collate


def log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def seq_logps(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        logps, _ = response_logprobs_from_batch(model, batch)
    return logps


def pair_logps(policy, reference, batch, dev_p, dev_r):
    with torch.no_grad():                        # frozen reference: no gradients at all
        ref_c = seq_logps(reference, batch["chosen"], dev_r).to(dev_p)
        ref_r = seq_logps(reference, batch["rejected"], dev_r).to(dev_p)
    pol_c = seq_logps(policy, batch["chosen"], dev_p)
    pol_r = seq_logps(policy, batch["rejected"], dev_p)
    return pol_c, pol_r, ref_c, ref_r


@torch.no_grad()
def evaluate(policy, reference, tok, pairs, args, dev_p, dev_r):
    policy.eval()
    collate = make_collate(tok, args.max_length, args.max_prompt_length)
    losses, accs, margins = [], [], []
    for i in range(0, len(pairs), args.eval_bs):
        batch = collate(pairs[i : i + args.eval_bs])
        pc, pr, rc, rr = pair_logps(policy, reference, batch, dev_p, dev_r)
        r_w = args.beta * (pc - rc)
        r_l = args.beta * (pr - rr)
        losses.append(-F.logsigmoid(r_w - r_l))
        accs.append((r_w > r_l).float())
        margins.append(r_w - r_l)
    policy.train()
    return {
        "val_loss": torch.cat(losses).mean().item(),
        "val_reward_acc": torch.cat(accs).mean().item(),
        "val_reward_margin": torch.cat(margins).mean().item(),
    }


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--sft_path", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adamw"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_bs", type=int, default=4)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()
    args.out_dir = args.out_dir or f"outputs/dpo_{args.dataset}"
    args.ckpt_dir = args.ckpt_dir or f"outputs/checkpoints/dpo_{args.dataset}"

    assert torch.cuda.device_count() >= 2, "need 2 GPUs (policy on cuda:0, reference on cuda:1)"
    set_seed(args.seed)
    dev_p, dev_r = torch.device("cuda:0"), torch.device("cuda:1")
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    loader_fn, train_split, val_split = DATASETS[args.dataset]
    train_pairs = loader_fn(train_split, args.n_train, args.seed)
    val_pairs = loader_fn(val_split, args.n_val, args.seed)
    print(f"[{args.dataset}] train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")

    tok = AutoTokenizer.from_pretrained(args.sft_path)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(dev_p)
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(dev_r)
    reference.eval()
    reference.requires_grad_(False)
    policy.config.use_cache = False
    reference.config.use_cache = False
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable()
    policy.train()

    loader = DataLoader(
        train_pairs, batch_size=1, shuffle=True,
        collate_fn=make_collate(tok, args.max_length, args.max_prompt_length),
        generator=torch.Generator().manual_seed(args.seed),
    )
    total_steps = (len(loader) // args.accum) * args.epochs

    if args.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    cfg = dict(vars(args))
    cfg.update(
        torch=torch.__version__, transformers=transformers.__version__,
        datasets=datasets.__version__, prompt_template=PROMPT_TEMPLATE,
        reference_checkpoint=args.sft_path, n_train_actual=len(train_pairs),
        n_val_actual=len(val_pairs), optimizer_steps=total_steps,
        effective_batch=args.accum, lr_schedule="fixed (no warmup, no decay)",
        precision="fp32 weights + fp16 autocast + GradScaler (policy); reference fp32 weights + fp16 autocast",
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"optimizer steps: {total_steps}")

    # step-0 check: policy == reference, so expect loss ~0.6931 and reward accuracy 0.0
    v0 = evaluate(policy, reference, tok, val_pairs, args, dev_p, dev_r)
    log(metrics_path, {"type": "val", "step": 0, **v0})
    print("   >>> step 0 (before training):", {k: round(v, 4) for k, v in v0.items()})

    best_key, best_step = (-1.0, -float("inf")), 0
    step, micro, win, win_n = 0, 0, {"loss": 0.0, "acc": 0.0, "margin": 0.0}, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            pc, pr, rc, rr = pair_logps(policy, reference, batch, dev_p, dev_r)
            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
            scaler.scale(loss / args.accum).backward()
            with torch.no_grad():
                r_w = args.beta * (pc.detach() - rc)
                r_l = args.beta * (pr.detach() - rr)
                win["loss"] += loss.item()
                win["acc"] += (r_w > r_l).float().mean().item()
                win["margin"] += (r_w - r_l).mean().item()
                win_n += 1
            micro += 1
            if micro % args.accum != 0:
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                rec = {"type": "train", "step": step, "loss": win["loss"] / win_n,
                       "reward_acc": win["acc"] / win_n, "reward_margin": win["margin"] / win_n,
                       "grad_norm": gnorm, "peak_mem_gb": torch.cuda.max_memory_allocated(dev_p) / 1e9,
                       "elapsed_min": (time.time() - t0) / 60}
                log(metrics_path, rec)
                print(f"step {step}/{total_steps} | loss {rec['loss']:.4f} | acc {rec['reward_acc']:.3f} "
                      f"| margin {rec['reward_margin']:.4f} | gnorm {gnorm:.2f} "
                      f"| peak {rec['peak_mem_gb']:.1f}GB | {rec['elapsed_min']:.1f}min")
                win, win_n = {"loss": 0.0, "acc": 0.0, "margin": 0.0}, 0

            if step % args.eval_every == 0 or step == total_steps:
                v = evaluate(policy, reference, tok, val_pairs, args, dev_p, dev_r)
                log(metrics_path, {"type": "val", "step": step, **v})
                print(f"   >>> step {step} | val loss {v['val_loss']:.4f} | val acc {v['val_reward_acc']:.3f} "
                      f"| val margin {v['val_reward_margin']:.4f}")
                key = (v["val_reward_acc"], -v["val_loss"])
                if key > best_key:
                    best_key, best_step = key, step
                    save_model(policy, tok, os.path.join(args.ckpt_dir, "best"))
                    print("   >>> new best (by val reward accuracy), saved")

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    save_model(policy, tok, os.path.join(args.ckpt_dir, "final"))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"best_step": best_step, "best_val_reward_acc": best_key[0],
                   "best_val_loss": -best_key[1], "steps": step,
                   "minutes": (time.time() - t0) / 60,
                   "peak_mem_gb": torch.cuda.max_memory_allocated(dev_p) / 1e9}, f, indent=2)
    print("DONE. best step:", best_step, "| best val reward acc:", best_key[0])


if __name__ == "__main__":
    main()
