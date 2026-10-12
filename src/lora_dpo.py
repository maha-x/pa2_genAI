import argparse
import json
import os
import random

import datasets
import numpy as np
import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def to_record(row):
    """Plain strings -> TRL conversational DPO record (role-message lists)."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def plot_log(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [r for r in log if "loss" in r and "eval_loss" not in r]
    ev = [r for r in log if "eval_loss" in r]
    panels = [("loss", "eval_loss", "DPO loss"),
              ("rewards/accuracies", "eval_rewards/accuracies", "Reward accuracy"),
              ("rewards/margins", "eval_rewards/margins", "Reward margin")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tk, vk, name) in zip(axes, panels):
        ax.plot([r["step"] for r in train if tk in r], [r[tk] for r in train if tk in r],
                label="train (per logging step)")
        ex = [r["step"] for r in ev if vk in r]
        if ex:
            ax.plot(ex, [r[vk] for r in ev if vk in r], "o-", label="held-out validation")
        ax.set_xlabel("optimizer step")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("LoRA-DPO persona tuning (SmolLM2-360M-Instruct)")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--out_dir", default="outputs/lora_persona")
    p.add_argument("--adapter_dir", default="outputs/checkpoints/lora_persona")
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = json.load(open(args.data_path))
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.n_val], rows[args.n_val:]
    train_ds = Dataset.from_list([to_record(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_record(r) for r in val_rows])
    print(f"train pairs: {len(train_ds)} | held-out validation pairs: {len(eval_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=args.target_modules.split(","), bias="none", task_type="CAUSAL_LM",
    )
    cfg = DPOConfig(
        output_dir=os.path.join(args.out_dir, "trainer_tmp"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=True,
        seed=args.seed,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=lora)

    n_trainable = sum(q.numel() for q in trainer.model.parameters() if q.requires_grad)
    n_total = sum(q.numel() for q in trainer.model.parameters())
    print(f"trainable parameters: {n_trainable:,} of {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    trainer.evaluate()          # step-0 baseline: adapter starts at zero, so rewards are 0 and accuracy 0
    trainer.train()

    trainer.save_model(args.adapter_dir)      # saves only the LoRA adapter
    tok.save_pretrained(args.adapter_dir)

    log = trainer.state.log_history
    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")
    info = dict(vars(args))
    info.update(torch=torch.__version__, transformers=transformers.__version__, trl=trl.__version__,
                peft=peft.__version__, datasets=datasets.__version__,
                n_train=len(train_ds), n_val=len(eval_ds),
                trainable_params=n_trainable, total_params=n_total,
                reference="the base model with the adapter disabled (TRL default when peft_config is used)",
                precision="fp32 weights + fp16 mixed precision")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=2)
    plot_log(log, os.path.join(args.out_dir, "lora_curves.png"))
    ev = [r for r in log if "eval_loss" in r]
    print("final eval:", {k: round(v, 4) for k, v in (ev[-1].items() if ev else []) if isinstance(v, float)})
    print("DONE. adapter saved to", args.adapter_dir)


if __name__ == "__main__":
    main()
