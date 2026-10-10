import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import format_prompt


def save(results, items, out_path):
    ordered = [results[it["id"]] for it in items if it["id"] in results]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--eval_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--generator_name", required=True)
    p.add_argument("--dataset_name", default="alpaca_eval")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"  # IMPORTANT for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()

    with open(args.eval_path) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    # Resume support: skip prompts that already have an answer saved
    results = {}
    if os.path.exists(args.out_path):
        with open(args.out_path) as f:
            results = {r["id"]: r for r in json.load(f)}
    todo = [it for it in items if it["id"] not in results]
    todo.sort(key=lambda it: len(it["instruction"]))  # similar lengths = less padding

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        prompts = [format_prompt(it["instruction"]) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy decoding
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for it, text in zip(batch, texts):
            results[it["id"]] = {
                "id": it["id"],
                "instruction": it["instruction"],
                "output": text.strip(),
                "generator": args.generator_name,
                "dataset": args.dataset_name,
            }
        save(results, items, args.out_path)  # checkpoint after every batch
        print(f"Done {len(results)}/{len(items)}")


if __name__ == "__main__":
    main()
