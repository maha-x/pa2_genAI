import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--adapter_path", default=None, help="omit to generate with the plain base model")
    p.add_argument("--eval_path", default="eval_sets/persona_style_eval_30.json")
    p.add_argument("--out_path", required=True)
    p.add_argument("--generator_name", required=True)
    p.add_argument("--dataset_name", default="persona_style_eval")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16).to("cuda:0")
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
        print("loaded and merged adapter from", args.adapter_path)
    model.eval()

    items = json.load(open(args.eval_path))
    if args.limit:
        items = items[: args.limit]

    results = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["instruction"]}],
                                           tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=args.repetition_penalty,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for it, text in zip(batch, texts):
            results.append({"id": it["id"], "instruction": it["instruction"], "output": text.strip(),
                            "generator": args.generator_name, "dataset": args.dataset_name,
                            "category": it.get("category")})
        print(f"done {len(results)}/{len(items)}", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    json.dump(results, open(args.out_path, "w"), indent=2)
    print("saved", args.out_path)


if __name__ == "__main__":
    main()
