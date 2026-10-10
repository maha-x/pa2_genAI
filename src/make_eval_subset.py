import json
import os
import random

from huggingface_hub import hf_hub_download


def main(n=300, seed=42, out="eval_sets/alpaca_eval_300_seed42.json"):
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)
    print("Total AlpacaEval prompts:", len(data))  # expect 805

    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(data)), n))

    subset = [
        {
            "id": f"alpaca_eval_{i}",
            "instruction": data[i]["instruction"],
            "source": data[i].get("dataset"),
        }
        for i in idxs
    ]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved {len(subset)} prompts to {out}")


if __name__ == "__main__":
    main()
