import argparse
import json
import os
import random


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True)
    p.add_argument("--dpo_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--sft_name", default="sft")
    p.add_argument("--dpo_name", default="dpo")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    sft, dpo = load(a.sft_path), load(a.dpo_path)
    ids = [i for i in sft if i in dpo]          # keep the SFT file's order
    if len(ids) != len(sft) or len(ids) != len(dpo):
        print(f"WARNING: only {len(ids)} ids in common (sft {len(sft)}, dpo {len(dpo)})")

    rng = random.Random(a.seed)                 # fixed seed -> same A/B order every time
    n_dpo_first = n_identical = 0
    os.makedirs(os.path.dirname(a.out_path) or ".", exist_ok=True)
    with open(a.out_path, "w") as f:
        for i in ids:
            assert sft[i]["instruction"] == dpo[i]["instruction"], f"instruction mismatch for {i}"
            dpo_first = rng.random() < 0.5
            first, second = (dpo[i], sft[i]) if dpo_first else (sft[i], dpo[i])
            names = (a.dpo_name, a.sft_name) if dpo_first else (a.sft_name, a.dpo_name)
            n_dpo_first += dpo_first
            n_identical += first["output"].strip() == second["output"].strip()
            f.write(json.dumps({
                "pair_id": i,
                "instruction": sft[i]["instruction"],
                "category": sft[i].get("category"),
                "response_a": first["output"],
                "response_b": second["output"],
                "hidden": {"a": names[0], "b": names[1]},   # judge never sees this
            }) + "\n")
    print(f"pairs written: {len(ids)} -> {a.out_path}")
    print(f"{a.dpo_name} placed first (as Response A) in {n_dpo_first} pairs, {len(ids) - n_dpo_first} as B")
    print(f"identical responses: {n_identical}")


if __name__ == "__main__":
    main()
