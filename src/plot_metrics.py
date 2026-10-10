import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=5)
    a = p.parse_args()

    train, val = [], []
    for line in open(a.metrics):
        r = json.loads(line)
        (train if r["type"] == "train" else val).append(r)

    xs = [r["step"] for r in train]
    ys = [r["loss"] for r in train]
    k = max(1, min(a.window, len(ys)))
    sm = [sum(ys[i : i + k]) / k for i in range(len(ys) - k + 1)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, alpha=0.3, label="train loss (raw, avg per logging window)")
    ax.plot(xs[k - 1 :], sm, label=f"train loss (moving avg, window={k})")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("per-token cross-entropy loss")
    ax.set_title("SFT: Qwen2.5-0.5B on UltraChat")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
