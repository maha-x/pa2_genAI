import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="DPO")
    p.add_argument("--window", type=int, default=10)
    a = p.parse_args()

    train, val = [], []
    for line in open(a.metrics):
        r = json.loads(line)
        (train if r["type"] == "train" else val).append(r)

    best_step = None
    summ = os.path.join(os.path.dirname(a.metrics), "summary.json")
    if os.path.exists(summ):
        best_step = json.load(open(summ)).get("best_step")

    panels = [("loss", "val_loss", "DPO loss"),
              ("reward_acc", "val_reward_acc", "Reward accuracy"),
              ("reward_margin", "val_reward_margin", "Reward margin")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = [r["step"] for r in train]
    for ax, (tk, vk, name) in zip(axes, panels):
        ys = [r[tk] for r in train]
        k = max(1, min(a.window, len(ys)))
        sm = [sum(ys[i : i + k]) / k for i in range(len(ys) - k + 1)]
        ax.plot(xs, ys, alpha=0.3, label="train (raw, per logging window)")
        ax.plot(xs[k - 1 :], sm, label=f"train (moving avg, window={k})")
        ax.plot([r["step"] for r in val], [r[vk] for r in val], "o-", label="validation")
        if best_step:
            ax.axvline(best_step, color="gray", linestyle="--", label=f"best checkpoint (step {best_step})")
        ax.set_xlabel("optimizer step")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(a.title)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
