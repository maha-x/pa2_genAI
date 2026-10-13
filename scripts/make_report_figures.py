"""Builds the two training-curve figures used in the report from the metrics.jsonl files in the repo.
Usage (run from the repository root):

    python scripts/make_report_figures.py

Reads outputs/sft, outputs/dpo_hh, outputs/dpo_uf and outputs/lora_persona_beta03 (metrics.jsonl)
and writes report/fig_sft_lora.pdf/.png and report/fig_dpo.pdf/.png.
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREY, RED = "#1f6fb4", "#d95f02", "#666666", "#b2182b"

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def moving_avg(xs, ys, k):
    k = max(1, min(k, len(ys)))
    return xs[k - 1:], [sum(ys[i:i + k]) / k for i in range(len(ys) - k + 1)]

# ------------------------------------------------------------------ figure 1: SFT + final LoRA run
sft = load("outputs/sft/metrics.jsonl")
tr = [r for r in sft if r["type"] == "train" and math.isfinite(r["loss"])]
va = [r for r in sft if r["type"] == "val"]
lora = load("outputs/lora_persona_beta03/metrics.jsonl")
ltr = [r for r in lora if "loss" in r and "eval_loss" not in r]
lev = [r for r in lora if "eval_loss" in r]

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
a = ax[0]
a.plot([r["step"] for r in tr], [r["loss"] for r in tr], color=BLUE, alpha=0.28, lw=0.8, label="train (raw)")
x, y = moving_avg([r["step"] for r in tr], [r["loss"] for r in tr], 5)
a.plot(x, y, color=BLUE, lw=1.4, label="train (avg of 5 logs)")
a.plot([r["step"] for r in va], [r["val_loss"] for r in va], "o-", color=ORANGE, ms=3, lw=1.1, label="validation")
best = min(va, key=lambda r: r["val_loss"])
a.plot([best["step"]], [best["val_loss"]], "*", color=RED, ms=9, zorder=5, label=f"best (step {best['step']})")
a.set_ylim(1.05, 1.62); a.set_xlim(-30, 990); a.set_xlabel("optimizer step"); a.set_ylabel("per-token loss"); a.set_title("(a) SFT, Qwen2.5-0.5B"); a.legend(loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.4)

b = ax[1]
b.plot([r["step"] for r in ltr], [r["loss"] for r in ltr], color=BLUE, lw=1.3, label="train (per step)")
b.plot([0] + [r["step"] for r in lev], [math.log(2)] + [r["eval_loss"] for r in lev], "o-", color=ORANGE, ms=3.5, lw=1.1,
       label="held-out (20 pairs, per epoch)")
b.axhline(math.log(2), color=GREY, ls=":", lw=0.9)
b.text(39, math.log(2) + 0.015, "ln 2", ha="right", va="bottom", fontsize=6, color=GREY)
b.set_xlabel("optimizer step"); b.set_ylabel("DPO loss"); b.set_title(r"(b) LoRA-DPO ($\beta$=0.3): loss"); b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6)

c = ax[2]
c.plot([r["step"] for r in ltr], [r["rewards/margins"] for r in ltr], color=BLUE, lw=1.3, label="train margin")
c.plot([0] + [r["step"] for r in lev], [0.0] + [r["eval_rewards/margins"] for r in lev], "o-", color=ORANGE, ms=3.5, lw=1.1,
       label="held-out margin")
c2 = c.twinx(); c2.spines["right"].set_visible(True); c2.grid(False)
c2.plot([r["step"] for r in ltr], [r["rewards/accuracies"] for r in ltr], color=GREY, lw=0.9, ls="--", label="train accuracy (right)")
c2.set_ylim(-0.05, 1.1); c2.set_ylabel("reward accuracy")
c.set_xlabel("optimizer step"); c.set_ylabel("reward margin"); c.set_title(r"(c) LoRA-DPO: margin and accuracy")
h1, l1 = c.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels(); c.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=5.8, bbox_to_anchor=(1.0, 0.04))
fig.tight_layout(w_pad=0.9)
os.makedirs("report", exist_ok=True)
fig.savefig("report/fig_sft_lora.pdf"); fig.savefig("report/fig_sft_lora.png", dpi=170)

# ------------------------------------------------------------------ figure 2: DPO HH vs UF
runs = {"HH-RLHF": ("outputs/dpo_hh", BLUE, 300), "UltraFeedback": ("outputs/dpo_uf", ORANGE, 468)}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45))
panels = [("loss", "val_loss", "DPO loss"), ("reward_acc", "val_reward_acc", "reward accuracy"), ("reward_margin", "val_reward_margin", "reward margin")]
for axis, (tk, vk, title) in zip(ax, panels):
    for name, (d, col, best_step) in runs.items():
        m = load(f"{d}/metrics.jsonl")
        t = [r for r in m if r["type"] == "train"]; v = [r for r in m if r["type"] == "val"]
        axis.plot([r["step"] for r in t], [r[tk] for r in t], color=col, alpha=0.2, lw=0.7)
        x, y = moving_avg([r["step"] for r in t], [r[tk] for r in t], 10)
        axis.plot(x, y, color=col, lw=1.3, label=f"{name} train (avg 50 steps)")
        axis.plot([r["step"] for r in v], [r[vk] for r in v], "o--", color=col, ms=3, lw=0.9, label=f"{name} validation")
        bv = next(r for r in v if r["step"] == best_step)
        axis.plot([best_step], [bv[vk]], "*", color=RED, ms=9, zorder=5)
    axis.set_xlabel("optimizer step"); axis.set_title(title)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=BLUE, lw=1.3), Line2D([], [], color=BLUE, marker="o", ms=3, ls="--", lw=0.9),
           Line2D([], [], color=ORANGE, lw=1.3), Line2D([], [], color=ORANGE, marker="o", ms=3, ls="--", lw=0.9),
           Line2D([], [], color=RED, marker="*", ms=8, ls="")]
labels = ["HH-RLHF train (avg of 10 logs = 50 steps; raw faint)", "HH-RLHF validation (every 100 steps)",
          "UltraFeedback train (same smoothing)", "UltraFeedback validation", "selected checkpoint"]
fig.tight_layout(w_pad=0.9, rect=[0, 0.17, 1, 1])
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.3, frameon=False, columnspacing=1.2)
fig.savefig("report/fig_dpo.pdf"); fig.savefig("report/fig_dpo.png", dpi=170)
print("figures written")
