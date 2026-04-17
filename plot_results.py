"""
plot_results.py
Reads results/accuracy.csv and results/calibration.csv
and generates all plots required by the project statement:

  1. Clean accuracy vs ε (per arch, per training mode)
  2. Robust accuracy (FGSM, PGD) vs ε
  3. ECE vs ε
  4. NLL vs ε

Run from the project root:
    python plot_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
acc_df = pd.read_csv("results/accuracy.csv")
cal_df = pd.read_csv("results/calibration.csv")

ARCHS = acc_df["arch"].unique().tolist()
ATTACK_COLORS = {
    "Clean":   "#2196F3",
    "FGSM":    "#FF9800",
    "PGD-10":  "#F44336",
    "PGD-40":  "#9C27B0",
}

# ── Helper ────────────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def training_label(train_eps):
    return "Standard" if train_eps == 0.0 else f"FGSM-AT ε={train_eps:.2f}"


# ── Plot 1 & 2: Accuracy vs ε (one figure per arch) ──────────────────────────
for arch in ARCHS:
    arch_acc = acc_df[acc_df["arch"] == arch]
    train_eps_values = sorted(arch_acc["train_eps"].unique())

    fig, axes = plt.subplots(
        1, len(train_eps_values),
        figsize=(5 * len(train_eps_values), 4),
        sharey=True,
    )
    if len(train_eps_values) == 1:
        axes = [axes]

    for ax, tr_eps in zip(axes, train_eps_values):
        subset = arch_acc[arch_acc["train_eps"] == tr_eps]
        for attack in subset["attack"].unique():
            row = subset[subset["attack"] == attack]
            color = ATTACK_COLORS.get(attack, None)
            ax.plot(
                row["epsilon"], row["accuracy"],
                marker="o", label=attack, color=color,
            )
        ax.set_title(training_label(tr_eps))
        ax.set_xlabel("ε")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{arch} — Accuracy vs ε", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{arch}_accuracy_vs_eps.png")


# ── Plot 3: ECE vs ε ──────────────────────────────────────────────────────────
for arch in ARCHS:
    arch_cal = cal_df[cal_df["arch"] == arch]
    train_eps_values = sorted(arch_cal["train_eps"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    for tr_eps in train_eps_values:
        subset = arch_cal[
            (arch_cal["train_eps"] == tr_eps) &
            (arch_cal["attack"] == "Clean")
        ]
        ax.plot(
            subset["epsilon"], subset["ECE"],
            marker="o", label=training_label(tr_eps),
        )

    ax.set_title(f"{arch} — ECE (Clean) vs Training ε")
    ax.set_xlabel("ε (training)")
    ax.set_ylabel("ECE")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"{arch}_ece_vs_eps.png")


# ── Plot 4: NLL vs ε ──────────────────────────────────────────────────────────
for arch in ARCHS:
    arch_cal = cal_df[cal_df["arch"] == arch]
    train_eps_values = sorted(arch_cal["train_eps"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    for tr_eps in train_eps_values:
        subset = arch_cal[
            (arch_cal["train_eps"] == tr_eps) &
            (arch_cal["attack"] == "Clean")
        ]
        ax.plot(
            subset["epsilon"], subset["NLL"],
            marker="o", label=training_label(tr_eps),
        )

    ax.set_title(f"{arch} — NLL (Clean) vs Training ε")
    ax.set_xlabel("ε (training)")
    ax.set_ylabel("NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"{arch}_nll_vs_eps.png")


# ── Plot 5: ECE vs ε breakdown by attack type ─────────────────────────────────
for arch in ARCHS:
    arch_cal = cal_df[cal_df["arch"] == arch]
    # Use the adversarially trained model with highest eps
    max_tr_eps = arch_cal["train_eps"].max()
    for tr_eps in sorted(arch_cal["train_eps"].unique()):
        subset = arch_cal[arch_cal["train_eps"] == tr_eps]
        fig, ax = plt.subplots(figsize=(7, 4))
        for attack in subset["attack"].unique():
            row = subset[subset["attack"] == attack]
            color = ATTACK_COLORS.get(attack, None)
            ax.plot(
                row["epsilon"], row["ECE"],
                marker="o", label=attack, color=color,
            )
        ax.set_title(f"{arch} [{training_label(tr_eps)}] — ECE vs ε by attack")
        ax.set_xlabel("ε (evaluation)")
        ax.set_ylabel("ECE")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        tr_tag = "std" if tr_eps == 0.0 else f"adv{tr_eps:.2f}"
        savefig(fig, f"{arch}_{tr_tag}_ece_by_attack.png")

print("\nAll plots saved to", PLOTS_DIR)
