"""
calibration.py — ECE, NLL y Reliability Diagram.
Siguiendo Guo et al. (2017), M=15 bins.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def get_predictions(model, loader, device, attack_fn=None):
    model.eval()
    all_conf, all_correct = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack_fn is not None:
            with torch.enable_grad():
                x = attack_fn(x, y)
        probs = F.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        all_conf.append(conf.cpu().numpy())
        all_correct.append((pred == y).cpu().numpy())
    return np.concatenate(all_conf), np.concatenate(all_correct)


def compute_ece(confidences, correctness, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correctness[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


@torch.no_grad()
def compute_nll(model, loader, device, attack_fn=None):
    model.eval()
    total_nll, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack_fn is not None:
            with torch.enable_grad():
                x = attack_fn(x, y)
        log_probs = F.log_softmax(model(x), dim=1)
        total_nll += F.nll_loss(log_probs, y, reduction='sum').item()
        total     += x.size(0)
    return total_nll / total


def reliability_diagram(confidences, correctness, n_bins=15,
                        title="Reliability Diagram", save_path=None):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(correctness[mask].mean())

    ece = compute_ece(confidences, correctness, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.6, label='Accuracy')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title} — ECE={ece*100:.2f}%')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig
