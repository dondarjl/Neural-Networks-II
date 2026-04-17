"""
evaluate.py
Orquesta los experimentos de robustez y calibración sobre múltiples
epsilons y ataques. Usa las funciones puras de calibration.py.
"""

import pandas as pd
import torch
import torch.nn.functional as F

from src.attacks import attack_suite          # fgsm_attack, pgd_attack, attack_suite
from src.calibration import (                 # funciones puras
    get_predictions,
    compute_ece,
    compute_nll,
    reliability_diagram,
)


# ─────────────────────────────────────────────
# 1. ROBUSTEZ: accuracy bajo ataques vs epsilon
# ─────────────────────────────────────────────

def eval_attack_accuracy(model, loader, device, attack_fn):
    """Accuracy del modelo bajo un único ataque."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
    return correct / total


def sweep_attacks(model, loader, device, eps_list,
                  pgd_steps_list=(10, 40), max_batches=None):
    """
    Devuelve un DataFrame con accuracy para múltiples
    ataques y valores de epsilon.

    Columnas: epsilon | attack | accuracy
    """
    rows = []
    for eps in eps_list:
        suite = attack_suite(model, eps, pgd_steps_list=pgd_steps_list)
        for name, fn in suite.items():
            if max_batches is None:
                acc = eval_attack_accuracy(model, loader, device, fn)
            else:
                model.eval()
                correct, total = 0, 0
                for bi, (x, y) in enumerate(loader):
                    if bi >= max_batches:
                        break
                    x, y = x.to(device), y.to(device)
                    x_adv = fn(x, y)
                    with torch.no_grad():
                        pred = model(x_adv).argmax(1)
                    correct += (pred == y).sum().item()
                    total   += x.size(0)
                acc = correct / total
            rows.append({
                "epsilon": float(eps),
                "attack":  name.split(" (eps=")[0],
                "accuracy": acc,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2. CALIBRACIÓN: ECE y NLL vs epsilon
# ─────────────────────────────────────────────

def sweep_calibration(model, loader, device, eps_list,
                      pgd_steps=40, n_bins=15):
    """
    Devuelve un DataFrame con ECE y NLL para múltiples
    ataques y valores de epsilon.

    Columnas: epsilon | attack | ECE | NLL
    """
    rows = []
    for eps in eps_list:
        suite = attack_suite(model, eps, pgd_steps_list=(pgd_steps,))
        for name, fn in suite.items():
            confs, corrects = get_predictions(model, loader, device,
                                              attack_fn=fn)
            ece = compute_ece(confs, corrects, n_bins=n_bins)
            nll = compute_nll(model, loader, device, attack_fn=fn)
            rows.append({
                "epsilon": float(eps),
                "attack":  name.split(" (eps=")[0],
                "ECE":     ece,
                "NLL":     nll,
            })
    return pd.DataFrame(rows)



def eval_reliability_diagrams(model, loader, device, eps,
                               pgd_steps=40, save_dir="plots",
                               arch_name="model", training_label="std"):
    """
    Genera reliability diagrams para Clean, FGSM y PGD-{pgd_steps}
    con el epsilon dado.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    suite = attack_suite(model, eps, pgd_steps_list=(pgd_steps,))
    for name, fn in suite.items():
        confs, corrects = get_predictions(model, loader, device,
                                          attack_fn=fn)
        attack_label = name.split(" (eps=")[0]
        title     = f"{arch_name} [{training_label}] — {attack_label} ε={eps:.2f}"
        save_path = f"{save_dir}/{arch_name}_{training_label}_{attack_label}_eps{eps:.2f}.png"
        reliability_diagram(confs, corrects, title=title,
                            save_path=save_path)
        print(f"  Saved: {save_path}")