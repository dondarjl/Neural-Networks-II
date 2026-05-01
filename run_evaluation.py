import os
import torch
import pandas as pd
from glob import glob

from src.models.small_cnn import SmallCIFARCNN
from src.models.resnet18 import get_resnet18_cifar
from src.data_loader import get_cifar10_loaders
from src.evaluate import sweep_attacks, sweep_calibration, eval_reliability_diagrams
from src.utils import get_device, BATCH_SIZE, EPS_LIST, ARCHS, LR, EPOCHS

device = get_device()
_, test_loader, _ = get_cifar10_loaders(BATCH_SIZE)
os.makedirs("results", exist_ok=True)

# ── Subset para evaluación rápida ─────────────────────────────
FAST_MODE = False
FAST_FRACTION = 0.2  # 2000 imágenes en vez de 10000

if FAST_MODE:
    from torch.utils.data import Subset, DataLoader
    import numpy as np

    n = int(len(test_loader.dataset) * FAST_FRACTION)
    idx = np.random.choice(len(test_loader.dataset), n, replace=False)
    test_loader = DataLoader(
        Subset(test_loader.dataset, idx),
        batch_size=test_loader.batch_size,
        num_workers=test_loader.num_workers,
        pin_memory=True,
    )
    print(f"  ⚡ FAST_MODE: evaluando con {n} imágenes ({FAST_FRACTION*100:.0f}%)")
# ── Helpers de reanudación ─────────────────────────────────────────────────────

def checkpoint_path(arch, tag, kind):
    """Ruta del CSV parcial que actúa como flag de completado."""
    return f"results/.done_{arch}_{tag}_{kind}.csv"

def is_done(arch, tag, kind):
    return os.path.exists(checkpoint_path(arch, tag, kind))

def mark_done(df, arch, tag, kind):
    """Guarda el resultado y marca la tarea como completada."""
    df.to_csv(checkpoint_path(arch, tag, kind), index=False)

# ── Helpers de modelo ──────────────────────────────────────────────────────────

def get_model(arch):
    if arch == "smallcnn":
        return SmallCIFARCNN()
    else:
        return get_resnet18_cifar()

# ── Loop principal ─────────────────────────────────────────────────────────────

for arch in ARCHS:

    model_paths = [("std", 0.0)] + [(f"adv_eps{eps}", eps) for eps in EPS_LIST if eps > 0]

    for tag, train_eps in model_paths:

        acc_done = is_done(arch, tag, "acc")
        cal_done = is_done(arch, tag, "cal")

        if acc_done and cal_done:
            print(f"  ↩  Saltando {arch}/{tag} (ya completado)")
            continue

        # Cargar modelo solo si hay algo que hacer
        print(f"  ▶  Procesando {arch}/{tag} ...")
        model = get_model(arch)
        model.load_state_dict(torch.load(
            f"checkpoints/{arch}/{tag}.pth",
            map_location=device,
            weights_only=True,
        ))
        model.to(device)

        # Robustness
        if not acc_done:
            df_acc = sweep_attacks(model, test_loader, device, EPS_LIST)
            df_acc["arch"] = arch
            df_acc["train_eps"] = train_eps
            mark_done(df_acc, arch, tag, "acc")
            print(f"     ✓ accuracy guardado")

        # Calibration
        if not cal_done:
            df_cal = sweep_calibration(model, test_loader, device, EPS_LIST)
            df_cal["arch"] = arch
            df_cal["train_eps"] = train_eps
            mark_done(df_cal, arch, tag, "cal")
            print(f"     ✓ calibración guardada")

        # Reliability diagrams (sin checkpoint: son plots, rápidos de regenerar)
        eval_reliability_diagrams(
            model, test_loader, device,
            eps=0.08, pgd_steps=10,
            save_dir="plots",
            arch_name=arch,
            training_label=tag,
        )

# ── Consolidar resultados finales ──────────────────────────────────────────────

acc_files = sorted(glob("results/.done_*_acc.csv"))
cal_files = sorted(glob("results/.done_*_cal.csv"))

if not acc_files or not cal_files:
    print("⚠ No se encontraron resultados parciales.")
else:
    pd.concat([pd.read_csv(f) for f in acc_files]).to_csv("results/accuracy.csv", index=False)
    pd.concat([pd.read_csv(f) for f in cal_files]).to_csv("results/calibration.csv", index=False)
    print(f"\n✅ Todos los experimentos completados. "
          f"({len(acc_files)} combinaciones arch/train_eps)")