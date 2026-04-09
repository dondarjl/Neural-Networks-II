import os
import torch
import pandas as pd

from src.models.small_cnn import SmallCIFARCNN
from src.models.resnet18 import get_resnet18_cifar
from src.data_loader import get_cifar10_loaders
from src.evaluate import sweep_attacks, sweep_calibration
from src.utils import get_device, BATCH_SIZE, EPS_LIST, ARCHS, LR, EPOCHS

device = get_device()

_, test_loader, _ = get_cifar10_loaders(BATCH_SIZE)

def get_model(arch):
    if arch == "smallcnn":
        return SmallCIFARCNN()
    else:
        return get_resnet18_cifar()

all_results = []

for arch in ARCHS:

    model_paths = [("std", 0.0)] + [(f"adv_eps{eps}", eps) for eps in EPS_LIST if eps > 0]

    for tag, train_eps in model_paths:

        model = get_model(arch)
        model.load_state_dict(torch.load(f"checkpoints/{arch}/{tag}.pth"))
        model.to(device)

        # Robustness
        df_acc = sweep_attacks(model, test_loader, device, EPS_LIST)
        df_acc["arch"] = arch
        df_acc["train_eps"] = train_eps

        # Calibration
        df_cal = sweep_calibration(model, test_loader, device, EPS_LIST)
        df_cal["arch"] = arch
        df_cal["train_eps"] = train_eps

        all_results.append((df_acc, df_cal))

# Save
os.makedirs("results", exist_ok=True)

pd.concat([x[0] for x in all_results]).to_csv("results/accuracy.csv", index=False)
pd.concat([x[1] for x in all_results]).to_csv("results/calibration.csv", index=False)

print("All experiments completed.")