print('hola')


import os
import torch
import torch.optim as optim

from src.trainer import train_one_epoch, train_one_epoch_adv_fgsm

from src.models.resnet18 import get_resnet18_cifar
from src.models.small_cnn import SmallCIFARCNN
from src.data_loader import get_cifar10_loaders
from src.utils import get_device, set_seed, BATCH_SIZE, EPS_LIST, ARCHS, LR, EPOCHS


device = get_device()
set_seed(42)

train_loader, _, _ = get_cifar10_loaders(BATCH_SIZE)

def get_model(arch):
    if arch == "smallcnn":
        return SmallCIFARCNN()
    else:
        return get_resnet18_cifar()

for arch in ARCHS:

    # -------------------------
    # 1. STANDARD MODEL
    # -------------------------
    model = get_model(arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer)
        print(f"[{arch}][STD] Epoch {epoch+1}: {loss:.4f}")

    os.makedirs(f"checkpoints/{arch}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{arch}/std.pth")

    # -------------------------
    # 2. ADVERSARIAL MODELS
    # -------------------------
    for eps in EPS_LIST:
        model = get_model(arch).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            loss = train_one_epoch_adv_fgsm(model, train_loader, optimizer, eps)
            print(f"[{arch}][ADV eps={eps}] Epoch {epoch+1}: {loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/{arch}/adv_eps{eps}.pth")