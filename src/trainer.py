import torch
import torch.nn.functional as F
from src.utils import get_device
from src.attacks import fgsm_attack
from src.attacks import fgsm_attack_train

device = get_device()


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


def train_one_epoch_adv_fgsm(model, loader, optimizer, eps, lam=0.5):
    """
    Simple adversarial training:
    loss = (1-lam)*CE(clean) + lam*CE(adv)
    where adv is generated using FGSM on the current model.
    """
    model.train()
    total_loss, total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        # --- generate adversarial examples (using current model) ---
        # We generate x_adv in "eval-style" but gradients wrt x are required.
        x_adv = fgsm_attack_train(model, x, y, eps=eps)
        # --- compute mixed loss and update parameters ---
        logits_clean = model(x)
        logits_adv   = model(x_adv)

        loss_clean = F.cross_entropy(logits_clean, y)
        loss_adv   = F.cross_entropy(logits_adv, y)

        loss = (1 - lam) * loss_clean + lam * loss_adv
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total