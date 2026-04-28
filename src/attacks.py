import torch
import torch.nn.functional as F
from src.utils import clamp01



def fgsm_attack(model, x, y, eps, targeted=False, y_target=None):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    if targeted:
        assert y_target is not None
        loss = F.cross_entropy(logits, y_target)
        grad_sign = torch.autograd.grad(loss, x_adv)[0].sign()
        x_adv = x_adv - eps * grad_sign
    else:
        loss = F.cross_entropy(logits, y)
        grad_sign = torch.autograd.grad(loss, x_adv)[0].sign()
        x_adv = x_adv + eps * grad_sign
    return clamp01(x_adv.detach())


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True,
               targeted=False, y_target=None):
    model.eval()
    x_orig = x.detach()
    if random_start:
        x_adv = x_orig + (2*torch.rand_like(x_orig) - 1.0) * eps
        x_adv = clamp01(x_adv)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        if targeted:
            assert y_target is not None
            loss = F.cross_entropy(logits, y_target)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - alpha * grad.sign()
        else:
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + alpha * grad.sign()

        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = clamp01(x_orig + delta)

    return x_adv.detach()


def attack_suite(model, eps, pgd_steps_list=(10, 40)):
    """
    Devuelve un dict de callables (x, y) -> x_adv para un epsilon dado.
    Usado por sweep_attacks y sweep_calibration en evaluate.py.
    """
    suite = {}
    suite["Clean"] = lambda x, y: x

    if eps > 0:
        suite[f"FGSM (eps={eps:.2f})"] = (
            lambda x, y, _eps=eps: fgsm_attack(model, x, y, eps=_eps)
        )
        for steps in pgd_steps_list:
            alpha = eps / 4
            suite[f"PGD-{steps} (eps={eps:.2f})"] = (
                lambda x, y, a=alpha, s=steps, _eps=eps:
                    pgd_attack(model, x, y, eps=_eps, alpha=a,
                               steps=s, random_start=True)
            )

    return suite


def fgsm_attack_train(model, x, y, eps):
    """
    FGSM para adversarial training.
    NO cambia el modo del modelo (train stays train).
    """
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)

    grad = torch.autograd.grad(loss, x_adv)[0]

    x_adv = x_adv + eps * grad.sign()

    return torch.clamp(x_adv.detach(), 0.0, 1.0)