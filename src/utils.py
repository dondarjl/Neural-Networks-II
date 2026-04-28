import torch

BATCH_SIZE = 128
EPS_LIST = [0.0, 0.01, 0.02, 0.04, 0.08]
EPOCHS = 10
LR = 1e-3
ARCHS = ["smallcnn", "resnet18"]


def clamp01(x):
    """Clampea los valores del tensor al rango válido [0, 1] de una imagen."""
    return torch.clamp(x, 0.0, 1.0)

def set_seed(seed):
    """Fija la semilla para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """Devuelve GPU si está disponible, si no CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")