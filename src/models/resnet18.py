"""
resnet18.py – ResNet-18 adaptada para CIFAR-10 (imágenes 32×32).

Cambios respecto a la versión ImageNet:
  1. conv1: 7×7 stride-2  →  3×3 stride-1  (no aplasta las imágenes pequeñas)
  2. Se elimina el MaxPool inicial
  3. FC final: 512 → num_classes
"""
import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)

    # 1. Reemplazar primera conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 2. Eliminar el MaxPool que reduciría 32→16 antes del primer bloque residual
    model.maxpool = nn.Identity()

    # 3. Cabeza clasificadora para 10 clases
    model.fc = nn.Linear(512, num_classes)

    return model