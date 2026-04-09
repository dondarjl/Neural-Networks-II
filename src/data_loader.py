# dataloader.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size):
    # CIFAR-10 in [0,1]
    cifar_transform = transforms.Compose([transforms.ToTensor()])

    cifar_train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=cifar_transform)
    cifar_test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar_transform)

    # DataLoader stability: use num_workers=0 (safe on macOS / Python 3.12)
    cifar_train_loader = DataLoader(cifar_train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=torch.cuda.is_available())
    cifar_test_loader  = DataLoader(cifar_test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    cifar_classes = cifar_train_ds.classes

    return cifar_train_loader, cifar_test_loader, cifar_classes