
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
num_workers = os.cpu_count()

def create_dataloaders(transform: torchvision.transforms,
                       batch_size:int,
                       num_workers:int =num_workers):

    data_path = Path = ("data/")
    IMG_SIZE = 224
    train_data = datasets.CIFAR10(root= data_path, 
                                  train=True,
                                  download=True,
                                  transform=transform)
    test_data = datasets.CIFAR10(root=data_path,
                                 train=False,
                                 download=True,
                                 transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names
