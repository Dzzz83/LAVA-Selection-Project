import torch
import torchvision.transforms as transforms 
MEAN =(0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
"""
    FLip and flops augmentation for CIFAR-10 and CIFAR-100
"""
def get_week_augmentation():
    train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean =MEAN, std=STD)
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return train_transforms, val_transforms
def get_Trival_Augmentation():
    train_transform = transforms.Compose([
            transforms.TrivialAugmentWide(), 
            transforms.ToTensor(),
            transforms.Normalize(mean =MEAN, std=STD)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return train_transform, val_transforms