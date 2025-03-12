import torch
from torchvision import transforms

from PIL import ImageFilter, ImageOps
import random


def get_dataset_normalize(dataset: str):
    """Get corresponding normalization transform for each dataset."""

    if dataset == 'cifar100':
        return transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif dataset == 'cifar10':
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'tinyimagenet':
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset in ['imagenet100', 'imagenet']:
        return transforms.Normalize(
            (0.485, 0.456, 0.406), (0.228, 0.224, 0.225))
    elif dataset in ['clear10', 'clear100']:
        return transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'svhn':
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif dataset == 'cars':
        return transforms.Normalize(
            (0.4707, 0.4602, 0.4550), (0.2638, 0.2629, 0.2678))    

    else:
        raise ValueError(f'Base Trandforms for dataset "{dataset}" not supported')


def get_dataset_crop(dataset: str):
    """Get corresponding crop transform for each dataset."""

    if dataset == 'cifar100':
        # return transforms.RandomCrop(32, padding=4),
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.), antialias=True)
    elif dataset == 'cifar10':
        # return transforms.RandomCrop(32, padding=4)
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.), antialias=True)
    elif dataset in ['imagenet100', 'imagenet', 'clear100']:
        # return transforms.RandomCrop(64, padding=8)
        return transforms.RandomResizedCrop(224, scale=(0.2, 1.), antialias=True)
    else:
        raise ValueError("Dataset not supported.")
    
    
def get_common_transforms(dataset: str = 'cifar100'):
    "Common transforms for self supervised models for better comparison"

    normalize = get_dataset_normalize(dataset)
    crop = get_dataset_crop(dataset)
    all_transforms = [
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ]
    return all_transforms

class MultipleCropsTransform:
    """Take N random augmented views of one image."""
    def __init__(self, base_transform, n_crops=20, online_transforms=True):
        self.base_transform = base_transform
        self.n_crops = n_crops
        self.online_transforms = online_transforms

    def __call__(self, x):
        if self.online_transforms:
            stacked_views = [] # List of tensors, each contains one view for all samples in x
            for _ in range(self.n_crops):
                view_list = [self.base_transform(sample) for sample in x]
                stacked_views.append(torch.stack(view_list, dim=0))
            return stacked_views
        else:
            return [self.base_transform(x) for i in range(self.n_crops)]


def get_transforms(dataset: str = 'cifar100', n_crops: int = 2, online_transforms: bool = True):
    """Returns augmentations for self supervised models"""
    return MultipleCropsTransform(transforms.Compose(get_common_transforms(dataset)), n_crops, online_transforms)
