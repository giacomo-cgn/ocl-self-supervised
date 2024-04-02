import torch
from torchvision import transforms

from PIL import ImageFilter, ImageOps
import random


def get_dataset_transforms(dataset: str):
    """Get corresponding normalization transform for each dataset."""

    if dataset == 'cifar100':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )])
    elif dataset == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )])
    elif dataset == 'tinyimagenet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )])
    elif dataset in ['imagenet100', 'imagenet']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
    )])
    else:
        raise ValueError("Dataset not supported")


def get_dataset_crop(dataset: str):
    """Get corresponding crop transform for each dataset."""

    if dataset == 'cifar100':
        # return transforms.RandomCrop(32, padding=4),
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset == 'cifar10':
        # return transforms.RandomCrop(32, padding=4)
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset in ['imagenet100', 'imagenet']:
        # return transforms.RandomCrop(64, padding=8)
        return transforms.RandomResizedCrop(224, scale=(0.2, 1.))
    else:
        raise ValueError("Dataset not supported.")


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k
    
class MultipleCropsTransform:
    """Take N random crops of one image."""
    def __init__(self, base_transform, n_crops=20):
        self.base_transform = base_transform
        self.n_crops = n_crops

    def __call__(self, x):
        crops = []
        for sample in x:
            for _ in range(self.n_crops):
                crops.append(self.base_transform(sample))
        
        # Concat all crops in a single torch vector and return
        return torch.stack(crops, dim=0)
        


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_transforms_simsiam(dataset: str = 'cifar100'):
    """Returns SimSiam augmentations with dataset specific crop."""

    all_transforms = [
        get_dataset_crop(dataset),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip()
    ]
    
    all_transforms.append(get_dataset_crop(dataset))

    return TwoCropsTransform(transforms.Compose(all_transforms))

def get_transforms_barlow_twins(dataset: str = 'cifar100'):
    """Returns Barlow Twins augmentations with dataset specific crop."""
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
        ]

    return all_transforms

def get_transforms_byol(dataset: str = 'cifar100'):
    """Returns BYOL augmentations with dataset specific crop."""
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(sigma=[.1, 2.]),
        ]
    return all_transforms
 

def get_common_transforms(dataset: str = 'cifar100'):
    "Common transforms for self supervised models for better comparison"
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ]

    return all_transforms
    

def get_transforms(dataset: str, model: str, n_crops: int = 2):
    """Returns augmentations for self supervised models"""

    if model == "simsiam":
        all_transforms = get_transforms_simsiam(dataset)

    elif model == "barlow_twins":
        all_transforms = get_transforms_barlow_twins(dataset)

    elif model == "byol":
        all_transforms = get_transforms_byol(dataset)

    elif model == "common":
        all_transforms = get_common_transforms(dataset)

    else:
        raise ValueError(f"Model {model} not supported")

    if n_crops > 2:
        return MultipleCropsTransform(transforms.Compose(all_transforms), n_crops)
    else:
        return TwoCropsTransform(transforms.Compose(all_transforms))
