from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import random


from .transforms import clamp_transform, get_dataset_normalize


def get_cla_transforms(dataset: str = "cifar100", n_crops: int = 2):

    normalize = get_dataset_normalize(dataset=dataset)

    if dataset in ['cifar10', 'cifar100']:
        blur_kernel = 5
    elif dataset in ['imagenet', 'imagenet100', 'clear10', 'clear100']:
        blur_kernel = 23
    elif dataset in ['tinyimagenet']:
        blur_kernel = 9
    else:
        raise ValueError(f"Dataset {dataset} not supported for blur kernel")

    augs = lambda tanh_val : transforms.Compose([
        get_dataset_crop_scale(dataset=dataset, scale=(1.0-(0.9*tanh_val) , 1.)), # tanh_val=0.0 -> 1.0, tanh_val=1.0 -> 0.1
        transforms.RandomHorizontalFlip(p=0.1 + (0.4*tanh_val)), # tanh_val=0.0 -> 0.1 , tanh_val=1.0 -> 0.8
        transforms.RandomApply(
            [transforms.Lambda(clamp_transform),
            transforms.ColorJitter(brightness=0.1 + (0.6*tanh_val), contrast=0.1 + (0.6*tanh_val),
                                    saturation=0.1 + (0.6*tanh_val), hue=0.03 + (0.27*tanh_val))]
                                    , p=0.1 + (0.8*tanh_val)), # tanh_val=0.0 -> 0.1 , tanh_val=1.0 -> 0.9
        transforms.RandomGrayscale(p=0.05 + (0.45*tanh_val)), # tanh_val=0.0 -> 0.05 , tanh_val=1.0 -> 0.
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel)], p=0.1 + (0.4*tanh_val)), # tanh_val=0.0 -> 0.1 , tanh_val=1.0 -> 0.8
        transforms.RandomSolarize(threshold=128, p=0.1 + (0.4*tanh_val)), # tanh_val=0.0 -> 0.1 , tanh_val=1.0 -> 0.8

        transforms.ConvertImageDtype(torch.float),
        normalize
    ])

    return MultipleCropsTransform(augs, n_crops)

def get_dataset_crop_scale(dataset: str, scale= (0.2, 1.)):
    """Get corresponding crop transform for each dataset."""

    if dataset == 'cifar100':
        # return transforms.RandomCrop(32, padding=4),
        return transforms.RandomResizedCrop(32, scale=scale)
    elif dataset == 'cifar10':
        # return transforms.RandomCrop(32, padding=4)
        return transforms.RandomResizedCrop(32, scale=scale)
    elif dataset in ['imagenet100', 'imagenet', 'clear100']:
        # return transforms.RandomCrop(64, padding=8)
        return transforms.RandomResizedCrop(224, scale=scale)
    else:
        raise ValueError("Dataset not supported.")



def gaussian_blur_pytorch(image: torch.Tensor) -> torch.Tensor:
    """
    Apply Gaussian blur on an image using torchvision's Gaussian blur, 
    replicating the behavior of PIL's GaussianBlur.
    
    Args:
    - image (torch.Tensor): Input image tensor of shape (C, H, W).

    Returns:
    - torch.Tensor: The blurred image.
    """

    # Generate random sigma between 0.1 and 2.0 in torch
    sigma = random.random() * 1.9 + 0.1

    # Calculate kernel size (ensures it's an odd number)
    kernel_size = 2 * round(sigma) + 1

    # Apply Gaussian blur using torchvision's Gaussian blur
    return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma])



class MultipleCropsTransform:
    """Take N random augmented views of one image. With tanh value for augmentations difficulty"""
    def __init__(self, augs_to_generate, n_crops=20):
        self.augs_to_generate = augs_to_generate
        self.n_crops = n_crops

    def __call__(self, x, tanh_values):
        stacked_views = [] # List of tensors, each contains one view for all samples in x
        for _ in range(self.n_crops):
            view_list = [self.augs_to_generate(tanh_val.item())(sample) for sample, tanh_val in zip(x, tanh_values)]
            stacked_views.append(torch.stack(view_list, dim=0))
            
        return stacked_views
    

