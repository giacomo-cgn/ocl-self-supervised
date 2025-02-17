
from torchvision import transforms
import torchvision.transforms.functional as F
import torch

from .transforms import get_dataset_normalize

def get_randaug_transforms(dataset_name: str = 'cifar100', n_crops: int = 2):

    normalize = get_dataset_normalize(dataset_name)


    augs = lambda tanh_val : transforms.Compose([
        transforms.RandAugment(num_ops=round(2 + (tanh_val*6)), magnitude=int(30*tanh_val), num_magnitude_bins=31),
        transforms.ConvertImageDtype(torch.float),
        normalize
    ])

    return MultipleCropsTransform(augs, n_crops)


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