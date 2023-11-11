import torch
from torch.utils.data import Dataset
from torchvision import models

from .optims import LARS


# Convert Avalanche dataset with labels and task labels to Pytorch dataset with only input tensors
class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, _, _ = self.data[idx]
        return input_tensor
    

def find_encoder(encoder_name: str):
    if encoder_name == 'resnet18':
        return models.resnet18
    elif encoder_name == 'resnet34':
        return models.resnet34
    elif encoder_name == 'resnet50':
        return models.resnet50
    else:
        raise ValueError(f"Invalid encoder {encoder_name}")

    
    
def init_optim(optim_name, params, lr, momentum, weight_decay):
    if optim_name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'LARS':
        return LARS(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer {optim_name}")