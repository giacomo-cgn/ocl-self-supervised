import torch
from torch.utils.data import Dataset

# Convert Avalanche dataset with labels and task labels to Pytorch dataset with only input tensors
class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, _, _ = self.data[idx]
        return input_tensor 