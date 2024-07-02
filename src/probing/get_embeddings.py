import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Tuple


def get_embeddings(encoder: nn.Module,
                   dataset: Dataset,
                   train_data: bool,
                   mb_size: int = 256,
                   device: str = 'cpu',
                   )-> Dataset:

    train_loader = DataLoader(dataset=dataset, batch_size=mb_size, shuffle=train_data)

    with torch.no_grad():
        # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
        encoder.eval()

        # Get encoder activations for tr dataloader
        activations_list = []
        labels_list = []
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            activations_list.append(activations.detach().cpu())
            labels_list.append(labels.detach().cpu())
        activations = torch.cat(activations_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return Dataset(activations, labels)