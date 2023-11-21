import os
import pandas as pd

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
    


def write_final_scores(folder_path):
    """
    Report final aggregated scores of the probing

    """
    output_file = os.path.join(folder_path, "final_scores.csv")
    with open(output_file, "w") as output_f:
        # Write header
        output_f.write("probe_ratio,avg_val_acc,avg_test_acc\n")

        # Get all subfolder paths starting with "probing_ratio"
        probing_ratios_subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                                    if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("probing_ratio")]
        
        # For each probing tr ratio
        for subfolder in probing_ratios_subfolders:
            probing_tr_ratio = subfolder.split("probing_ratio")[1]
            probe_df_list = []

            # Read all csv, one for each experience on which probing has been executed
            for file in os.listdir(subfolder):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(subfolder, file))
                    probe_df_list.append(df)

            # Get final test and validation accuracies
            final_df = probe_df_list[-1]
            final_avg_test_acc =  final_df['test_acc'].mean()
            final_avg_val_acc = final_df['val_acc'].mean()

            output_f.write(f"{probing_tr_ratio}, {final_avg_val_acc}, {final_avg_test_acc}\n")






