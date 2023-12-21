import os
import pandas as pd
import argparse

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

@torch.no_grad() 
def update_ema_params(model_params, ema_model_params, momentum):
    for po, pm in zip(model_params, ema_model_params):
            pm.data.mul_(momentum).add_(po.data, alpha=(1. - momentum))
    


def write_final_scores(folder_input_path, output_file):
    """
    Report final aggregated scores of the probing

    """
    # output_file = os.path.join(folder_path, "final_scores.csv")
    with open(output_file, "w") as output_f:
        # Write header
        output_f.write("probe_ratio,avg_val_acc,avg_test_acc\n")

        # Get all subfolder paths starting with "probing_ratio"
        probing_ratios_subfolders = [os.path.join(folder_input_path, f) for f in os.listdir(folder_input_path) 
                                    if os.path.isdir(os.path.join(folder_input_path, f)) and f.startswith("probing_ratio")]

        # For each probing tr ratio
        for subfolder in probing_ratios_subfolders:
            probing_tr_ratio = subfolder.split("probing_ratio")[1]
            probe_exp_df_list = [] # List of tuples (Dataframe, exp_index)

            # Read all csv, one for each experience on which probing has been executed
            for file in os.listdir(subfolder):
                if file.endswith('.csv'):
                    probe_exp = int(file.split('.csv')[0].split('probe_exp_')[-1]) # Finds exp_idx from filename
                    df = pd.read_csv(os.path.join(subfolder, file))
                    probe_exp_df_list.append((df, probe_exp))

            # Find df with highest exp_index in probe_exp_df_list
            final_df = max(probe_exp_df_list, key=lambda x: x[1])[0]
            # Get final test and validation accuracies
            final_avg_test_acc =  final_df['test_acc'].mean()
            final_avg_val_acc = final_df['val_acc'].mean()


            output_f.write(f"{probing_tr_ratio},{final_avg_val_acc},{final_avg_test_acc}\n")


def read_command_line_args():
    """
    Parses command line arguments
    """
    def str_to_bool(s):
        if s.lower() in ('true', 't', 'yes', 'y', '1'):
            return True
        elif s.lower() in ('false', 'f', 'no', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='no_strategy')
    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--optim-momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num-exps', type=int, default=20)
    parser.add_argument('--save-folder', type=str, default='./logs')
    parser.add_argument('--dim-proj', type=int, default=2048)
    parser.add_argument('--dim-pred', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--probing-separate', type=str_to_bool, default=True)
    parser.add_argument('--probing-upto', type=str_to_bool, default=True)
    parser.add_argument('--probing-val-ratio', type=float, default=0.1)
    parser.add_argument('--omega', type=float, default=0.5)
    parser.add_argument('--momentum-ema', type=float, default=0.999)
    parser.add_argument('--ema-align-proj', type=str_to_bool, default=True)
    parser.add_argument('--align-criterion', type=str, default='ssl')
    parser.add_argument('--mem-size', type=int, default=2000)
    parser.add_argument('--mb-passes', type=int, default=3)
    parser.add_argument('--tr-mb-size', type=int, default=32)
    parser.add_argument('--repl-mb-size', type=int, default=32)
    parser.add_argument('--eval-mb-size', type=int, default=1024)
    parser.add_argument('--common-transforms', type=str_to_bool, default=True)
    parser.add_argument('--use-probing-tr-ratios', type=str_to_bool, default=False)
    parser.add_argument('--iid', type=str_to_bool, default=False)
    parser.add_argument('--save-model-final', type=str_to_bool, default=True)

    # Models specific params
    parser.add_argument('--lambd', type=float, default=5e-3)
    parser.add_argument('--byol-momentum', type=float, default=0.9)
    parser.add_argument('--return-momentum-encoder', type=bool, default=True)


    args = parser.parse_args()

    return args







