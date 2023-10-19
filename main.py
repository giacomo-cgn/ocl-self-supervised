from avalanche.benchmarks.classic import SplitCIFAR100

import torch
import os
import tqdm as tqdm

from torchvision.models import resnet18

from src.replay_simsiam import ReplaySimSiam
from src.transforms import get_dataset_transforms

save_folder = './logs'

# Dataset
dataset_name = 'cifar100'
n_experiences = 20
first_exp_with_half_classes = False
return_task_id = False
shuffle = True
class_ids_from_zero_in_each_exp = False
class_ids_from_zero_from_first_exp = True
use_transforms = True

benchmark = SplitCIFAR100(
            n_experiences,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=get_dataset_transforms(dataset_name),
            eval_transform=get_dataset_transforms(dataset_name),
        )

# Device
if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Model
model = ReplaySimSiam(device=device, save_folder=save_folder)

# Self supervised training over the experiences
for exp_idx, experience in enumerate(benchmark.train_stream):
    print('Beginning self supervised training for experience:', exp_idx)
    network = model.train_experience(experience, exp_idx)      


