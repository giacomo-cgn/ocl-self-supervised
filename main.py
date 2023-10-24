from avalanche.benchmarks.classic import SplitCIFAR100

import torch
import os
import datetime
import tqdm as tqdm

from torchvision.models import resnet18

from src.replay_simsiam import ReplaySimSiam
from src.transforms import get_dataset_transforms
from src.probing import LinearProbing

# Configs TODO: parse args for these
dataset_name = 'cifar100'
model_name = 'replay_simsiam'
n_experiences = 20
save_folder = './logs'
probing_epochs = 1

# Save path
str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
folder_name = f'{model_name}_{dataset_name}_{str_now}'
save_pth = os.path.join(save_folder, folder_name)
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

probing_pth = os.path.join(save_pth, 'probing')
if not os.path.exists(probing_pth):
    os.makedirs(probing_pth)

# Dataset
first_exp_with_half_classes = False
return_task_id = False
shuffle = True
class_ids_from_zero_in_each_exp = False
class_ids_from_zero_from_first_exp = True
use_transforms = True
num_classes = 100

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
model = ReplaySimSiam(device=device, save_pth=save_pth)


# Self supervised training over the experiences
for exp_idx, experience in enumerate(benchmark.train_stream):
    print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
    network = model.train_experience(experience, exp_idx)

    # Do linear probing on current encoder for all experiences (past, current and future)
    for probe_exp_idx, probe_tr_experience in enumerate(benchmark.train_stream):
        probe_save_file = os.path.join(probing_pth, f'probe_exp_{exp_idx}.csv')

        dim_features = network.projector[0].weight.shape[1]
        probe = LinearProbing(network.encoder, dim_features=dim_features, num_classes=num_classes,
                               device=device, save_file=probe_save_file, test_every_epoch=True, exp_idx=probe_exp_idx)
        print(f'-- Probing on experience: {probe_exp_idx} --')
        train_loss, train_accuracy, test_accuracy = probe.probe(
             probe_tr_experience, benchmark.test_stream[probe_exp_idx], num_epochs=probing_epochs)


 


