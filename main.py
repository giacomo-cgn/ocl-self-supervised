from avalanche.benchmarks.classic import SplitCIFAR100

import torch
import os
import datetime
import tqdm as tqdm

from src.replay_simsiam import ReplaySimSiam
from src.replay_barlow_twins import ReplayBarlowTwins
from src.no_strategy_simsiam import NoStrategySimSiam
from src.no_strategy_barlow_twins import NoStrategyBarlowTwins
from src.no_strategy_byol import NoStrategyBYOL
from src.replay_byol import ReplayBYOL

from src.transforms import get_dataset_transforms
from src.probing import LinearProbing

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='no_strategy_simsiam')
parser.add_argument('--encoder', type=str, default='resnet18')
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--num-exps', type=int, default=20)
parser.add_argument('--save-folder', type=str, default='./logs')
parser.add_argument('--probing-epochs', type=int, default=1)
parser.add_argument('--mem-size', type=int, default=2000)
parser.add_argument('--mb-passes', type=int, default=3)
parser.add_argument('--tr-mb-size', type=int, default=32)
parser.add_argument('--repl-mb-size', type=int, default=32)
parser.add_argument('--common-transforms', type=bool, default=True)
parser.add_argument('--use-probing-tr-ratios', type=bool, default=True)
parser.add_argument('-iid', '--iid', type=bool, default=False)
# Models specific params
parser.add_argument('--lambd', type=float, default=5e-3)
parser.add_argument('--byol-momentum', type=float, default=0.9)
parser.add_argument('--return-momentum-encoder', type=bool, default=True)


args = parser.parse_args()

# Ratios of tr set used for training linear probe
if args.use_probing_tr_ratios:
     probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
else:
     probing_tr_ratio_arr = [1]


# Set up save folders
str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
folder_name = f'{args.model}_{args.dataset}_{str_now}'
if args.iid:
    folder_name = 'iid_' + folder_name
save_pth = os.path.join(args.save_folder, folder_name)
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

probing_pth_dict = {}
for probing_tr_ratio in probing_tr_ratio_arr:  
    probing_pth = os.path.join(save_pth, f'probing_ratio{probing_tr_ratio}')
    if not os.path.exists(probing_pth):
        os.makedirs(probing_pth)
    probing_pth_dict[probing_tr_ratio] = probing_pth

# Save general args
with open(save_pth + '/config.txt', 'a') as f:
    f.write(f'Model: {args.model}\n')
    f.write(f'Encoder: {args.encoder}\n')
    f.write(f'Optimizer: {args.optim}\n')
    f.write(f'Dataset: {args.dataset}\n')
    f.write(f'Number of Experiences: {args.num_exps}\n')
    f.write(f'Probing Epochs: {args.probing_epochs}\n')
    f.write(f'Memory Size: {args.mem_size}\n')
    f.write(f'MB Passes: {args.mb_passes}\n')
    f.write(f'Train MB Size: {args.tr_mb_size}\n')
    f.write(f'Replay MB Size: {args.repl_mb_size}\n')
    f.write(f'Use Common Transforms: {args.common_transforms}\n')
    f.write(f'Use Probing Train Ratios: {args.use_probing_tr_ratios}\n')
    f.write(f'IID pretraining: {args.iid}\n')

# Dataset
first_exp_with_half_classes = False
return_task_id = False
shuffle = True
use_transforms = True
num_classes = 100
probe_benchmark = SplitCIFAR100(
            args.num_exps,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            shuffle=shuffle,
            train_transform=get_dataset_transforms(args.dataset),
            eval_transform=get_dataset_transforms(args.dataset),
        )
if args.iid:
    # If pretraining iid, create benchmark with only 1 experience
    pretr_benchmark  = SplitCIFAR100(
            1,
            first_exp_with_half_classes=first_exp_with_half_classes,
            return_task_id=return_task_id,
            shuffle=shuffle,
            train_transform=get_dataset_transforms(args.dataset),
            eval_transform=get_dataset_transforms(args.dataset),
    )
else:
     # Use same benchmark for pretraining and probing
     pretr_benchmark = probe_benchmark
         

# Device
if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Model
if args.model == 'no_strategy_simsiam':
     model = NoStrategySimSiam(encoder=args.encoder, optim=args.optim,
                               train_mb_size=args.tr_mb_size, mb_passes=args.mb_passes,
                               dataset_name=args.dataset, save_pth=save_pth, device=device,
                               save_model=False, common_transforms=args.common_transforms)
elif args.model == 'replay_simsiam':
     model = ReplaySimSiam(encoder=args.encoder, optim=args.optim, mem_size=args.mem_size,
                           train_mb_size=args.tr_mb_size, replay_mb_size=args.repl_mb_size,
                           mb_passes=args.mb_passes, dataset_name=args.dataset, save_pth=save_pth,
                           device=device, save_model=False, common_transforms=args.common_transforms)
     
elif args.model == 'replay_barlow_twins':
     model = ReplayBarlowTwins(lambd=args.lambd, encoder=args.encoder, optim=args.optim,
                               mem_size=args.mem_size, train_mb_size=args.tr_mb_size,
                               mb_passes=args.mb_passes, dataset_name=args.dataset, save_pth=save_pth,
                               device=device, save_model=False, common_transforms=args.common_transforms)
elif args.model == 'no_strategy_barlow_twins':
     model = NoStrategyBarlowTwins(lambd=args.lambd, encoder=args.encoder, optim=args.optim,
                                   train_mb_size=args.tr_mb_size,
                                   mb_passes=args.mb_passes, dataset_name=args.dataset, save_pth=save_pth,
                                   device=device, save_model=False, common_transforms=args.common_transforms)
elif args.model == 'no_strategy_byol':
     model =  NoStrategyBYOL(byol_momentum=args.byol_momentum, return_momentum_encoder=args.return_momentum_encoder,
                            encoder=args.encoder, optim=args.optim,
                            train_mb_size=args.tr_mb_size,
                            mb_passes=args.mb_passes, dataset_name=args.dataset, save_pth=save_pth,
                            device=device, save_model=False, common_transforms=args.common_transforms)
     
elif args.model == 'replay_byol':
     model = ReplayBYOL(byol_momentum=args.byol_momentum, return_momentum_encoder=args.return_momentum_encoder,
                               encoder=args.encoder, optim=args.optim, mem_size=args.mem_size,
                               train_mb_size=args.tr_mb_size,
                               mb_passes=args.mb_passes, dataset_name=args.dataset, save_pth=save_pth,
                               device=device, save_model=False, common_transforms=args.common_transforms)
     
else:
     # Throw exception
     raise Exception(f"Model {args.model} not supported")

# Self supervised training over the experiences
for exp_idx, experience in enumerate(pretr_benchmark.train_stream):
    print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
    network = model.train_experience(experience, exp_idx)

    # Do linear probing on current encoder for all experiences (past, current and future)
    for probe_exp_idx, probe_tr_experience in enumerate(probe_benchmark.train_stream):

        # Sample only a portion of the tr samples for probing
        for probing_tr_ratio in probing_tr_ratio_arr:

            probe_save_file = os.path.join(probing_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')
            dim_features = network.get_embedding_dim() 

            probe = LinearProbing(network.get_encoder(), dim_features=dim_features, num_classes=num_classes,
                                device=device, save_file=probe_save_file, test_every_epoch=True,
                                exp_idx=probe_exp_idx, tr_samples_ratio=probing_tr_ratio)
            
            print(f'-- Probing on experience: {probe_exp_idx}, probe tr ratio: {probing_tr_ratio} --')

            train_loss, train_accuracy, test_accuracy = probe.probe(
                probe_tr_experience, probe_benchmark.test_stream[probe_exp_idx], num_epochs=args.probing_epochs)
