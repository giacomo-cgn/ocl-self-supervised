import os
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import NCExperience

from ..utils import UnsupervisedDataset, init_optim
from ..transforms import get_transforms

class AlignBuffer():

    def __init__(self,
                 model: torch.nn.Module = None,
                 buffer = None,
                 optim: str = 'SGD',
                 lr: float = 5e-4,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 train_mb_size: int = 32,
                 train_epochs: int = 1,
                 mb_passes: int = 3,
                 device = 'cpu',
                 dataset_name: str = 'cifar100',
                 save_pth: str  = None,
                 save_model: bool = False,
                 common_transforms: bool = True,
                 replay_mb_size: int = 32,
                 omega: float = 0.1,
                 align_criterion: str = 'ssl',
                 use_aligner: bool = True,
                 align_after_proj: bool = True
    ):
        if model is None:
            raise Exception(f'This strategy requires a SSL model')
        if buffer is None:
            raise Exception(f'This strategy requires a buffer')

        self.model = model
        self.buffer = buffer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.common_transforms = common_transforms
        self.replay_mb_size = replay_mb_size
        self.omega = omega
        self.align_criterion_name = align_criterion
        self.use_aligner = use_aligner
        self.align_after_proj = align_after_proj

        self.strategy_name = 'align_buffer'
        self.model_and_strategy_name = self.strategy_name + '_' + self.model.get_name()

        # Set up feature alignment criterion
        if self.align_criterion_name == 'ssl':
            self.align_criterion = self.model.get_criterion()
        elif self.align_criterion_name == 'mse':
            self.align_criterion = nn.MSELoss()

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common')
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.model.get_name())

        # Set up alignment projector (use dim_pred as hidden layer dim)
        dim_align_layer = self.model.get_predictor_dim()
        if self.align_after_proj:
            dim_proj = self.model.get_projector_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_proj, dim_align_layer, bias=False),
                                                nn.BatchNorm1d(dim_align_layer),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim_align_layer, dim_proj)).to(self.device)
        else:
            dim_encoder_embed = self.model.get_embedding_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_encoder_embed, dim_align_layer, bias=False),
                                                nn.BatchNorm1d(dim_align_layer),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim_align_layer, dim_encoder_embed)).to(self.device)
        

        # Set up optimizer
        params_to_optimize = list(self.model.parameters()) + list(self.alignment_projector.parameters())
        self.optimizer = init_optim(optim, params_to_optimize, lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'optim: {optim}\n') 
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'optim-momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')
                f.write(f'train_epochs: {self.train_epochs}\n')
                f.write(f'mb_passes: {self.mb_passes}\n')
                f.write(f'replay_mb_size: {self.replay_mb_size}\n')
                f.write(f'omega: {self.omega}\n')
                f.write(f'align_criterion: {self.align_criterion_name}\n')
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'align_after_proj: {self.align_after_proj}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')
                
                # Write memory columns
                with open(os.path.join(self.save_pth, 'gpu_memory.csv'), 'a') as f:
                    f.write('cuda.memory_allocated,cuda.memory_reserved,cuda.max_memory_reserved,exp_idx,epoch,mb_idx,mb_pass\n')




    def train_experience(self, 
                         experience: NCExperience,
                         exp_idx: int
                         ):
        # Prepare data
        exp_data = UnsupervisedDataset(experience.dataset)  
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True)

        self.model.train()
        
        for epoch in range(self.train_epochs):
            for mb_idx, mbatch in tqdm(enumerate(data_loader)):
                mbatch = mbatch.to(self.device)
                new_mbatch = copy.deepcopy(mbatch)
                new_mbatch_size = len(new_mbatch)

                for k in range(self.mb_passes):
                    if len(self.buffer.buffer) > self.replay_mb_size:
                        use_replay = True
                        # Sample from buffer and concat
                        replay_batch, replay_z_old, replay_indices = self.buffer.sample(self.replay_mb_size)
                        replay_batch, replay_z_old = replay_batch.to(self.device), replay_z_old.to(self.device)
                        combined_batch = torch.cat((replay_batch, mbatch), dim=0)
                    else:
                        # Do not sample buffer if not enough elements in it
                        use_replay = False
                        combined_batch = mbatch

                    # Apply transforms
                    x1, x2 = self.transforms(combined_batch)

                    # Forward pass (z is after projector, e before projector)
                    loss, z1, z2, e1, e2 = self.model(x1, x2)

                    if not self.align_after_proj:
                        # Use encoder features instead projector features
                        z1 = e1
                        z2 = e2

                    if use_replay:
                        # Take only embed features from replay batch
                        replay_z_new_1 = z1[:self.replay_mb_size]
                        replay_z_new_2 = z2[:self.replay_mb_size]

                        if self.use_aligner:
                            # Align features after aligner
                            aligned_features_1 = self.alignment_projector(replay_z_new_1)
                            aligned_features_2 = self.alignment_projector(replay_z_new_2)
                        else:
                            # Do not use aligner
                            aligned_features_1 = replay_z_new_1
                            aligned_features_2 = replay_z_new_2

                        loss_align = 0.5*self.align_criterion(aligned_features_1, replay_z_old) + 0.5*self.align_criterion(aligned_features_2, replay_z_old)

                        loss += self.omega * loss_align.mean()

                        # Update replayed samples with avg of last extracted features
                        self.buffer.update_features(((replay_z_new_1+replay_z_new_2)/2).detach(), replay_indices)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.model.after_backward()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')
                        with open(os.path.join(self.save_pth, 'gpu_memory.csv'), 'a') as f:
                            f.write(f'{torch.cuda.memory_allocated(self.device)/1024/1024/1024}GB,{torch.cuda.memory_reserved(self.device)/1024/1024/1024}GB,{torch.cuda.max_memory_reserved(self.device)/1024/1024/1024}GB,{exp_idx},{epoch},{mb_idx},{k}\n')


                # Update buffer with new samples
                self.buffer.add(new_mbatch.detach(), z1[-new_mbatch_size:].detach()) # Use only first view features, as augmentations are random

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
