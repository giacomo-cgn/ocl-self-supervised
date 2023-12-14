import os
from tqdm import tqdm
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from avalanche.benchmarks.scenarios import NCExperience

from ..reservoir_buffer import ReservoirBufferUnlabeled
from ..utils import UnsupervisedDataset, find_encoder, init_optim, update_ema_params
from ..ssl_models.barlow_twins import BarlowTwins
from ..transforms import get_transforms_barlow_twins, get_common_transforms

class AlignEMABarlowTwins():

    def __init__(self,
               encoder: str = 'resnet18',
               optim: str = 'SGD',
               lr: float = 5e-4,
               momentum: float = 0.9,
               weight_decay: float = 1e-4,
               lambd: float = 5e-3,
               dim_proj: int = 2048,
               dim_pred: int = 512,
               mem_size: int = 2000,
               omega: float = 0.5,
               momentum_ema: float = 0.999,
               use_replay: bool = True,
               align_after_proj: bool = True,
               use_mse_align: bool = True,
               replay_mb_size: int = 32,
               train_mb_size: int = 32,
               train_epochs: int = 1,
               mb_passes: int = 3,
               device = 'cpu',
               dataset_name: str = 'cifar100',
               save_pth: str  = None,
               save_model: bool = False, 
               common_transforms: bool = True):
        
        self.lambd = lambd
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dim_proj = dim_proj
        self.dim_pred = dim_pred
        self.mem_size = mem_size
        self.omega = omega
        self.momentum_ema = momentum_ema
        self.use_replay = use_replay
        self.align_after_proj = align_after_proj
        self.use_mse_align = use_mse_align
        self.replay_mb_size = replay_mb_size
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.common_transforms = common_transforms

        if self.use_replay:
            # Set up buffer
            self.buffer = ReservoirBufferUnlabeled(self.mem_size)

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_common_transforms(self.dataset_name)
        else:
            self.transforms = get_transforms_barlow_twins(self.dataset_name)

        # Set up encoder
        self.encoder = find_encoder(encoder)

        # Set up model
        self.model = BarlowTwins(self.encoder, dim_proj, self.lambd).to(self.device)
        self.model_name = 'align_ema_barlow_twins'

        # Set up optimizer
        self.optimizer = init_optim(optim, self.model.parameters(), lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)
        
        # Set up EMA model that is targeted for alignment. It is the EMA of encoder+projector
        self.ema_encoder = copy.deepcopy(self.model.get_encoder())
        self.ema_projector = copy.deepcopy(self.model.get_projector())

        # Stop gradient in EMA model
        self.ema_encoder.requires_grad_(False)
        self.ema_projector.requires_grad_(False)

        # Set up alignment projector (use dim_pred as hidden layer dim)
        if self.align_after_proj:
            self.alignment_projector = nn.Sequential(nn.Linear(self.dim_proj, self.dim_pred, bias=False),
                                                nn.BatchNorm1d(self.dim_pred),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.dim_pred, self.dim_proj)).to(self.device)
        else:
            dim_encoder_embed = self.model.get_embedding_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_encoder_embed, self.dim_pred, bias=False),
                                                nn.BatchNorm1d(self.dim_pred),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.dim_pred, dim_encoder_embed)).to(self.device)
        

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write hyperparameters
                f.write(f'encoder: {self.encoder}\n')
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'Lambda Barlow Twins: {self.lambd}\n')
                f.write(f'momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                f.write(f'dim_proj: {self.dim_proj}\n')
                f.write(f'dim_pred: {self.dim_pred}\n')
                f.write(f'Omega: {self.omega}\n')
                f.write(f'momentum_ema: {self.momentum_ema}\n')
                f.write(f'use_replay in align to ema: {self.use_replay}\n')
                f.write(f'align after projector: {self.align_after_proj}\n')
                f.write(f'mem_size: {self.mem_size}\n')
                f.write(f'replay_mb_size: {self.replay_mb_size}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')
                f.write(f'train_epochs: {self.train_epochs}\n')
                f.write(f'mb_passes: {self.mb_passes}\n')
                f.write(f'device: {self.device}\n')
                f.write(f'dataset_name: {self.dataset_name}\n')

                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')


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
                new_mbatch = mbatch

                for k in range(self.mb_passes):
                    if self.use_replay and len(self.buffer.buffer) > self.replay_mb_size:
                        # Sample from buffer and concat
                        replay_batch = self.buffer.sample(self.replay_mb_size).to(self.device)
                        combined_batch = torch.cat((replay_batch, mbatch), dim=0)
                    else:
                        # Do not sample buffer if not enough elements in it
                        combined_batch = mbatch

                    # Apply transforms
                    x1, x2 = self.transforms(combined_batch)

                    # Forward pass
                    loss, z1, z2, e1, e2 = self.model(x1, x2)

                    # EMA model pass
                    with torch.no_grad():
                        ema_e1 = self.ema_encoder(x1)
                        ema_e2 = self.ema_encoder(x2)
                        ema_z1 = self.ema_projector(e1)
                        ema_z2 = self.ema_projector(e2)

                    if self.use_mse_align:
                        align_criterion = mse_loss
                    else:
                        align_criterion = self.model.get_criterion()
                    
                    if self.align_after_proj:
                        # Align features
                        aligned_features_1 = self.alignment_projector(z1)
                        aligned_features_2 = self.alignment_projector(z2)
                        loss_align = 0.5*align_criterion(aligned_features_1, ema_z1) + 0.5*align_criterion(aligned_features_2, ema_z2)
                    else:
                        # Align features
                        aligned_features_1 = self.alignment_projector(e1)
                        aligned_features_2 = self.alignment_projector(e2)
                        loss_align = 0.5*align_criterion(aligned_features_1, ema_e1) + 0.5*align_criterion(aligned_features_2, ema_e2)

                    loss += self.omega * loss_align.mean()

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                    # Update EMA model
                    update_ema_params(self.model.get_encoder().parameters(),
                                       self.ema_encoder.parameters(), self.momentum_ema)
                    update_ema_params(self.model.get_projector().parameters(),
                                      self.ema_projector.parameters(), self.momentum_ema)
                
                if self.use_replay:
                    # Update buffer with new samples
                    self.buffer.add(new_mbatch.detach()) # Use only first view features, as augmentations are random

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
