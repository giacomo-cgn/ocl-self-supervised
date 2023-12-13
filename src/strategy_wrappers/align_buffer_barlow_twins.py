import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import NCExperience

from ..reservoir_buffer import ReservoirBufferUnlabeledFeatures
from ..utils import UnsupervisedDataset, find_encoder, init_optim
from ..ssl_models.barlow_twins import BarlowTwins
from ..transforms import get_transforms_barlow_twins, get_common_transforms

class AlignBufferBarlowTwins():

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
        self.replay_mb_size = replay_mb_size
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.common_transforms = common_transforms

        # Set up buffer
        self.buffer = ReservoirBufferUnlabeledFeatures(self.mem_size)

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_common_transforms(self.dataset_name)
        else:
            self.transforms = get_transforms_barlow_twins(self.dataset_name)

        # Set up encoder
        self.encoder = find_encoder(encoder)

        # Set up model
        self.model = BarlowTwins(self.encoder, dim_proj, self.lambd).to(self.device)
        self.model_name = 'align_buffer_barlow_twins'

        # Set up optimizer
        self.optimizer = init_optim(optim, self.model.parameters(), lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)
        
        # Set up alignment projector (use dim_pred as hidden layer dim)
        self.alignment_projector = nn.Sequential(nn.Linear(self.dim_proj, self.dim_pred, bias=False),
                                                nn.BatchNorm1d(self.dim_pred),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.dim_pred, self.dim_proj)).to(self.device)
        


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
                    if len(self.buffer.buffer) > self.replay_mb_size:
                        use_replay = True
                        # Sample from buffer and concat
                        replay_batch, replay_z_old = self.buffer.sample(self.replay_mb_size)
                        replay_batch, replay_z_old = replay_batch.to(self.device), replay_z_old.to(self.device)
                        combined_batch = torch.cat((replay_batch, mbatch), dim=0)
                    else:
                        # Do not sample buffer if not enough elements in it
                        use_replay = False
                        combined_batch = mbatch

                    # Apply transforms
                    x1, x2 = self.transforms(combined_batch)

                    # Forward pass
                    loss, z1, z2, _, _ = self.model(x1, x2)

                    if use_replay:
                        # Take only embed features from replay batch
                        replay_z_new_1 = z1[-self.replay_mb_size:]
                        replay_z_new_2 = z2[-self.replay_mb_size:]

                        aligned_features_1 = self.alignment_projector(replay_z_new_1)
                        aligned_features_2 = self.alignment_projector(replay_z_new_2)

                        barlow_twins_loss = self.model.get_criterion()

                        loss_align = 0.5*barlow_twins_loss(aligned_features_1, replay_z_old) + 0.5*barlow_twins_loss(aligned_features_2, replay_z_old)

                        loss += self.omega * loss_align

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                # Update buffer with new samples
                self.buffer.add(new_mbatch.detach(), z1[:self.train_mb_size].detach()) # Use only first view features, as augmentations are random

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
