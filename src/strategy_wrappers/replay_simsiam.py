import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import NCExperience

from .reservoir_buffer import ReservoirBufferUnlabeled
from .utils import UnsupervisedDataset, find_encoder, init_optim
from .ssl_models.simsiam import SimSiam
from .transforms import get_transforms_simsiam, get_common_transforms

class ReplaySimSiam():

    def __init__(self,
               encoder: str = 'resnet18',
               optim: str = 'SGD',
               lr: float = 5e-4,
               momentum: float = 0.9,
               weight_decay: float = 1e-4,
               dim_proj: int = 2048,
               dim_pred: int = 512,
               mem_size: int = 2000,
               replay_mb_size: int = 32,
               train_mb_size: int = 32,
               train_epochs: int = 1,
               mb_passes: int = 3,
               device = 'cpu',
               dataset_name: str = 'cifar100',
               save_pth: str  = None,
               save_model: bool = False, 
               common_transforms: bool = True):

        self.momentum = momentum
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dim_proj = dim_proj
        self.dim_pred = dim_pred
        self.mem_size = mem_size
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
        self.buffer = ReservoirBufferUnlabeled(self.mem_size)

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_common_transforms(self.dataset_name)
        else:
            self.transforms = get_transforms_simsiam(self.dataset_name)

        # Set up encoder
        self.encoder = find_encoder(encoder)

        # Set up model
        self.model = SimSiam(self.encoder, dim_proj, dim_pred).to(self.device)
        self.model_name = 'replay_simsiam'

        # Set up optimizer
        self.optimizer = init_optim(optim, self.model.parameters(), lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write hyperparameters
                f.write(f'encoder: {self.encoder}\n')
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                f.write(f'dim_proj: {self.dim_proj}\n')
                f.write(f'dim_pred: {self.dim_pred}\n')
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
                    if len(self.buffer.buffer) > self.train_mb_size:
                        # Sample from buffer and concat
                        replay_batch = self.buffer.sample(self.replay_mb_size).to(self.device)
                        combined_batch = torch.cat((replay_batch, mbatch), dim=0)
                    else:
                        # Do not sample buffer if not enough elements in it
                        combined_batch = mbatch

                    # Apply transforms
                    x1, x2 = self.transforms(combined_batch)

                    # Forward pass
                    loss = self.model(x1, x2)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                # Update buffer with new samples
                self.buffer.add(new_mbatch.detach())

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
