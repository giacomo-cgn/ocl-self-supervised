import datetime
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models

from avalanche.benchmarks.scenarios import NCExperience

from .reservoir_buffer import ReservoirBufferUnlabeled
from .utilities import UnsupervisedDataset
from .simsiam import SimSiam
from .transforms import get_transforms_simsiam

class ReplaySimSiam():

    def __init__(self,
               encoder: str = 'resnet18',
               lr: float = 0.001,
               momentum: float = 0.9,
               weight_decay: float = 1e-4,
               dim_features: int = 2048,
               dim_pred: int = 512,
               mem_size: int = 2000,
               replay_mb_size: int = 32,
               train_mb_size: int = 32,
               train_epochs: int = 1,
               mb_passes: int = 5,
               device = 'cpu',
               dataset_name: str = 'cifar100',
               save_folder: str  = None,
               save_model: bool = False):

        if replay_mb_size is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = replay_mb_size

        self.momentum = momentum
        self.lr = lr
        self.momentum = momentum
        self.weigth_decay = weight_decay
        self.dim_features = dim_features
        self.dim_pred = dim_pred
        self.mem_size = mem_size
        self.replay_mb_size = replay_mb_size
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_model = save_model

        # Set up buffer
        self.buffer = ReservoirBufferUnlabeled(self.mem_size)

        # Set up transforms
        self.transforms = get_transforms_simsiam(self.dataset_name)

        # Set up encoder
        if encoder == 'resnet18':
            self.encoder = models.resnet18
        if encoder == 'resnet34':
            self.encoder = models.resnet34
        if encoder == 'resnet50':
            self.encoder = models.resnet50

        # Set up model
        self.model = SimSiam(self.encoder, dim_features, dim_pred).to(self.device)
        self.model_name = 'replay_simsiam'

        # Set up optmizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weigth_decay)
        
        # Set up criterion
        self.criterion = nn.CosineSimilarity(dim=1)


        # Set save subfolder
        if save_folder is not None:
            str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
            folder_name = f'{self.model_name}_{self.dataset_name}_{str_now}'

            self.save_pth = os.path.join(save_folder, folder_name)
            if not os.path.exists(self.save_pth):
                os.makedirs(self.save_pth)


            # Save model configuration
            with open(self.save_pth + '/config.txt', 'w') as f:
                # Write hyperparameters
                f.write(f'encoder: {self.encoder}\n')
                f.write(f'lr: {self.lr}\n')
                f.write(f'momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weigth_decay}\n')
                f.write(f'dim_features: {self.dim_features}\n')
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
                    f.write('loss, exp_idx, epoch, mb_idx, mb_pass\n')

        else:
            self.save_pth = None


    def train_experience(self, 
                         experience: NCExperience,
                         exp_idx: int
                         ):
        # Prepare data
        exp_data = UnsupervisedDataset(experience.dataset)  
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True)
        
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
                    p1, p2, z1, z2 = self.model(x1, x2)

                    loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()}, {exp_idx}, {epoch}, {mb_idx}, {k}\n')

                # Update buffer with new samples
                self.buffer.add(new_mbatch.detach())

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model

                


                


