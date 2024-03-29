import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import NCExperience

from ..utils import UnsupervisedDataset, init_optim
from ..transforms import get_transforms

class LUMP():

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
                 alpha_lump: float = 0.4,
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
        self.alpha_lump = alpha_lump

        self.strategy_name = 'lump'
        self.model_and_strategy_name = self.strategy_name + '_' + self.model.get_name()

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common')
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.model.get_name())

        # Set up optimizer
        self.optimizer = init_optim(optim, self.model.parameters(), lr=self.lr,
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
                f.write(f'alpha_lump: {self.alpha_lump}\n')


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
                curr_mbatch_size = new_mbatch.shape[0]

                for k in range(self.mb_passes):
                    lambd = np.random.beta(self.alpha_lump, self.alpha_lump)

                    # Augment stream minibatch
                    x1, x2 = self.transforms(new_mbatch)

                    if len(self.buffer.buffer) > curr_mbatch_size:
                        # Apply mixup
                        replay_batch, _, _ = self.buffer.sample(curr_mbatch_size) # Sample same number of current mb size
                        replay_batch = replay_batch.to(self.device)
                        # Augment replay minibatch
                        x1_replay, x2_replay = self.transforms(replay_batch)

                        mixed_x1 = lambd * x1 + (1 - lambd) * x1_replay
                        mixed_x2 = lambd * x2 + (1 - lambd) * x2_replay
                    else:
                        # No mixup
                        mixed_x1 = x1
                        mixed_x2 = x2

                    # Forward pass
                    loss, z1, z2, _, _ = self.model(mixed_x1, mixed_x2)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')
                
                    self.model.after_backward()

                # Update buffer with new samples
                self.buffer.add(new_mbatch.detach(), z1.detach())

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
