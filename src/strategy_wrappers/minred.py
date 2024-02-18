import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import NCExperience

from ..utils import UnsupervisedDataset, init_optim
from ..transforms import get_transforms

class MinRed():

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

        self.strategy_name = 'minred'
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
                f.write(f'replay_mb_size: {self.replay_mb_size}\n')

                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')

                with open(os.path.join(self.save_pth, 'not_none_minred_features.csv'), 'a') as f:
                    f.write('not_none_ratio,exp_idx,epoch,mb_idx,mb_pass\n')


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
                # Forward pass on stream minibatch to get features
                with torch.no_grad():
                    e_mbatch = self.model.get_encoder()(mbatch.detach())
                    z_mbatch = self.model.get_projector()(e_mbatch)

                # Add stream minibatch and features to buffer
                self.buffer.add(mbatch.detach(), z_mbatch.detach())

                for k in range(self.mb_passes):
                    if len(self.buffer.buffer) > 0:
                        # Sample from buffer (indices needed for buffer features update)
                        replay_batch_size = min(self.replay_mb_size, len(self.buffer.buffer))
                        replay_batch, _, replay_indices = self.buffer.sample(replay_batch_size)
                        replay_batch = replay_batch.to(self.device)

                        # Apply transforms
                        x1, x2 = self.transforms(replay_batch)

                        # Forward pass
                        loss, z1, z2, _, _ = self.model(x1, x2)

                        # Update replayed samples with avg of last extracted features
                        self.buffer.update_features(((z1+z2)/2).detach(), replay_indices)

                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Save loss, exp_idx, epoch, mb_idx and k in csv
                        if self.save_pth is not None:
                            with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                                f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                            not_none_ratio = len([idx for idx, value in enumerate(self.buffer.buffer_features) 
                                                  if value is not None])/self.buffer.stored_samples
                            with open(os.path.join(self.save_pth, 'not_none_minred_features.csv'), 'a') as f:
                                f.write(f'{not_none_ratio},{exp_idx},{epoch},{mb_idx},{k}\n')
                    
                        self.model.after_backward()


        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.model
