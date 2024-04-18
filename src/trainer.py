import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .utils import UnsupervisedDataset, init_optim
from .transforms import get_transforms
from .ssl_models import AbstractSSLModel
from .strategies import AbstractStrategy


class Trainer():

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 strategy: AbstractStrategy = None,
                 optim: str = 'SGD',
                 lr: float = 0.01,
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
                 num_views: int = 2
               ):
        
        if ssl_model is None:
            raise Exception(f'A SSL model is requred')            

        self.ssl_model = ssl_model
        self.strategy = strategy
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
        self.num_views = num_views # == 2 for most Instance Discrimination methods, but can vary e.g. EMP

        self.model_and_strategy_name = self.strategy.get_name() + '_' + self.ssl_model.get_name()

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common', n_crops=num_views)
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.ssl_model.get_name())

        # List of params to optimize
        params_to_optimize = self.ssl_model.get_params() + self.strategy.get_params()

        # Set up optimizer
        self.optimizer = init_optim(optim, params_to_optimize, lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- TRAINER CONFIG ----\n')
                f.write(f'optim: {optim}\n') 
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'optim-momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                f.write(f'num_views: {self.num_views}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')
                f.write(f'train_epochs: {self.train_epochs}\n')
                f.write(f'mb_passes: {self.mb_passes}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')


    def train_experience(self, 
                         dataset,
                         exp_idx: int
                         ):
        # Prepare data
        exp_data = UnsupervisedDataset(dataset)  
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True)

        self.ssl_model.train()

        self.strategy.before_experience()
        
        for epoch in range(self.train_epochs):
            for mb_idx, stream_mbatch in tqdm(enumerate(data_loader)):
                stream_mbatch = stream_mbatch.to(self.device)

                stream_mbatch = self.strategy.before_mb_passes(stream_mbatch)

                for k in range(self.mb_passes):
                    # Apply strategy modifications before forward pass (e.g. concat replay samples from buffer)
                    mbatch = self.strategy.before_forward(stream_mbatch)

                    # Apply transforms, obtains a list of tensors, each containing 1 view for every sample in the mbatch
                    x_views_list = self.transforms(mbatch)

                    x_views_list = self.strategy.after_transforms(x_views_list)
                    
                    # Concat all tensors in the list in a single tensor
                    # x_views = torch.cat(x_views_list, dim=0)

                    # Forward pass of SSL model (z: projector features, e: encoder features)
                    loss, z_list, e_list = self.ssl_model(x_views_list)

                    # Subdivide into minibatch of features, each containing features corresponding to one view
                    # z_list, e_list = z_views.chunk(self.n_patches, dim=0), e_views.chunk(self.n_patches, dim=0)

                    # Strategy after forward pass
                    loss_strategy = self.strategy.after_forward(x_views_list, loss, z_list, e_list)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss_strategy.backward()
                    self.optimizer.step()

                    self.ssl_model.after_backward()
                    self.strategy.after_backward()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

                self.strategy.after_mb_passes()

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.ssl_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self.ssl_model
