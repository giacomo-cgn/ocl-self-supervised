import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from ..utils import UnsupervisedDataset, init_optim
from ..transforms import get_transforms


class EMP(nn.Module):

    def __init__(self,
                 encoder: nn.Module = None,
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
                 n_patches: int = 20,
                 dim_proj: int = 2048,
                 emp_tcr_param: float = 1,
                 emp_tcr_eps: float = 0.2,
                 emp_patch_sim: int = 200,
               ):
        
        super(EMP, self).__init__()

        if encoder is None:
            raise Exception(f'This strategy requires an encoder.')

        # Create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = encoder(num_classes=dim_proj, zero_init_residual=True)

        # Build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN 

        # Replace the fc clf layer with nn.Identity(),
        # so the encoder outputs feature maps instead of clf outputs
        self.encoder.fc = nn.Identity()      

        # self.encoder = encoder
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
        self.n_patches = n_patches
        self.dim_proj = dim_proj
        self.emp_tcr_param = emp_tcr_param
        self.emp_tcr_eps = emp_tcr_eps
        self.emp_patch_sim = emp_patch_sim

        self.strategy_name = 'EMP'
        self.model_and_strategy_name = 'EMP'

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common', n_crops=self.n_patches)
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.model.get_name())

        # Set up optimizer
        self.optimizer = init_optim(optim, self.parameters(), lr=self.lr,
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
                f.write(f'dim_proj: {self.dim_proj}\n')
                f.write(f'n_patches: {self.n_patches}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')

        self.encoder = self.encoder.to(self.device)
        self.projector = self.projector.to(self.device)

        # Loss definitions
        self.contractive_loss = Similarity_Loss()
        self.criterion = TotalCodingRate(eps=self.emp_tcr_eps)

    def forward(self, x):
        z = self.projector(self.encoder(x))
        return z


    def train_experience(self, 
                         dataset,
                         exp_idx: int
                         ):
        # Prepare data
        exp_data = UnsupervisedDataset(dataset)  
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True)

        self.train()
        
        for epoch in range(self.train_epochs):
            for mb_idx, mbatch in tqdm(enumerate(data_loader)):
                mbatch = mbatch.to(self.device)

                for k in range(self.mb_passes):
                    # Apply transforms
                    x_patches = self.transforms(mbatch)

                    # Forward pass
                    z = self.forward(x_patches)

                    # Subdivide z projections in patches from same sample
                    z_list = z.chunk(self.n_patches, dim=0)
                    # Get average z projection for all patches belonging to same sample
                    z_avg = chunk_avg(z, self.n_patches)

                    loss_contract, _ = self.contractive_loss(z_list, z_avg)
                    loss_TCR = cal_TCR(z, self.criterion, self.n_patches)
                    
                    loss = self.emp_patch_sim*loss_contract + self.emp_tcr_param*loss_TCR

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # self.model.after_backward()

                    # Save loss, exp_idx, epoch, mb_idx and k in csv
                    if self.save_pth is not None:
                        with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                            f.write(f'{loss.item()},{exp_idx},{epoch},{mb_idx},{k}\n')

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.save_pth, f'model_exp{exp_idx}.pth'))

        return self
    
    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_proj
    
    def get_predictor_dim(self):
        return self.dim_predictor
    
    def get_criterion(self):
        return self.criterion
    
    def get_name(self):
        return self.name

    def after_backward(self):
        return
    



class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out
    

def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches,dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss/num_patches
    return loss

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)
    

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)