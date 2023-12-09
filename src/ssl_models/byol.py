import torch
from torch import nn
import torch.nn.functional as F
import copy

from ..utils import update_ema_params

class BYOL(nn.Module):

    def __init__(self, base_encoder, dim_proj=2048, dim_pred=512,
                  byol_momentum=0.9, return_momentum_encoder=True):
        super(BYOL, self).__init__()

        self.byol_momentum = byol_momentum
        self.return_momentum_encoder = return_momentum_encoder

        # Online encoder
        self.online_encoder = base_encoder(num_classes=dim_proj, zero_init_residual=True)
        # Online projector
        prev_dim = self.online_encoder.fc.weight.shape[1]
        self.online_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.online_encoder.fc,
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.online_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        # Replace the fc clf layer with nn.Identity(),
        # so the encoder outputs feature maps instead of clf outputs
        self.online_encoder.fc = nn.Identity()

        # Momentum encoder
        self.momentum_encoder = copy.deepcopy(self.online_encoder)
        self.momentum_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradient in momentum network
        self.momentum_encoder.requires_grad_(False)
        self.momentum_projector.requires_grad_(False)

        # Build predictor for online network
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        def loss_byol(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)
        self.criterion = loss_byol

    def forward(self, x1, x2):

        
        
        # Both augmentations are passed in both momentum and online nets 

        z1_onl = self.online_projector(self.online_encoder(x1))
        z1_mom = self.momentum_projector(self.momentum_encoder(x1))

        z2_onl = self.online_projector(self.online_encoder(x2))
        z2_mom = self.momentum_projector(self.momentum_encoder(x2))

        p1 = self.predictor(z1_onl)
        p2 = self.predictor(z2_onl)

        loss = self.criterion(p1, z2_mom.detach()) + self.criterion(p2, z1_mom.detach())

        return loss.mean(), z1_onl, z2_onl
    
    @torch.no_grad()
    def update_momentum(self):
        # Update encoder
        update_ema_params(
            self.online_encoder.parameters(), self.momentum_encoder.parameters(), self.byol_momentum) 
        
        # Update projector
        update_ema_params(
           self.online_projector.parameters(), self.momentum_projector.parameters(), self.byol_momentum)

    def get_encoder(self):
        if self.return_momentum_encoder:
            return self.momentum_encoder
        else:
            return self.online_encoder
        
    def get_embedding_dim(self):
        return self.online_projector[0].weight.shape[1]
    
    def get_criterion(self):
        return self.criterion