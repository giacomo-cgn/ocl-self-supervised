from torch import nn
import torch

class BarlowTwins(nn.Module):

    def __init__(self, encoder, dim_features=2048, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.lambd = lambd

        # Create encoder
        self.encoder = encoder(num_classes=dim_features, zero_init_residual=True)

        # Create 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim_features, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # Replace the fc clf layer with nn.Identity()
        # so the encoder outputs feature maps instead of clf outputs
        self.encoder.fc = nn.Identity()

        def barlow_twins_loss(z1, z2):
            batch_size = z1.shape[0]
            # empirical cross-correlation matrix
            c = z1.T @ z2

            c.div_(batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            return on_diag + self.lambd * off_diag
        self.criterion = barlow_twins_loss

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        loss = self.criterion(z1, z2)
        return loss, z1, z2
    
    def get_encoder(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_criterion(self):
        return self.criterion
    
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()