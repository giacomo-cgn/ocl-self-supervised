from torch import nn
import torch

class BarlowTwins(nn.Module):

    def __init__(self, encoder, dim_features=2048, dim_pred=512, lambd=5e-3, save_pth=None):
        super(BarlowTwins, self).__init__()
        self.save_pth = save_pth
        self.model_name = 'barlow_twins'
        self.dim_features = dim_features
        self.dim_predictor = dim_pred # Not needed for Barlow Twins itself, but can be needed for additional layers in some strategies

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
                                        #nn.BatchNorm1d(dim_features, affine=False)
                                        ) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.bn_loss = nn.BatchNorm1d(dim_features, affine=False)

        # Replace the fc clf layer with nn.Identity()
        # so the encoder outputs feature maps instead of clf outputs
        self.encoder.fc = nn.Identity()

        def barlow_twins_loss(z1, z2):
            z1 = self.bn_loss(z1)
            z2 = self.bn_loss(z2)

            batch_size = z1.shape[0]
            # print("z1 shape:", z1.shape)
            # print("z1:", z1)
            # print("z2 shape:", z2.shape)
            # print("z2:", z2)

            # empirical cross-correlation matrix
            c = z1.T @ z2
            c.div_(batch_size)
            # print("c shape:", c.shape)
            # print("c:", c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            return on_diag + self.lambd * off_diag
        self.criterion = barlow_twins_loss

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'Lambda: {self.lambd}\n')
                f.write(f'dim_features: {dim_features}\n')
                f.write(f'dim_predictor: {dim_pred}\n')

    def forward(self, x1, x2):

        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        loss = self.criterion(z1, z2)
        return loss, z1, z2, e1, e2
    
    def get_encoder(self):
        return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_features
    
    def get_predictor_dim(self):
        return self.dim_predictor
    
    def get_criterion(self):
        return self.criterion
    
    def get_name(self):
        return self.model_name

    def after_backward(self):
        return
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()