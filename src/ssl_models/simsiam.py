from torch import nn

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim_proj=2048, dim_pred=512, save_pth=None):
        super(SimSiam, self).__init__()
        self.save_pth = save_pth
        self.model_name = 'simsiam'
        self.dim_projector = dim_proj
        self.dim_predictor = dim_pred

        # Set up criterion
        self.criterion = nn.CosineSimilarity(dim=1)

        # Create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim_proj, zero_init_residual=True)

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

        # Build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'dim_predictor: {dim_pred}\n')

    def forward(self, x1, x2):
        # Compute features for both views
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        z1 = self.projector(e1) # NxC
        z2 = self.projector(e2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

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
        return self.dim_projector
    
    def get_predictor_dim(self):
        return self.dim_predictor
    
    def get_criterion(self):
        return self.criterion
    
    def get_name(self):
        return self.model_name

    def after_backward(self):
        return