from torch import nn

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim_proj=2048, dim_pred=512):
        """
        dim_proj: feature dimension (default: 2048)
        dim_pred: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
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

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

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
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_criterion(self):
        return self.criterion
