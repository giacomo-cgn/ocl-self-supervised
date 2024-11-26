from ..ssl_models import AbstractSSLModel
from ..strategies import AbstractStrategy
from ..backbones import get_encoder

import torch
from torch import nn

import copy



class DoubleResnet(AbstractStrategy, AbstractSSLModel):
    def __init__(self,
                 base_encoder: nn.Module,
                 dim_backbone_features: int,
                 image_size: int = 32,
                 dim_proj: int = 2048,
                 dim_pred: int = 512,
                 buffer = None,
                 device: str = 'cpu',
                 replay_mb_size: int = 32,
                 return_buffer_encoder: bool = False,
                 save_pth: str = None):
        
        # nn.Module.__init__(self)
        super().__init__()

        self.online_encoder = base_encoder
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.default_replay_mb_size = replay_mb_size
        self.save_pth = save_pth
        self.model_name = 'simsiam'
        self.dim_projector = dim_proj
        self.dim_predictor = dim_pred
        self.return_buffer_encoder = return_buffer_encoder

        self.strategy_name = 'double_resnet'
        self.model_name = 'double_resnet'       

        # Set up criterion
        self.criterion = nn.CosineSimilarity(dim=1)

        # Build a 3-layer projector
        self.online_projector = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_proj),
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.online_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN


        # Build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        self.buffer_projector = copy.deepcopy(self.online_projector)
        self.buffer_predictor = copy.deepcopy(self.predictor)
        self.buffer_encoder = copy.deepcopy(self.online_encoder)
        
        # Initialize Buffer-trained ResNet-9
        # self.buffer_encoder, buffer_dim_features = get_encoder('resnet9', image_size, self.model_name, False)

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL AND STRATEGY CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'dim_predictor: {dim_pred}\n')

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch
        self.replay_mb_size = len(stream_mbatch)

        if len(self.buffer.buffer) >= self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch = replay_batch.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices

            assert len(stream_mbatch) == len(replay_batch), "Stream and replay batch must have the same length for DoubleResnet"
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch


    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        # Compute features for both views
        e1 = self.online_encoder(x1)
        e2 = self.buffer_encoder(x2)

        z1 = self.online_projector(e1) # NxC
        z2 = self.buffer_projector(e2) # NxC

        if self.use_replay:

            replay_z1 = z1[:self.replay_mb_size] # Passed through online encoder (is the target)
            replay_z2 = z2[:self.replay_mb_size] # Passed through buffer encoder
            replay_p = self.buffer_predictor(replay_z2)

            stream_z1 = z1[self.replay_mb_size:] # Passed through online encoder
            stream_z2 = z2[self.replay_mb_size:] # Passed through buffer encoder (is the target)
            stream_p = self.predictor(stream_z1)

            loss = -(self.criterion(replay_p, replay_z1).mean() + self.criterion(stream_p, stream_z2).mean())

            return loss, [z1, z2], [e1, e2]
        
        else:
            return None, [z1, z2], [e1, e2]

    
    def get_encoder(self):
       return self.online_encoder
    
    def get_encoder_for_eval(self):
        if self.return_buffer_encoder:
            return self.buffer_encoder
        else:
            return self.online_encoder
    
    def get_projector(self):
        if self.return_buffer_encoder:
            return self.buffer_projector
        else:
            return self.online_projector
        
    def get_embedding_dim(self):
        return self.online_projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_projector
    
    def get_criterion(self):
        return self.criterion, True
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())
    

    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features for replayed samples"""
        self.z_list = z_list
        if self.use_replay:
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay) # CONSIDER TO UPDATE IF COMBINING WITH MINRED BUFFER
            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
        return loss
    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

        # Get features only of the streaming mbatch and their avg across views
        z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
        z_stream_avg = sum(z_list_stream)/len(z_list_stream)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach())





        