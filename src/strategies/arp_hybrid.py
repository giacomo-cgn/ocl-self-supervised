import torch
from torch import nn

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..buffers import HybridMinRedFIFOBuffer

class ARPHybrid(AbstractStrategy):
    """Continual SSL strategy that aligns current representations of buffer 
    samples to their "past" representations stored in the buffer."""

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer: HybridMinRedFIFOBuffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 omega: float = 0.1,
                 align_criterion: str = 'ssl',
                 use_aligner: bool = True,
                 align_after_proj: bool = True,
                 aligner_dim: int = 512,
                 fifo_samples_ratio: float = 0.5
                ):

        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.omega = omega
        self.align_criterion_name = align_criterion
        self.use_aligner = use_aligner
        self.align_after_proj = align_after_proj
        self.aligner_dim = aligner_dim
        self.fifo_samples_ratio = fifo_samples_ratio

        self.strategy_name = 'arp_hybrid'

       # Set up feature alignment criterion
        if self.align_criterion_name == 'ssl':
            criterion, is_binary = self.ssl_model.get_criterion()
            if is_binary:
                self.align_criterion = criterion
            else:
                raise Exception(f"Needs a binary criterion for alignment, cannot use {self.ssl_model.get_name()} as alignment loss.")
        elif self.align_criterion_name == 'mse':
            self.align_criterion = nn.MSELoss()
        elif self.align_criterion_name == 'cosine':
            self.align_criterion = nn.CosineSimilarity(dim=1)
        else:
            raise Exception(f"Invalid alignment criterion: {self.align_criterion_name}")

        # Set up alignment projector
        if self.align_after_proj:
            dim_proj = self.ssl_model.get_projector_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_proj, self.aligner_dim, bias=False),
                                                nn.BatchNorm1d(self.aligner_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.aligner_dim, dim_proj)).to(self.device)
        else:
            dim_encoder_embed = self.ssl_model.get_embedding_dim()
            self.alignment_projector = nn.Sequential(nn.Linear(dim_encoder_embed, self.aligner_dim, bias=False),
                                                nn.BatchNorm1d(self.aligner_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.aligner_dim, dim_encoder_embed)).to(self.device)


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'omega: {self.omega}\n')
                f.write(f'align_criterion: {self.align_criterion_name}\n')
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'align_after_proj: {self.align_after_proj}\n')
                f.write(f'aligner_dim: {self.aligner_dim}\n')
                f.write(f'arp hybrid fifo_samples_ratio: {self.fifo_samples_ratio}\n')

    def get_params(self):
        """Get trainable parameters of the strategy.
        
        Returns:
            alignment_projector (nn.Module): The alignment projector module.
        """
        return list(self.alignment_projector.parameters())
    

    def before_mb_passes(self, stream_mbatch):
        """Add the stream mbatch to the buffer, with mbatch features obtained 
        with an additional encoder pass."""
        # Skip if mb size == 1 (problems with batchnorm)
        if not len(stream_mbatch) == 1:
            with torch.no_grad():
                e_mbatch = self.ssl_model.get_encoder()(stream_mbatch.detach())
                z_mbatch = self.ssl_model.get_projector()(e_mbatch)

            # Add stream minibatch and features to buffer
            if self.buffer.using_only_fifo:
                self.buffer.add(stream_mbatch.detach(), z_mbatch.detach())
            else:
                self.buffer.add(stream_mbatch.detach(), z_mbatch.detach(), self.alignment_projector)

        return stream_mbatch
    

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.use_only_fifo = self.buffer.using_only_fifo

        replay_batch_size = min(self.replay_mb_size, len(self.buffer.buffer_fifo))
        
        if self.use_only_fifo:
            # Sample only from FIFO buffer
            fifo_batch, _, fifo_indices = self.buffer.sample_fifo(replay_batch_size)
            fifo_batch = fifo_batch.to(self.device)
            self.fifo_indices = fifo_indices
            return fifo_batch

        else:
            # Sample from FIFO and MinRed
            self.fifo_mb_size = int(replay_batch_size * self.fifo_samples_ratio)
            self.minred_mb_size = replay_batch_size - self.fifo_mb_size
            fifo_batch, _, fifo_indices = self.buffer.sample_fifo(self.fifo_mb_size)
            minred_batch, minred_z_old, minred_indices = self.buffer.sample_minred(self.minred_mb_size)

            fifo_batch, minred_batch, minred_z_old = fifo_batch.to(self.device), minred_batch.to(self.device), minred_z_old
            self.fifo_indices, self.minred_indices = fifo_indices, minred_indices
            self.minred_z_old = minred_z_old

            return torch.cat((minred_batch, fifo_batch), dim=0)


    def after_forward(self, x_views_list, loss, z_list, e_list):
        """Calculate alignment loss and update replayed samples with new encoder features
            z_list: a list of minibatches, each minibatch corresponds to the one view of the samples
        """
        if not self.align_after_proj:
            # Use encoder features instead projector features
            z_list = e_list

        self.z_list = z_list

        if not self.use_only_fifo:
            # Take only the features from the minred batch (for each view minibatch in z_list,
            #  take only the first minred_mb_size elements)
            z_list_minred = [z[:self.minred_mb_size] for z in z_list]
            z_list_fifo = [z[self.minred_mb_size:] for z in z_list]
            # Concatenate the features from all views
            z_minred = torch.cat(z_list_minred, dim=0)
            z_fifo = torch.cat(z_list_fifo, dim=0)

            if self.use_aligner:
                # Align features after aligner
                aligned_features = self.alignment_projector(z_minred)
            else:
                # Do not use aligner
                aligned_features = z_minred

            # Extend the target old features extracted from the buffer, with copies of itself.
            # It is needed because we use the same minred_z_old as target for all the features 
            # corresponding to different views.
            extended_minred_z_old = self.minred_z_old.repeat(len(z_minred) // self.minred_z_old.size(0), 1)
            assert len(extended_minred_z_old) == len(z_minred)

            # Compute alignment loss between aligned features and EMA features
            loss_align = self.align_criterion(aligned_features, extended_minred_z_old)
            loss += self.omega * loss_align.mean()

            # Update replayed samples with avg of last extracted features
            avg_minred_z = sum(z_list_minred)/len(z_list_minred)
            self.buffer.update_features_minred(avg_minred_z.detach(), self.minred_indices)
            avg_fifo_z = sum(z_list_fifo)/len(z_list_fifo)
            self.buffer.update_features_fifo(avg_fifo_z.detach(), self.fifo_indices)

        else:
            # Only FIFO buffer - no alignment
            avg_fifo_z = sum(z_list)/len(z_list)
            self.buffer.update_features_fifo(avg_fifo_z.detach(), self.fifo_indices)
        
        return loss