import copy

import torch
from torch import nn

from ..utils import update_ema_params
from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..analyze_features import OnlineFeatureMetrics
from .replay_priority_overlap import cosine_similarity_matrix

class APREOverlap(AbstractStrategy):
    """Continual SSL strategy that aligns current representations of buffer 
    samples to their "past" representations obtained via EMA of the current network.
    + overlap loss"""

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 omega: float = 0.1,
                 align_criterion: str = 'ssl',
                 momentum_ema: float = 0.999,
                 use_aligner: bool = True,
                 aligner_dim: int = 512,
                 align_after_proj: bool = True,
                 use_buffer_overlap: bool = True,
                 clamp_overlap_loss: bool = True,
                 overlap_omega: float = 1.0,
                 overlap_num_buffer_samples: int = 500
               ):
        
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.omega = omega
        self.align_criterion_name = align_criterion
        self.momentum_ema = momentum_ema
        self.use_aligner = use_aligner
        self.aligner_dim = aligner_dim
        self.use_buffer_overlap = use_buffer_overlap
        self.clamp_overlap_loss = clamp_overlap_loss
        self.overlap_omega = overlap_omega
        self.overlap_num_buffer_samples = overlap_num_buffer_samples

        self.online_feature_metrics = OnlineFeatureMetrics(save_pth)

        self.strategy_name = 'apre_overlap'

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
            self.align_criterion = lambda x,y: -nn.CosineSimilarity(dim=1)(x,y)
        else:
            raise Exception(f"Invalid alignment criterion: {self.align_criterion_name}")

        
        # Set up EMA model that is targeted for alignment. It is the EMA of encoder+projector
        self.ema_encoder = copy.deepcopy(self.ssl_model.get_encoder())
        self.ema_projector = copy.deepcopy(self.ssl_model.get_projector())

        # Stop gradient in EMA model
        self.ema_encoder.requires_grad_(False)
        self.ema_projector.requires_grad_(False)

        # Set up alignment projector
        dim_proj = self.ssl_model.get_projector_dim()
        self.alignment_projector = nn.Sequential(nn.Linear(dim_proj, self.aligner_dim, bias=False),
                                            nn.BatchNorm1d(self.aligner_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.aligner_dim, dim_proj)).to(self.device)

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'omega: {self.omega}\n')
                f.write(f'align_criterion: {self.align_criterion_name}\n')
                f.write(f'momentum_ema: {self.momentum_ema}\n')
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'aligner_dim: {self.aligner_dim}\n')
                f.write(f'use_buffer_overlap: {self.use_buffer_overlap}\n')
                f.write(f'clamp_overlap_loss: {self.clamp_overlap_loss}\n')
                f.write(f'overlap_omega: {self.overlap_omega}\n')
                f.write(f'overlap_num_buffer_samples: {self.overlap_num_buffer_samples}\n')

    def get_params(self):
        """Get trainable parameters of the strategy.
        
        Returns:
            alignment_projector (nn.Module): The alignment projector module.
        """
        return list(self.alignment_projector.parameters())

    
    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch

        if self.buffer.get_curr_len() > self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch = replay_batch.to(self.device)
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices

        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch
    
    def after_forward(self, x_views_list, loss, z_list, e_list):
        """Calculate alignment loss and update replayed samples with new encoder features
            z_list: a list of minibatches, each minibatch corresponds to the one view of the samples
        """

        self.z_list = z_list
        self.e_list = e_list
        self.loss = loss


        if self.use_replay:
            # Take only the features from the replay batch (for each minibatch in z_list, take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            # Concatenate the features from all views
            z_replay = torch.cat(z_list_replay, dim=0)

            if self.use_aligner:
                # Align features after aligner
                aligned_features = self.alignment_projector(z_replay)
            else:
                # Do not use aligner
                aligned_features = z_replay


            # EMA model pass only on replay samples
            with torch.no_grad():
                x_replay_list = [x[:self.replay_mb_size] for x in x_views_list]
                ema_e = self.ema_encoder(torch.cat(x_replay_list, dim=0))
                ema_z = self.ema_projector(ema_e)

            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            e_list_replay = [e[:self.replay_mb_size] for e in e_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            replay_loss = loss[:self.replay_mb_size]

            e_std, e_mean, e_cos_dist, e_angle = self.online_feature_metrics.calculate_stats_online(e_list)
            z_std, z_mean, z_cos_dist, z_angle = self.online_feature_metrics.calculate_stats_online(z_list)
            e_stats = {"std": e_std[:self.replay_mb_size], "mean": e_mean[:self.replay_mb_size], "cos_dist": e_cos_dist[:self.replay_mb_size], "angle": e_angle[:self.replay_mb_size]}
            z_stats = {"std": z_std[:self.replay_mb_size], "mean": z_mean[:self.replay_mb_size], "cos_dist": z_cos_dist[:self.replay_mb_size], "angle": z_angle[:self.replay_mb_size]}

            self.buffer.update_features(avg_replayed_z.detach(), replay_loss.detach(), self.replay_indices,
                                        e_stats=e_stats, z_stats=z_stats)

            # Compute alignment loss between aligned features and EMA features
            loss_align = self.align_criterion(aligned_features, ema_z)
            loss += self.omega * loss_align.mean()

        # Overlap loss
        if self.use_replay:
            if self.use_buffer_overlap:
                # Limit the number of samples used from the buffer
                # take only the 500 buffer samples with highest loss. loss is in buffer.buffer_loss
                if self.buffer.get_curr_len() > self.overlap_num_buffer_samples:
                    # Get the indices of top loss samples in the buffer, excluding self.replay_indices
                    if self.replay_indices is not None:
                        # Exclude replay indices, do not want duplicate them in the overlap loss
                        mask = torch.ones_like(self.buffer.buffer_loss, dtype=torch.bool)
                        mask[self.replay_indices] = False
                        masked_buffer = self.buffer.buffer_loss[mask]
                        buffer_indices = torch.topk(masked_buffer, min(self.overlap_num_buffer_samples, masked_buffer.size(0)), largest=True).indices
                        buffer_indices = torch.nonzero(mask, as_tuple=False)[buffer_indices].squeeze()
                    else:
                        # If no replay indices, just take the top indices from the buffer loss
                        _, buffer_indices = torch.topk(self.buffer.buffer_loss, self.overlap_num_buffer_samples, largest=True)

                    e_angle_buffer = torch.stack(self.buffer.buffer_e_stats['angle'])[buffer_indices].detach()
                    e_mean_buffer = torch.stack(self.buffer.buffer_e_stats['mean'])[buffer_indices].detach()
                else:
                    # If not enough samples in the buffer, use all of them
                    e_angle_buffer = torch.stack(self.buffer.buffer_e_stats['angle']).detach()
                    e_mean_buffer = torch.stack(self.buffer.buffer_e_stats['mean']).detach()
                    
                e_angle_buffer = e_angle_buffer.to(self.device)  # [N2, D]
                e_mean_buffer = e_mean_buffer.to(self.device)    # [N2, D]
            else:
                e_angle_buffer = e_angle # [N2, D]
                e_mean_buffer = e_mean  # [N2, D]

            centers_dist = cosine_similarity_matrix(e_mean, e_mean_buffer)
            sum_radius = e_angle.unsqueeze(1) + e_angle_buffer.unsqueeze(0)
            overlap_gap = sum_radius - centers_dist
            if self.clamp_overlap_loss:
                overlap_gap = torch.clamp(overlap_gap, min=0.0)
            loss_overlap = overlap_gap.mean(dim=1)
            loss += self.overlap_omega * loss_overlap

        return loss

    def after_backward(self):
        """Update EMA model after each mb pass backward."""
        # Update EMA model
        update_ema_params(self.ssl_model.get_encoder().parameters(),
                            self.ema_encoder.parameters(), self.momentum_ema)
        update_ema_params(self.ssl_model.get_projector().parameters(),
                            self.ema_projector.parameters(), self.momentum_ema)
        

    def after_mb_passes(self):
        """Update buffer with new samples after all mb pass with streaming mbatch."""

         # Get features only of the streaming mbatch and their avg across views
        z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
        e_list_stream = [e[-len(self.stream_mbatch):] for e in self.e_list]
        z_stream_avg = sum(z_list_stream)/len(z_list_stream)
        self.stream_loss = self.loss[-len(self.stream_mbatch):]

        e_std, e_mean, e_cos_dist, e_angle = self.online_feature_metrics.calculate_stats_online(e_list_stream)
        z_std, z_mean, z_cos_dist, z_angle = self.online_feature_metrics.calculate_stats_online(z_list_stream)
        e_stats = {"std": e_std, "mean": e_mean, "cos_dist": e_cos_dist, "angle": e_angle}
        z_stats = {"std": z_std, "mean": z_mean, "cos_dist": z_cos_dist, "angle": z_angle}

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach(), batch_loss=self.stream_loss.detach(),
                        e_stats=e_stats, z_stats=z_stats)
        
        # Calculate Online metrics
        self.online_feature_metrics.calculate_metrics_online(self.buffer.buffer_e_stats, self.buffer.buffer_z_stats)