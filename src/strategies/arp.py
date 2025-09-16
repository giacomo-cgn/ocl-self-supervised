import torch
from torch import nn

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..analyze_features import OnlineFeatureMetrics

class ARP(AbstractStrategy):
    """Continual SSL strategy that aligns current representations of buffer 
    samples to their "past" representations stored in the buffer."""

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 omega: float = 0.1,
                 align_criterion: str = 'ssl',
                 use_aligner: bool = True,
                 align_after_proj: bool = True,
                 aligner_dim: int = 512
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
        self.aligner_dim = aligner_dim

        self.online_feature_metrics = OnlineFeatureMetrics(save_pth)


        self.strategy_name = 'arp'

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
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'aligner_dim: {self.aligner_dim}\n')

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
            replay_batch, replay_z_old, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch, replay_z_old = replay_batch.to(self.device), replay_z_old.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
            self.replay_z_old = replay_z_old

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
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            # Concatenate the features from all views
            z_replay = torch.cat(z_list_replay, dim=0)

            if self.use_aligner:
                # Align features after aligner
                aligned_features = self.alignment_projector(z_replay)
            else:
                # Do not use aligner
                aligned_features = z_replay

            # Extend the target old features extracted from the buffer, with copies of itself.
            # It is needed because we use the same replay_z_old as target for all the features 
            # corresponding to different views.
            extended_replay_z_old = self.replay_z_old.repeat(len(z_replay) // self.replay_z_old.size(0), 1)
            assert len(extended_replay_z_old) == len(z_replay)

            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first replay_mb_size elements)
            z_list_replay = [z[:self.replay_mb_size] for z in z_list]
            e_list_replay = [e[:self.replay_mb_size] for e in e_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            replay_loss = loss[:self.replay_mb_size]

            e_std, e_mean, e_cos_dist, e_angle = self.online_feature_metrics.calculate_stats_online(e_list_replay)
            z_std, z_mean, z_cos_dist, z_angle = self.online_feature_metrics.calculate_stats_online(z_list_replay)
            e_stats = {"std": e_std, "mean": e_mean, "cos_dist": e_cos_dist, "angle": e_angle}
            z_stats = {"std": z_std, "mean": z_mean, "cos_dist": z_cos_dist, "angle": z_angle}  

            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices, replay_loss.detach(),
                                        e_stats=e_stats, z_stats=z_stats)

            # Compute alignment loss between aligned features and EMA features
            loss_align = self.align_criterion(aligned_features, extended_replay_z_old)
            loss += self.omega * loss_align.mean()
        
        return loss


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

        # # Get features only of the streaming mbatch and their avg across views
        # z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
        # z_stream_avg = sum(z_list_stream)/len(z_list_stream)

        # # Update buffer with new stream samples and avg features
        # self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach())
