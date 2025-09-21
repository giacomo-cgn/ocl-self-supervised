import torch
from torch.nn import functional as F

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..analyze_features import OnlineFeatureMetrics

class ReplayPriorityOverlap(AbstractStrategy):
    """
    Replay strategy, only that it concatenates the replay buffer with the current batch for the first minibatch pass (mb_pass),
    and then only uses the replay buffer for the rest of the mb_passes.
    """

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 stream_mb_size: int = 32,
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
        self.stream_mb_size = stream_mb_size
        self.use_buffer_overlap = use_buffer_overlap
        self.clamp_overlap_loss = clamp_overlap_loss
        self.overlap_omega = overlap_omega
        self.overlap_num_buffer_samples = overlap_num_buffer_samples

        self.online_feature_metrics = OnlineFeatureMetrics(save_pth)

        self.strategy_name = 'replay_priority_overlap'

        self.count_mb_passes = 0

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'use buffer overlap: {self.use_buffer_overlap}\n')
                f.write(f'clamp overlap loss: {self.clamp_overlap_loss}\n')
                f.write(f'overlap omega: {self.overlap_omega}\n')
                f.write(f'overlap num buffer samples: {self.overlap_num_buffer_samples}\n')

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch

        if self.count_mb_passes > 0:
            # Only use buffer samples
            self.curr_replay_size = self.replay_mb_size + self.stream_mb_size
        else:
            # Concat buffer and stream samples
            self.curr_replay_size = self.replay_mb_size
        self.curr_replay_size = min(self.curr_replay_size, self.buffer.get_curr_len())

        if self.buffer.get_curr_len() > 0:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.curr_replay_size)
            replay_batch = replay_batch.to(self.device)
            
            if self.count_mb_passes > 0:
                combined_batch = replay_batch
            else:
                combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)

            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch
    
    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features for replayed samples"""
        self.z_list = z_list
        self.e_list = e_list
        self.loss = loss
        if self.use_replay:
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first self.curr_replay_size elements)
            z_list_replay = [z[:self.curr_replay_size] for z in z_list]
            e_list_replay = [e[:self.curr_replay_size] for e in e_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            replay_loss = loss[:self.curr_replay_size]

            e_std, e_mean_replay, e_cos_dist, e_angle_replay = self.online_feature_metrics.calculate_stats_online(e_list_replay, detach=False)
            z_std, z_mean, z_cos_dist, z_angle = self.online_feature_metrics.calculate_stats_online(z_list_replay)
            e_stats = {"std": e_std.detach(), "mean": e_mean_replay.detach(), "cos_dist": e_cos_dist.detach(), "angle": e_angle_replay.detach()}
            z_stats = {"std": z_std, "mean": z_mean, "cos_dist": z_cos_dist, "angle": z_angle}

            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices, replay_loss.detach(),
                                        e_stats=e_stats, z_stats=z_stats)
        else:
            # If no replay, do not update buffer features
            self.replay_indices = None
            e_mean_replay, e_angle_replay = None, None
            

        if self.count_mb_passes == 0:
            # Update buffer with new samples after all mb_passes with streaming mbatch.

            # Get features only of the streaming mbatch and their avg across views
            z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
            e_list_stream = [e[-len(self.stream_mbatch):] for e in self.e_list]
            z_stream_avg = sum(z_list_stream)/len(z_list_stream)
            self.stream_loss = self.loss[-len(self.stream_mbatch):]

            e_std, e_mean_stream, e_cos_dist, e_angle_stream = self.online_feature_metrics.calculate_stats_online(e_list_stream, detach=False)
            z_std, z_mean, z_cos_dist, z_angle = self.online_feature_metrics.calculate_stats_online(z_list_stream)
            e_stats = {"std": e_std.detach(), "mean": e_mean_stream.detach(), "cos_dist": e_cos_dist.detach(), "angle": e_angle_stream.detach()}
            z_stats = {"std": z_std, "mean": z_mean, "cos_dist": z_cos_dist, "angle": z_angle}

            # Update buffer with new stream samples and avg features
            self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach(), batch_loss=self.stream_loss.detach(),
                            e_stats=e_stats, z_stats=z_stats)
        else:
            e_mean_stream, e_angle_stream = None, None

        self.count_mb_passes += 1

        # Concat replay and stream angle and mean features
        if e_mean_replay is not None and e_angle_replay is not None:
            e_mean = torch.cat((e_mean_replay, e_mean_stream), dim=0) if e_mean_stream is not None else e_mean_replay
            e_angle = torch.cat((e_angle_replay, e_angle_stream), dim=0) if e_angle_stream is not None else e_angle_replay
        else:
            e_mean = e_mean_stream
            e_angle = e_angle_stream

        # print(f'e_mean shape: {e_mean.shape}, e_angle shape: {e_angle.shape}')

        # Overlap loss
        if self.use_buffer_overlap:
            # Limit the number of samples used from the buffer
            # take only the 500 buffer samples with highest loss. loss is in buffer.buffer_loss
            if self.buffer.get_curr_len() > self.overlap_num_buffer_samples:
                # Get the indices of top loss samples in the buffer, excluding self.replay_indices
                if self.replay_indices is not None:
                    # Exclude replay indices, do not want duplicate them in the overlap loss
                    if isinstance(self.buffer.buffer_loss, list):
                        # If self.buffer.buffer_loss is a list, convert to tensor
                        buffer_loss_tensor = torch.stack(self.buffer.buffer_loss)
                    else:
                        buffer_loss_tensor = self.buffer.buffer_loss
                    mask = torch.ones_like(buffer_loss_tensor, dtype=torch.bool)
                    mask[self.replay_indices] = False
                    masked_buffer = buffer_loss_tensor[mask]
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
        # print(f'e_angle_buffer shape: {e_angle_buffer.shape}, e_mean_buffer shape: {e_mean_buffer.shape}')

        centers_dist = cosine_similarity_matrix(e_mean, e_mean_buffer)
        # print(f'centers_dist shape: {centers_dist.shape}')
        sum_radius = e_angle.unsqueeze(1) + e_angle_buffer.unsqueeze(0)
        overlap_gap = sum_radius - centers_dist
        if self.clamp_overlap_loss:
            overlap_gap = torch.clamp(overlap_gap, min=0.0)
        loss_overlap = overlap_gap.mean(dim=1)
        loss += self.overlap_omega * loss_overlap

        return loss
    

    def after_mb_passes(self):
        """Reset count_mb_passes after all mb_passes"""

        self.count_mb_passes = 0
        
        # Calculate Online metrics
        self.online_feature_metrics.calculate_metrics_online(self.buffer.buffer_e_stats, self.buffer.buffer_z_stats)


def cosine_similarity_matrix(A, B):
    # Normalize A and B along the last dimension
    A_norm = F.normalize(A, p=2, dim=1)  # (N1, d)
    B_norm = F.normalize(B, p=2, dim=1)  # (N2, d)

    # Compute cosine similarity: (N1, d) @ (d, N2) -> (N1, N2)
    similarity = A_norm @ B_norm.T
    return similarity