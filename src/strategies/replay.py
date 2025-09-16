import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..analyze_features import OnlineFeatureMetrics

class Replay(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                ):
            
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size

        self.online_feature_metrics = OnlineFeatureMetrics(save_pth)

        self.strategy_name = 'replay'

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')

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
        """ Only update buffer features for replayed samples"""
        self.z_list = z_list
        self.e_list = e_list
        self.loss = loss
        if self.use_replay:
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
        
        return loss
    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

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

        


 


