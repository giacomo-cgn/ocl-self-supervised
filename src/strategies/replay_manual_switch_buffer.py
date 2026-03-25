import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..analyze_features import OnlineFeatureMetrics
from ..buffers import ReservoirBuffer, FIFOBuffer

class ReplayManualSwitchBuffer(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 switch_exp_idx: int = -1,
                ):
            
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.switch_exp_idx = switch_exp_idx
        self.curr_exp_idx = -1
        self.has_switched_buffer = False
        self.use_replay = False

        if self.switch_exp_idx >= 0 and not isinstance(self.buffer, ReservoirBuffer):
            raise Exception('ReplayManualSwitchBuffer requires a ReservoirBuffer before switching')

        self.online_feature_metrics = OnlineFeatureMetrics(save_pth)

        self.strategy_name = 'replay_manual_switch_buffer'

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'switch_exp_idx: {self.switch_exp_idx}\n')

    def _clone_scalar_list(self, values):
        cloned = []
        for v in values:
            if hasattr(v, 'detach'):
                cloned.append(v.detach().clone())
            else:
                cloned.append(v)
        return cloned

    def _switch_buffer_if_needed(self):
        if self.has_switched_buffer:
            return
        if self.switch_exp_idx < 0:
            return
        if self.curr_exp_idx < self.switch_exp_idx:
            return
        if isinstance(self.buffer, FIFOBuffer):
            self.has_switched_buffer = True
            return
        if not isinstance(self.buffer, ReservoirBuffer):
            raise Exception(f'Cannot switch from unsupported buffer type {type(self.buffer)}')
        
        print(f'Switching replay buffer at experience {self.curr_exp_idx} from {type(self.buffer)} to FIFOBuffer')

        old_buffer = self.buffer
        new_buffer = FIFOBuffer(old_buffer.buffer_size, alpha_ema=old_buffer.alpha_ema)

        curr_len = old_buffer.get_curr_len()
        if curr_len > 0:
            new_buffer.buffer = [x.detach().clone() for x in old_buffer.buffer[:curr_len]]
            new_buffer.buffer_features = [z.detach().clone() for z in old_buffer.buffer_features[:curr_len]]

            if old_buffer.buffer_loss.size(0) >= curr_len and curr_len > 0:
                new_buffer.buffer_loss = [l.detach().clone() for l in old_buffer.buffer_loss[:curr_len]]

            new_buffer.lifetimes = [int(v.item()) for v in old_buffer.lifetimes[:curr_len]]
            new_buffer.extractions = [int(v.item()) for v in old_buffer.extractions[:curr_len]]

            for key, values in old_buffer.buffer_e_stats.items():
                new_buffer.buffer_e_stats[key] = self._clone_scalar_list(values[:curr_len])
            for key, values in old_buffer.buffer_z_stats.items():
                new_buffer.buffer_z_stats[key] = self._clone_scalar_list(values[:curr_len])

        new_buffer.finished_lifetimes = [
            int(v.item()) if hasattr(v, 'item') else int(v)
            for v in old_buffer.finished_lifetimes
        ]
        new_buffer.finished_extractions = [
            int(v.item()) if hasattr(v, 'item') else int(v)
            for v in old_buffer.finished_extractions
        ]

        self.buffer = new_buffer
        self.has_switched_buffer = True

        print(f'Finished switching replay buffer at experience {self.curr_exp_idx}. Buffer now has {self.buffer.get_curr_len()} samples.')

    def before_experience(self):
        self.curr_exp_idx += 1
        self._switch_buffer_if_needed()

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

        


 


