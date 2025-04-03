import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class ReplayBufferPriority(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 stream_mb_size: int = 32
                ):
            
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.stream_mb_size = stream_mb_size

        self.strategy_name = 'replay_buffer_priority'

        self.count_mb_passes = 0

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

        if self.count_mb_passes > 0:
            # Only use buffer samples
            self.curr_replay_size = self.replay_mb_size + self.stream_mb_size
        else:
            # Concat buffer and stream samples
            self.curr_replay_size = self.replay_mb_size
        self.curr_replay_size = min(self.curr_replay_size, len(self.buffer.buffer))

        if len(self.buffer.buffer) > 0:
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
        self.loss = loss
        if self.use_replay:
            # Take only the features from the replay batch (for each view minibatch in z_list,
            #  take only the first self.curr_replay_size elements)
            z_list_replay = [z[:self.curr_replay_size] for z in z_list]
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            replay_loss = loss[:self.curr_replay_size]
            self.buffer.update_features(avg_replayed_z.detach(), replay_loss.detach(), self.replay_indices)

        if self.count_mb_passes == 0:
            # Update buffer with new samples after all mb_passes with streaming mbatch.

            # Get features only of the streaming mbatch and their avg across views
            z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
            z_stream_avg = sum(z_list_stream)/len(z_list_stream)
            self.stream_loss = self.loss[-len(self.stream_mbatch):]
            # Update buffer with new stream samples and avg features
            self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach(), batch_loss=self.stream_loss.detach())

        self.count_mb_passes += 1
        
        return loss
    

    def after_mb_passes(self):
        """Reset count_mb_passes after all mb_passesl"""

        self.count_mb_passes = 0