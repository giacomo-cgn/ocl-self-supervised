import copy

import torch
from torch import nn

from ..utils import update_ema_params
from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

class APREPhi(AbstractStrategy):
    """Continual SSL strategy that aligns current representations of buffer 
    samples to their "past" representations obtained via EMA of the current network."""

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 phi: float = 2.0,
                 align_criterion: str = 'ssl',
                 momentum_ema: float = 0.999,
                 use_aligner: bool = True,
                 align_after_proj: bool = True,
                 aligner_dim: int = 512,
                 avg_seen_count: float = 38

               ):
        
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.phi = phi
        self.align_criterion_name = align_criterion
        self.momentum_ema = momentum_ema
        self.use_aligner = use_aligner
        self.align_after_proj = align_after_proj
        self.aligner_dim = aligner_dim
        self.avg_seen_count = avg_seen_count

        self.strategy_name = 'apre_phi'

        # Set up feature alignment criterion
        if self.align_criterion_name == 'ssl':
            criterion, is_binary = self.ssl_model.get_criterion()
            if is_binary:
                self.align_criterion = criterion
            else:
                raise Exception(f"Needs a binary criterion for alignment, cannot use {self.ssl_model.get_name()} as alignment loss.")
        elif self.align_criterion_name == 'mse':
            self.align_criterion = nn.MSELoss(reduction=None)
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
                f.write(f'phi (tanh for reg): {self.phi}\n')
                f.write(f'align_criterion: {self.align_criterion_name}\n')
                f.write(f'momentum_ema: {self.momentum_ema}\n')
                f.write(f'use_aligner: {self.use_aligner}\n')
                f.write(f'align_after_proj: {self.align_after_proj}\n')
                f.write(f'aligner_dim: {self.aligner_dim}\n')
                f.write(f'average seen count: {self.avg_seen_count}\n')

        self.z_sim_folder = os.path.join(self.save_pth, 'z_similarity')
        if not os.path.exists(self.z_sim_folder):
            os.makedirs(self.z_sim_folder)

        self.z_accuracy_file = os.path.join(self.z_sim_folder, 'z_acc.csv')
        with open(self.z_accuracy_file, 'a') as f:
            f.write('acc,repl_acc,stream_acc\n')
        



    def get_params(self):
        """Get trainable parameters of the strategy.
        
        Returns:
            alignment_projector (nn.Module): The alignment projector module.
        """
        return list(self.alignment_projector.parameters())

    
    def before_forward(self, stream_mbatch, stream_seen_count):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch
        self.stream_seen_count = stream_seen_count

        if len(self.buffer.buffer) >= self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, _, replay_seen_count, replay_indices, = self.buffer.sample(self.replay_mb_size)
            self.replay_seen_count = replay_seen_count
            replay_batch = replay_batch.to(self.device)
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            combined_seen_count = torch.cat((replay_seen_count, stream_seen_count), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices

        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch
            combined_seen_count = stream_seen_count

        return combined_batch, combined_seen_count
    
    def after_forward(self, x_views_list, loss, z_list, e_list, show_matrix, exp_idx, mb_idx, k_iter):
        """Calculate alignment loss and update replayed samples with new encoder features
            z_list: a list of minibatches, each minibatch corresponds to the one view of the samples
        """
        if not self.align_after_proj:
            # Use encoder features instead projector features
            z_list = e_list

        self.z_list = z_list

        #  similarity matrix
        z_sim1 = F.normalize(z_list[0], p=2, dim=1).detach()
        z_sim2 = F.normalize(z_list[1], p=2, dim=1).detach()
        similarity_matrix_z = torch.mm(z_sim1, z_sim2.T)
        if show_matrix:
            # show similarity matrix
            plt.imshow(similarity_matrix_z.cpu().detach().numpy())
            plt.colorbar()
            plt.savefig(os.path.join(self.z_sim_folder, f"exp{exp_idx}_mb{mb_idx}_k{k_iter}.png"))

        # ACCURACY
        # Get index of most similar vector for each row
        predicted_indices = similarity_matrix_z.argmax(dim=1)  # Index of max similarity along each row
        # Ground truth indices (assume correct match is same index)
        ground_truth_indices = torch.arange(z_sim1.shape[0], device=z_sim1.device)
        # Compute accuracy
        correct = (predicted_indices == ground_truth_indices).sum().item()
        accuracy = correct / z_sim1.shape[0]  # Fraction of correct matches

        if self.use_replay:
            replay_sim_matrix = similarity_matrix_z[:self.replay_mb_size, :self.replay_mb_size]
            stream_sim_matrix = similarity_matrix_z[self.replay_mb_size:, self.replay_mb_size:]

            # Compute accuracy in replay and stream matrix
            replay_correct = (replay_sim_matrix.argmax(dim=1) == torch.arange(replay_sim_matrix.shape[0], device=replay_sim_matrix.device)).sum().item()
            stream_correct = (stream_sim_matrix.argmax(dim=1) == torch.arange(stream_sim_matrix.shape[0], device=stream_sim_matrix.device)).sum().item()
            replay_accuracy = replay_correct / replay_sim_matrix.shape[0]
            stream_accuracy = stream_correct / stream_sim_matrix.shape[0]
             # Log accuracy
            with open(self.z_accuracy_file, 'a') as f:
                f.write(f'{accuracy:.2f},{replay_accuracy:.2f},{stream_accuracy:.2f}\n')
        else:
            # Log accuracy
             with open(self.z_accuracy_file, 'a') as f:
                f.write(f'{accuracy:.2f},{0},{accuracy:.2f}\n')


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

            num_views = len(z_list)

            # EMA model pass only on replay samples
            with torch.no_grad():
                x_replay_list = [x[:self.replay_mb_size] for x in x_views_list]
                ema_e = self.ema_encoder(torch.cat(x_replay_list, dim=0))
                if self.align_after_proj:
                    ema_z = self.ema_projector(ema_e)
                else:
                    # Directly use encoder features as alignment targets
                    ema_z = ema_e

            # Compute alignment loss between aligned features and EMA features
            loss_align = self.align_criterion(aligned_features, ema_z)

            tanh_seen_count = torch.tanh(self.phi/self.avg_seen_count * self.replay_seen_count).repeat(num_views).to(self.device)

            loss += (tanh_seen_count * loss_align).mean()

            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list_replay)/len(z_list_replay)
            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
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
        z_stream_avg = sum(z_list_stream)/len(z_list_stream)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach(), self.stream_seen_count.detach())