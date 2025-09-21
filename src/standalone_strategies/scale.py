import os
from tqdm import tqdm
import copy

import torch
from torch.utils.data import DataLoader
from torch.functional import F
import torch.nn as nn

from ..utils import UnsupervisedDataset
from ..transforms import get_transforms
from ..optims import init_optim


from ..ssl_models import AbstractSSLModel
from ..strategies.abstract_strategy import AbstractStrategy

class SCALE(AbstractStrategy, AbstractSSLModel):

    def __init__(self,
                 encoder: nn.Module,
                 dim_backbone_features: int = 512,
                 buffer = None,
                 buffer_type: str = 'scale',
                 device = 'cpu',
                 save_pth: str  = None,
                 train_mb_size: int = 32,
                 replay_mb_size: int = 32,

                 temperature_cont: float = 0.1,
                 temperature_past: float = 0.01,
                 temperature_curr: float = 0.1,
                 distill_power: float = 0.15,
                 temp_tsne: float = 0.1,
                 tsne_thresh_ratio: float = 0.1,
                 dim_features: int = 128,
    ):           
        super().__init__()

        if encoder is None:
            raise Exception(f'This strategy requires an encoder.')
        if buffer is None:
            raise Exception(f'This strategy requires a buffer')
        
        self.encoder = encoder.to(device)

        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.train_mb_size = train_mb_size
        self.replay_mb_size = replay_mb_size

        self.temperature_cont = temperature_cont
        self.temperature_past = temperature_past
        self.temperature_curr = temperature_curr
        self.distill_power = distill_power
        self.temp_tsne = temp_tsne
        self.tsne_thresh_ratio = tsne_thresh_ratio
        self.features_dim = dim_features

        if buffer_type == 'scale':
            self.use_scale_buffer = True
        else:
            self.use_scale_buffer = False

        self.strategy_name = 'SCALE'
        self.model_name = 'SCALE' 


    
        self.tr_distill_power = 0.0

        prev_dim = dim_backbone_features
        print('prev_dim:', prev_dim)
        self.proj_dim = prev_dim
        self.encoder.fc = nn.Identity() # Remove cls output layer

        self.projector = nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.ReLU(inplace=True),
                nn.Linear(prev_dim, self.features_dim)
            ).to(self.device)
        
        self.criterion = SupConLoss(stream_bsz=self.train_mb_size,
                                projector=self.projector,
                                temperature=self.temperature_cont,
                                device=self.device).to(self.device)
        
        self.criterion_reg = IRDLoss(projector=self.projector,
                            current_temperature=self.temperature_curr,
                            past_temperature=self.temperature_past, device=self.device).to(self.device)
        

        self.losses_contrast = AverageMeter()
        self.losses_distill = AverageMeter()


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'replay_mb_size: {self.replay_mb_size}\n')

                f.write(f'temperature_cont: {self.temperature_cont}\n')
                f.write(f'temperature_past: {self.temperature_past}\n')
                f.write(f'temperature_curr: {self.temperature_curr}\n')
                f.write(f'distill_power: {self.distill_power}\n')
                f.write(f'temp_tsne: {self.temp_tsne}\n')
                f.write(f'tsne_thresh_ratio: {self.tsne_thresh_ratio}\n')
                f.write(f'dim_features: {self.features_dim}\n')
                f.write(f'use_scale_buffer: {self.use_scale_buffer}\n')


                with open(os.path.join(self.save_pth, 'tr_distill_power.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')

                self.already_got_params = False

    def get_params(self):
        if self.already_got_params == False:
            all_parameters = [{
                'name': 'backbone',
                'params': [param for name, param in self.encoder.named_parameters()],
            }, {
                'name': 'heads',
                'params': [param for name, param in self.criterion.named_parameters()],
            }]
            self.already_got_params = True
        else:
            all_parameters = []
        return all_parameters
    
    def before_forward(self, stream_mbatch):
        self.stream_mbatch = stream_mbatch

        self.past_encoder = copy.deepcopy(self.encoder) # CHECKED! ONLY THE ENCODERS ARE COPIED, NOT THE PROJECTION HEADS!
        self.past_encoder.eval().to(self.device)

        if self.use_scale_buffer:
            # Try sampling from SCALE buffer
            replay_batch, replay_indices = self.buffer.sample(self.replay_mb_size)
            if replay_batch is None:
                # Not enough elements in buffer
                combined_batch = stream_mbatch
                self.use_replay = False
            else:
                # Concat buffer with stream samples
                combined_batch = torch.cat((replay_batch.to(self.device), stream_mbatch), dim=0)
                self.use_replay = True
                self.replay_indices = replay_indices
        else:
            # Try sampling from default buffer
            if self.buffer.get_curr_len() > self.replay_mb_size:
                self.use_replay = True
                # Sample from buffer and concat
                self.replay_batch, _, replay_indices = self.buffer.sample(self.replay_mb_size)
                self.replay_batch = self.replay_batch.to(self.device)
                combined_batch = torch.cat((self.replay_batch, self.stream_mbatch), dim=0)
                 # Save buffer indices of replayed samples
                self.replay_indices = replay_indices
            else:
                self.use_replay = False
                # Do not sample buffer if not enough elements in it
                combined_batch = stream_mbatch

        return combined_batch
    
    def forward(self, x_views_list):
        x1 = x_views_list[0]
        x2 = x_views_list[1]

        all_x = torch.cat((x1, x2), dim=0)

        combined_batch_size = x1.shape[0]
        loss_distill = .0

        x1_logits, loss_distill = self.criterion_reg(self.encoder, self.past_encoder, x1)
        self.losses_distill.update(loss_distill.item(), combined_batch_size)

        features_all = self.encoder(all_x)
        contrast_mask = similarity_mask_old(features_all, combined_batch_size,
                                            self.device, self.temp_tsne, self.tsne_thresh_ratio, self.train_mb_size)
        loss_contrast, z1, z2 = self.criterion(self.encoder, self.encoder, x1, x2,
                                mask=contrast_mask)
        
        self.losses_contrast.update(loss_contrast.item(), combined_batch_size)

        if self.tr_distill_power <= 0.0 and loss_distill > 0.0:
            self.tr_distill_power = self.losses_contrast.avg * self.distill_power / self.losses_distill.avg

        loss = loss_contrast + self.tr_distill_power * loss_distill

        # Split features_all in features1 and features2
        e1 = features_all[:combined_batch_size]
        e2 = features_all[combined_batch_size:]

        return loss, [z1, z2], [e1, e2]

    def after_forward(self, x_views_list, loss_batch, z_list, e_list):

        # Save distill power
        with open(os.path.join(self.save_pth, 'tr_distill_power.csv'), 'a') as f:
            f.write(f'{self.tr_distill_power},0,0,0,0\n')

        self.z_list = z_list
        self.e_list = e_list
        if self.use_replay:
            if self.use_scale_buffer:
                e_list_replay = [e[:self.replay_mb_size] for e in e_list]
                avg_replayed_e = sum(e_list_replay)/len(e_list_replay)
                self.buffer.update_embeddings(avg_replayed_e.detach(), self.replay_indices)

            else:
                # Take only the features from the replay batch (for each view minibatch in z_list,
                #  take only the first replay_mb_size elements)
                z_list_replay = [z[:self.replay_mb_size] for z in z_list]
                # Update replayed samples with avg of last extracted features
                avg_replayed_z = sum(z_list_replay)/len(z_list_replay)

                self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)

    def after_mb_passes(self):
        # Update buffer with new samples
        if self.use_scale_buffer:
            e_list_stream = [e[-len(self.stream_mbatch):] for e in self.e_list]
            avg_stream_e = sum(e_list_stream)/len(e_list_stream)
            all_embeddings, select_indexes = self.buffer.update_wo_labels(self.stream_mbatch.detach().cpu(), avg_stream_e.detach(), self.encoder)
        else:
            # Get features only of the streaming mbatch and their avg across views
            z_list_stream = [z[-len(self.stream_mbatch):] for z in self.z_list]
            z_stream_avg = sum(z_list_stream)/len(z_list_stream)

            # Update buffer with new stream samples and avg features
            self.buffer.add(self.stream_mbatch.detach(), z_stream_avg.detach())

    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.features_dim
    
    def get_criterion(self):
        return None, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,
                 stream_bsz,
                 projector,
                 temperature=0.07,
                 base_temperature=0.07,
                 device="cpu"):
        super(SupConLoss, self).__init__()
        self.stream_bsz = stream_bsz
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.projector = projector
        self.device = device

    def forward(self, backbone_stu, backbone_tch, x_stu, x_tch, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        The arguments format is designed to align with other losses.
        In SimCLR, the two backbones should be the same
        Args:
            backbone_stu: backbone for student
            backbone_tch: backbone for teacher
            x_stu: raw augmented vector of shape [bsz, ...].
            x_tch: raw augmented vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        z_stu = F.normalize(self.projector(backbone_stu(x_stu)), dim=1)
        z_tch = F.normalize(self.projector(backbone_tch(x_tch)), dim=1)

        batch_size = x_stu.shape[0]

        all_features = torch.cat((z_stu, z_tch), dim=0)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        # print(mean_log_prob_pos.shape, mean_log_prob_pos.max().item(), mean_log_prob_pos.mean().item(), mean_log_prob_pos.min().item())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batch_size)
        stream_mask = torch.zeros_like(loss).float().to(self.device)
        stream_mask[:, :self.stream_bsz] = 1
        loss = (stream_mask * loss).sum() / stream_mask.sum()
        return loss, z_stu, z_tch


class IRDLoss(nn.Module):
    """Instance-wise Relation Distillation (IRD) Loss for Contrastive Continual Learning
        https://arxiv.org/pdf/2106.14413.pdf
    """
    def __init__(self, projector, current_temperature=0.2,
                past_temperature=0.01, device="cpu"):
        super(IRDLoss, self).__init__()
        self.projector = projector
        self.curr_temp = current_temperature
        self.past_temp = past_temperature
        self.device = device

    def forward(self, backbone, past_backbone, x):
        """Compute loss for model.
        Args:
            backbone: current backbone
            past_backbone: past backbone
            x: raw input of shape [bsz * n_views, ...]
        Returns:
            A loss scalar.
        """

        cur_features = F.normalize(self.projector(backbone(x)), dim=1)
        past_features = F.normalize(self.projector(past_backbone(x)), dim=1)

        cur_features_sim = torch.div(torch.matmul(cur_features, cur_features.T),
                                    self.curr_temp)
        logits_mask = torch.scatter(
            torch.ones_like(cur_features_sim),
            1,
            torch.arange(cur_features_sim.size(0)).view(-1, 1).to(self.device),
            0
        )
        cur_logits_max, _ = torch.max(cur_features_sim * logits_mask, dim=1, keepdim=True)
        cur_features_sim = cur_features_sim - cur_logits_max.detach()
        row_size =cur_features_sim.size(0)
        cur_logits = torch.exp(cur_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            cur_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        # print('cur_logits', cur_logits * 1e4)

        past_features_sim = torch.div(torch.matmul(past_features, past_features.T), self.past_temp)
        past_logits_max, _ = torch.max(past_features_sim * logits_mask, dim=1, keepdim=True)
        past_features_sim = past_features_sim - past_logits_max.detach()
        past_logits = torch.exp(past_features_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
            past_features_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

        loss_distill = (- past_logits * torch.log(cur_logits)).sum(1).mean()
        #return loss_distill

        return cur_logits, loss_distill
    

def similarity_mask_old(feat_all, bsz, device, temp_tsne, tsne_thresh_ratio, batch_size):
    """Calculate the pairwise similarity and the mask for contrastive learning
    Args:
        feat_all: all hidden features of shape [n_views * bsz, ...].
        bsz: int, batch size of input data (stacked streaming and memory samples)
        opt: arguments
    Returns:
        contrast_mask: mask of shape [bsz, bsz]
    """
    #print(feat_all[0])
    #print(feat_all[1])
    feat_size = feat_all.size(0)
    n_views = int(feat_size / bsz)
    assert (n_views * bsz == feat_size), "Unmatch feature sizes and batch size!"

    # Compute the pairwise distance and similarity between each view
    # and add the similarity together for average
    simil_mat_avg = torch.zeros(bsz, bsz).to(device)
    mat_cnt = 0
    for i in range(n_views):
        for j in range(n_views):
            # feat_row and feat_col should be of size [bsz^2, bsz^2]
            #feat_row, feat_col = PairEnum(feat_all[i*bsz: (i+1)*bsz],
            #                              feat_all[j*bsz: (j+1)*bsz])
            #tmp_distance = -(((feat_row - feat_col) / temperature) ** 2.).sum(1)  # Euclidean distance
            # Note, all features are normalized
            # tSNE similarity
            # compute euclidean distance pairs
            simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                            feat_all[j*bsz: (j+1)*bsz].T)
            #print('\teuc dist', simil_mat * 1e4)
            tmp_distance = - torch.div(simil_mat, temp_tsne)
            tmp_distance = tmp_distance - 1000 * torch.eye(bsz).to(device)
            #print('\ttemp dist', tmp_distance * 1e4)
            simil_mat = 0.5 * torch.softmax(tmp_distance, 1) + 0.5 * torch.softmax(tmp_distance, 0)
            #print(torch.softmax(tmp_distance, 1))
            #print('simil_mat', simil_mat)

            # Add the new probability to the average probability
            simil_mat_avg = (mat_cnt * simil_mat_avg + simil_mat) / (mat_cnt + 1)
            mat_cnt += 1
    #print('simil_mat_avg', simil_mat_avg * 1e4)
    logits_mask = torch.scatter(
        torch.ones_like(simil_mat_avg),
        1,
        torch.arange(simil_mat_avg.size(0)).view(-1, 1).to(device),
        0
    )
    simil_max = simil_mat_avg[logits_mask.bool()].max()
    simil_mean = simil_mat_avg[logits_mask.bool()].mean()
    simil_min = simil_mat_avg[logits_mask.bool()].min()
    #print('prob_simil_avg: dim {}\tmax {}\tavg {}\tmin {}'.format(
    #    simil_mat_avg.shape[0], simil_max, simil_mean, simil_min))
    # Set diagonal of similarity matrix to ones
    masks = torch.eye(bsz).to(device)
    simil_mat_avg = simil_mat_avg * (1 - masks) + masks

    # mask out memory elements
    stream_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    stream_mask[:batch_size, :batch_size] = 1
    simil_mat_avg = simil_mat_avg * stream_mask

    contrast_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    tsne_simil_thres = simil_mean + tsne_thresh_ratio * (simil_max - simil_mean)
    # print(simil_thres)
    contrast_mask[simil_mat_avg > tsne_simil_thres] = 1

    return contrast_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count