import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from einops import rearrange
from typing import Sequence, List, Tuple

from .utils import UnsupervisedDataset
from .transforms import get_transforms


class FeatureAnalyzer():
    def __init__(self,
                 train_stream,
                 when_features_analysis: str = ['50','end'],
                 dataset_name: str = 'cifar100',
                 transforms_type: str = 'common',
                 num_views: int = 20,
                 num_exp_samples: int = 500,
                 overlap_thresh_multipliers_cosine: list = [0.1, 0.3, 1],
                 mb_size: int=10,
                 device: str ='cpu',
                 save_pth: str = None,
                 ):
        
        self.when_features_analysis = when_features_analysis
        self.overlap_thresh_multipliers_cosine = overlap_thresh_multipliers_cosine
        self.mb_size = mb_size
        self.num_exp_samples = num_exp_samples
        self.device = device

        # Select a subset for each experience and prepared dataloaders
        self.train_stream_dataloaders = [self.prepare_exp_data(exp_dataset) for exp_dataset in train_stream]

        if transforms_type == 'common':
            self.transforms = get_transforms(dataset=dataset_name, n_crops=num_views, online_transforms=True)
        else:
            raise Exception(f'Transforms type {transforms_type} not supported')
        
        if save_pth is not None:
            self.e_save_pth = os.path.join(save_pth, 'feature_analysis','e_features')
            os.makedirs(self.e_save_pth, exist_ok=True)
            self.z_save_pth = os.path.join(save_pth, 'feature_analysis','z_features')
            os.makedirs(self.z_save_pth, exist_ok=True)

            # Write deviation analysis headers
            with open(os.path.join(self.e_save_pth, 'deviation.csv'), 'a') as f:
                f.write('exp_idx,tr_step,std_deviation_buffer,std_deviation_current,std_deviation_past,std_deviation_future,'
                        'cosine_deviation_buffer,cosine_deviation_current,cosine_deviation_past,cosine_deviation_future\n')
            with open(os.path.join(self.z_save_pth, 'deviation.csv'), 'a') as f:
                f.write('exp_idx,tr_step,std_deviation_buffer,std_deviation_current,std_deviation_past,std_deviation_future,'
                        'cosine_deviation_buffer,cosine_deviation_current,cosine_deviation_past,cosine_deviation_future\n')

            # Write uniformity loss analysis headers
            with open(os.path.join(self.e_save_pth, 'uniformity_loss.csv'), 'a') as f:
                f.write('exp_idx,tr_step,loss_unif_buffer,loss_unif_current,loss_unif_past,loss_unif_future\n')
            with open(os.path.join(self.z_save_pth, 'uniformity_loss.csv'), 'a') as f:
                f.write('exp_idx,tr_step,loss_unif_buffer,loss_unif_current,loss_unif_past,loss_unif_future\n')

            # Write overlap analysis headers
            with open(os.path.join(self.e_save_pth, 'overlap.csv'), 'a') as f:
                f.write('exp_idx,tr_step,mult,overlap_b2b_list,overlap_b2c_list,overlap_b2p_list,overlap_b2f_list,'
                        'overlap_c2b_list,overlap_c2c_list,overlap_c2p_list,overlap_c2f_list,'
                        'overlap_p2b_list,overlap_p2c_list,overlap_p2p_list,overlap_p2f_list,'
                        'overlap_f2b_list,overlap_f2c_list,overlap_f2p_list,overlap_f2f_list\n')
            with open(os.path.join(self.z_save_pth, 'overlap.csv'), 'a') as f:
                f.write('exp_idx,tr_step,mult,overlap_b2b_list,overlap_b2c_list,overlap_b2p_list,overlap_b2f_list,'
                        'overlap_c2b_list,overlap_c2c_list,overlap_c2p_list,overlap_c2f_list,'
                        'overlap_p2b_list,overlap_p2c_list,overlap_p2p_list,overlap_p2f_list,'
                        'overlap_f2b_list,overlap_f2c_list,overlap_f2p_list,overlap_f2f_list\n')
        
            # Save configuration
            with open(save_pth + '/config.txt', 'a') as f:
                f.write('\n')
                f.write('---- ANALYZE FEATURES DEVIATION CONFIG ----\n')
                f.write(f'when (training steps) to analyze: {self.when_features_analysis}\n')
                f.write(f'num analyzed views: {num_views}\n')
                f.write(f'num experience samples: {num_exp_samples}\n')
                f.write(f'mb size analysis: {self.mb_size}\n')
                f.write(f'overlap std multipliers (for threshold) cosine: {self.overlap_thresh_multipliers_cosine}\n')

        else:
            self.e_save_pth = None
            self.z_save_pth = None




    def analyze_features(self, encoder, buffer_data, exp_idx, tr_step, projector=None):

        print(f'>>> Analyzing features deviation at step {tr_step}')

        buffer_dataloader = self.prepare_buffer_data(buffer_data)

        encoder.eval()
        if projector is not None:
            projector.eval()

        with torch.no_grad():

            e_buffer_metrics_dict, z_buffer_metrics_dict = self.extract_intermediate_metrics(encoder, buffer_dataloader, projector=projector)

            e_exp_metrics_dict_list = []
            z_exp_metrics_dict_list = []
            for exp_dataloader in self.train_stream_dataloaders:
                e_metrics_dict, z_metrics_dict = self.extract_intermediate_metrics(encoder, exp_dataloader, projector=projector)
                e_exp_metrics_dict_list.append(e_metrics_dict)
                z_exp_metrics_dict_list.append(z_metrics_dict)

            self.extract_write_final_metrics(e_exp_metrics_dict_list, e_buffer_metrics_dict, exp_idx, tr_step, self.e_save_pth)
            if projector is not None:
                self.extract_write_final_metrics(z_exp_metrics_dict_list, z_buffer_metrics_dict, exp_idx, tr_step, self.z_save_pth)
            
        encoder.train()
        if projector is not None:
            projector.train()


    def extract_write_final_metrics(self, exp_metrics_dict_list, buffer_metrics_dict, exp_idx, tr_step, folder):
        # Aggregate and calculate metrics for buffer and current experience
            mean_current = exp_metrics_dict_list[exp_idx]['mean']
            angle_current = exp_metrics_dict_list[exp_idx]['mean_angle']
            mean_buffer = buffer_metrics_dict['mean']
            angle_buffer = buffer_metrics_dict['mean_angle']

            cosine_deviation_current = exp_metrics_dict_list[exp_idx]['mean_cosine']
            cosine_deviation_buffer = buffer_metrics_dict['mean_cosine']
            std_deviation_current = exp_metrics_dict_list[exp_idx]['std']
            std_deviation_buffer = buffer_metrics_dict['std']

            loss_unif_buffer = lunif(mean_buffer)
            loss_unif_current = lunif(mean_current)
            overlap_b2b_list, _, _, _ = calculate_overlap_cosine(mean_buffer, mean_buffer, angle_buffer, angle_buffer, self.overlap_thresh_multipliers_cosine)
            overlap_c2c_list, _, _, _ = calculate_overlap_cosine(mean_current, mean_current, angle_current, angle_current, self.overlap_thresh_multipliers_cosine)
            overlap_b2c_list, overlap_c2b_list, _, _ = calculate_overlap_cosine(mean_buffer, mean_current, angle_buffer, angle_current, self.overlap_thresh_multipliers_cosine)

            if exp_idx > 0:
                # If not the first experience, aggregate past experiences
                mean_past = torch.cat([exp_metrics_dict['mean'] for exp_metrics_dict in exp_metrics_dict_list[:exp_idx]], dim=0)
                angle_past = torch.cat([exp_metrics_dict['mean_angle'] for exp_metrics_dict in exp_metrics_dict_list[:exp_idx]], dim=0)
                cosine_deviation_past = np.mean([exp_metrics_dict['mean_cosine'] for exp_metrics_dict in exp_metrics_dict_list[:exp_idx]])
                std_deviation_past = np.mean([exp_metrics_dict['std'] for exp_metrics_dict in exp_metrics_dict_list[:exp_idx]])
                loss_unif_past = lunif(mean_past)
                overlap_p2p_list, _, _, _ = calculate_overlap_cosine(mean_past, mean_past, angle_past, angle_past, self.overlap_thresh_multipliers_cosine)
                overlap_p2c_list, overlap_c2p_list, _, _ = calculate_overlap_cosine(mean_past, mean_current, angle_past, angle_current, self.overlap_thresh_multipliers_cosine)
                overlap_p2b_list, overlap_b2p_list, _, _ = calculate_overlap_cosine(mean_past, mean_buffer, angle_past, angle_buffer, self.overlap_thresh_multipliers_cosine)
            else:
                loss_unif_past = 0
                cosine_deviation_past = 0
                std_deviation_past = 0
                overlap_p2p_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_p2c_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_c2p_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_p2b_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_b2p_list = [0] * len(self.overlap_thresh_multipliers_cosine)

            if exp_idx < len(exp_metrics_dict_list) - 1:
                # If there are future experiences, aggregate and calculate metrics
                mean_future = torch.cat([exp_metrics_dict['mean'] for exp_metrics_dict in exp_metrics_dict_list[exp_idx+1:]], dim=0)
                angle_future = torch.cat([exp_metrics_dict['mean_angle'] for exp_metrics_dict in exp_metrics_dict_list[exp_idx+1:]], dim=0)
                cosine_deviation_future = np.mean([exp_metrics_dict['mean_cosine'] for exp_metrics_dict in exp_metrics_dict_list[exp_idx+1:]])
                std_deviation_future = np.mean([exp_metrics_dict['std'] for exp_metrics_dict in exp_metrics_dict_list[exp_idx+1:]])
                loss_unif_future = lunif(mean_future)
                overlap_f2f_list, _, _, _ = calculate_overlap_cosine(mean_future, mean_future, angle_future, angle_future, self.overlap_thresh_multipliers_cosine)
                overlap_f2c_list, overlap_c2f_list, _, _ = calculate_overlap_cosine(mean_future, mean_current, angle_future, angle_current, self.overlap_thresh_multipliers_cosine)
                overlap_f2b_list, overlap_b2f_list, _, _ = calculate_overlap_cosine(mean_future, mean_buffer, angle_future, angle_buffer, self.overlap_thresh_multipliers_cosine)
            else:
                loss_unif_future = 0
                cosine_deviation_future = 0
                std_deviation_future = 0
                overlap_f2f_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_f2c_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_c2f_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_f2b_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_b2f_list = [0] * len(self.overlap_thresh_multipliers_cosine)


            if exp_idx > 0 and exp_idx < len(exp_metrics_dict_list) - 1:
                # If there are past and future experiments, calculate the overlap between past and future
                overlap_p2f_list, overlap_f2p_list, _, _ = calculate_overlap_cosine(mean_past, mean_future, angle_past, angle_future, self.overlap_thresh_multipliers_cosine)
            else:
                overlap_p2f_list = [0] * len(self.overlap_thresh_multipliers_cosine)
                overlap_f2p_list = [0] * len(self.overlap_thresh_multipliers_cosine)

            # Write deviation
            with open(os.path.join(folder, 'deviation.csv'), 'a') as f:
                f.write(f'{exp_idx},{tr_step},{std_deviation_buffer},{std_deviation_current},{std_deviation_past},{std_deviation_future},'
                        f'{cosine_deviation_buffer},{cosine_deviation_current},{cosine_deviation_past},{cosine_deviation_future}\n')


            # Write uniformity loss
            with open(os.path.join(folder, 'uniformity_loss.csv'), 'a') as f:
                f.write(f'{exp_idx},{tr_step},{loss_unif_buffer},{loss_unif_current},{loss_unif_past},{loss_unif_future}\n')

            # Write overlap
            with open(os.path.join(folder, 'overlap.csv'), 'a') as f:
                for i, mult in enumerate(self.overlap_thresh_multipliers_cosine):
                    f.write(f'{exp_idx},{tr_step},{mult},{overlap_b2b_list[i]},{overlap_b2c_list[i]},{overlap_b2p_list[i]},{overlap_b2f_list[i]},'
                            f'{overlap_c2b_list[i]},{overlap_c2c_list[i]},{overlap_c2p_list[i]},{overlap_c2f_list[i]},'
                            f'{overlap_p2b_list[i]},{overlap_p2c_list[i]},{overlap_p2p_list[i]},{overlap_p2f_list[i]},'
                            f'{overlap_f2b_list[i]},{overlap_f2c_list[i]},{overlap_f2p_list[i]},{overlap_f2f_list[i]}\n')

            


    def extract_intermediate_metrics(self, encoder, dataloader, projector=None):
        """
        Extract features and compute features space initial metrics for a given dataset.

        Args:
            encoder (nn.Module): Encoder model.
            dataloader (DataLoader): DataLoader for the dataset.
            projector (nn.Module, optional): Projector model. Defaults to None.

            Returns:
            e_dict : dictionary containing the following keys:
                "mean": tensor of list of e mean vectors (mean of all view features)
                "mean_angle": tensor of list of e mean angles (mean of angles between all view features)
                "std": average standard deviation among all view features of e
                "mean_cosine": average cosine distance among all view features of e
                "std_cosine": std of cosine distance among all view features of e

            z_dict : dictionary containing the following keys:
                "mean": tensor of list of z mean vectors (mean of all view features)
                "mean_angle": tensor of list of z mean angles (mean of angles between all view features)
                "std": average standard deviation among all view features of z
                "mean_cosine": average cosine distance among all view features of z
                "std_cosine": std of cosine distance among all view features of z         
        """

        encoder.eval()
        if projector is not None:
            projector.eval()

        with torch.no_grad():

            e_std_list, e_mean_list, z_std_list, z_mean_list = [], [], [], []
            e_std_cosine_list, e_mean_cosine_list, z_std_cosine_list, z_mean_cosine_list = [], [], [], []
            e_mean_angle_list, z_mean_angle_list = [], []

            # Extract features from current data
            for _, current_mbatch in enumerate(tqdm(dataloader)):
                current_mbatch = current_mbatch.to(self.device)
                current_mbatch_views = self.transforms(current_mbatch)

                # Convert from list of minibatch views to list of per sample views
                V = torch.stack(current_mbatch_views, dim=0) # first stack current_mbatch_views → (v, b, d1, d2, …)
                per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d1, d2, …) and unbind

                for x_views in per_sample_views_list:
                    e_views = encoder(x_views)
                    # Compute mean and std of e_views
                    e_std_list.append(torch.std(e_views, correction=0))
                    e_mean_list.append(torch.mean(e_views, dim=0))

                    # Calculate pairwise cosine distance between e_views (obtains mean, std and mean angle of cosine dist)
                    e_cosine_mean, e_cosine_std, e_mean_angle = pairwise_cosine_dist_stats(e_views)
                    e_mean_cosine_list.append(e_cosine_mean)
                    e_std_cosine_list.append(e_cosine_std)
                    e_mean_angle_list.append(e_mean_angle)

                    if projector is not None:
                        z_views = projector(e_views)
                        # Compute mean and std of z_views
                        z_std_list.append(torch.std(z_views, correction=0))
                        z_mean_list.append(torch.mean(z_views, dim=0))

                        # Calculate pairwise cosine distance between z_views (obtains mean, std and mean angle of cosine dist)
                        z_cosine_mean, z_cosine_std, z_mean_angle = pairwise_cosine_dist_stats(z_views)
                        z_mean_cosine_list.append(z_cosine_mean)
                        z_std_cosine_list.append(z_cosine_std)
                        z_mean_angle_list.append(z_mean_angle)


            e_dict = {
                "mean": torch.stack(e_mean_list, dim=0).cpu(),
                "mean_angle": torch.stack(e_mean_angle_list, dim=0).cpu() ,
                "std": torch.mean(torch.stack(e_std_list), dim=0).cpu() ,
                "mean_cosine": torch.mean(torch.stack(e_mean_cosine_list), dim=0).cpu() ,
                "std_cosine": torch.mean(torch.stack(e_mean_cosine_list), dim=0).cpu() 
            }
            
            if projector is not None:
                z_dict = {
                    "mean": torch.stack(z_mean_list, dim=0).cpu() ,
                    "mean_angle": torch.stack(z_mean_angle_list, dim=0).cpu() ,
                    "std": torch.mean(torch.stack(z_std_list), dim=0).cpu() ,
                    "mean_cosine": torch.mean(torch.stack(z_mean_cosine_list), dim=0).cpu() ,
                    "std_cosine": torch.mean(torch.stack(z_std_cosine_list), dim=0).cpu()   
                }
            else:
                z_dict = None

            return e_dict, z_dict


    def prepare_buffer_data(self, buffer_data):
        # If buffer_data is a list, convert it to a tensor, stack along a new dim
        if isinstance(buffer_data, list):
            buffer_data = torch.stack(buffer_data, dim=0)
        buffer_dataloader = DataLoader(buffer_data, batch_size=self.mb_size, shuffle=False)
            
        return buffer_dataloader
    
    def prepare_exp_data(self, exp_dataset):
        # Select self.num_exp_samples samples from exp_dataset (i.e. select a subset of each experience dataset for analysis)
        subset_exp_data = UnsupervisedDataset(Subset(exp_dataset, np.random.choice(len(exp_dataset), self.num_exp_samples, replace=False)))
        exp_dataloader = DataLoader(subset_exp_data, batch_size=self.mb_size, shuffle=False)
        return exp_dataloader
    
    def get_when_to_analyze(self, tr_step, total_steps):
        """
        Determines whether feature deviation analysis should be performed at the current training step.
        Args:
            tr_step (int): Current training step.
            total_steps (int): Total number of training steps.

        Returns:
            bool: True if analysis should be performed at this step, False otherwise.
                 Returns True if:
                 - It's the final step (tr_step == total_steps - 1) and 'end' is in when_features_analysis
                 - The current step number (as string) is in when_features_analysis
        """
        if tr_step == (total_steps-1):
            if 'end' in self.when_features_analysis:
                return True
        if  str(tr_step) in self.when_features_analysis:
            return True
        return False
    
def pairwise_cosine_dist_stats(X: torch.Tensor):
    """
    Args:
        X: Tensor of shape (N, d)
    Returns:
        mean_dist: scalar tensor, average cosine distance over all pairs (i<j)
        std_dist:  scalar tensor, standard deviation of those distances
        mean_angle: average angle of pairwise cosine similarity
    """
    N = X.size(0)
    # Get all (i,j) index pairs with i < j
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)

    # Gather the corresponding vectors
    Xi = X[idx_i]   # shape (M, d) where M = N*(N-1)/2
    Xj = X[idx_j]   # same shape

    # Compute pairwise cosine similarity along the feature dimension
    cos_sim = F.cosine_similarity(Xi, Xj, dim=1)  # shape (M,)

    # Convert to cosine distance
    dists = 1.0 - cos_sim                         # shape (M,)

    # Calculate mean angle
    mean_angle = torch.acos(cos_sim).mean()

    # Statistics
    mean_dist = dists.mean()
    std_dist  = dists.std(correction=0) 
    
    return mean_dist, std_dist, mean_angle


def calculate_overlap_cosine(
    mean_features_1: torch.Tensor,  # [N1, D]
    mean_features_2: torch.Tensor,  # [N2, D]
    mean_angle_cosim_1: torch.Tensor,   # [N1] (mean angle of cosine similarity)
    mean_angle_cosim_2: torch.Tensor,   # [N2]
    thresh_multipliers: Sequence[float] = (1, 2, 3, 5),
    ) -> Tuple[List[float], List[float], List[float], List[float]]:

    """
    Calculate the overlap between two sets of features, given the mean and per-sample mean cosine dist.

    - Convert centroid cosine-sims -> angles θ_ij = arccos(sim_ij).
    - Convert each mean cosine-distance r -> angular radius φ = arccos(clamp(1 - r, -1,1)).
    - For each k in thresh_multipliers, count overlaps where
        θ_ij <= k * (φ1_i + φ2_j)
    Args:
        mean_features_1 (torch.Tensor[N1, D]): Feature centroids of set 1.
        mean_features_2 (torch.Tensor[N2, D]): Feature centroids of set 2.
        mean_angle_cosim_1 (torch.Tensor[N1]): Per-sample mean angle of cosine-similarity for set 1.
        mean_angle_cosim_2 (torch.Tensor[N2]): Per-sample mean angle of cosine-similarity for set 2.
        thresh_multipliers (Sequence[float], optional):
            Multiples of the sum of angular radii to use as overlap thresholds.
            Defaults to (1, 2, 3, 5).

    Returns:
        mean_num_overlap_1 (List[float]):
            Mean number of overlaps per sample in set 1, for each threshold multiplier.
        mean_num_overlap_2 (List[float]):
            Mean number of overlaps per sample in set 2, for each threshold multiplier.
        std_num_overlap_1 (List[float]):
            Standard deviation of overlaps per sample in set 1, for each threshold multiplier.
        std_num_overlap_2 (List[float]):
            Standard deviation of overlaps per sample in set 2, for each threshold multiplier.
    """

    N1, D = mean_features_1.shape
    N2, _ = mean_features_2.shape

    # Build [N1, N2, D] tensors for pairwise comparison
    a = mean_features_1.unsqueeze(1).expand(N1, N2, D)  # [N1, N2, D]
    b = mean_features_2.unsqueeze(0).expand(N1, N2, D)  # [N1, N2, D]

    # Pairwise cosine‐similarity (already in [-1,1])
    sim = cosine_similarity_chunked(a, b, dim=2, eps=1e-8)    # [N1, N2]

    # Angular distances between centroids
    theta12 = torch.acos(sim)                           # [N1, N2]

    phi1 = mean_angle_cosim_1.view(-1, 1)                # [N1, 1]
    phi2 = mean_angle_cosim_2.view(1, -1)                # [1, N2]

    phi_sum = phi1 + phi2                                # [N1, N2]

    means1, means2, stds1, stds2 = [], [], [], []

    for k in thresh_multipliers:
        thresh   = k * phi_sum
        overlaps = theta12 <= thresh                   # [N1, N2] mask

        cnt1 = overlaps.sum(dim=0).float()             # [N1]
        cnt2 = overlaps.sum(dim=1).float()             # [N2]

        means1.append(cnt1.mean().item())
        stds1.append(cnt1.std(correction=0).item())
        means2.append(cnt2.mean().item())
        stds2.append(cnt2.std(correction=0).item())

    return means1, means2, stds1, stds2

def calculate_per_sample_overlap_cosine(
    mean_features_1: torch.Tensor,  # [N1, D]
    mean_features_2: torch.Tensor,  # [N2, D]
    mean_angle_cosim_1: torch.Tensor,   # [N1] (mean angle of cosine similarity)
    mean_angle_cosim_2: torch.Tensor,   # [N2]
    ) -> Tuple[List[float], List[float], List[float], List[float]]:

    """
    Calculate the per-sampleoverlap between two sets of features, given the mean and per-sample mean cosine dist.

    - Convert centroid cosine-sims -> angles θ_ij = arccos(sim_ij).
    - Convert each mean cosine-distance r -> angular radius φ = arccos(clamp(1 - r, -1,1)).
    - For each k in thresh_multipliers, count overlaps where
        θ_ij <= k * (φ1_i + φ2_j)
    Args:
        mean_features_1 (torch.Tensor[N1, D]): Feature centroids of set 1.
        mean_features_2 (torch.Tensor[N2, D]): Feature centroids of set 2.
        mean_angle_cosim_1 (torch.Tensor[N1]): Per-sample mean angle of cosine-similarity for set 1.
        mean_angle_cosim_2 (torch.Tensor[N2]): Per-sample mean angle of cosine-similarity for set 2.
            Multiples of the sum of angular radii to use as overlap thresholds.
            Defaults to (1, 2, 3, 5).

    Returns:
        overlap_counts_1 torch.Tensor[N1]: Number of overlaps for each sample of set 1.
        overlap_counts_2 torch.Tensor[N2]: Number of overlaps for each sample of set 2.

    """

    N1, D = mean_features_1.shape
    N2, _ = mean_features_2.shape

    # Build [N1, N2, D] tensors for pairwise comparison
    a = mean_features_1.unsqueeze(1).expand(N1, N2, D)  # [N1, N2, D]
    b = mean_features_2.unsqueeze(0).expand(N1, N2, D)  # [N1, N2, D]

    # Pairwise cosine‐similarity (already in [-1,1])
    sim = cosine_similarity_chunked(a, b, dim=2, eps=1e-8)    # [N1, N2]

    # Angular distances between centroids
    theta12 = torch.acos(sim)                           # [N1, N2]

    phi1 = mean_angle_cosim_1.view(-1, 1)                # [N1, 1]
    phi2 = mean_angle_cosim_2.view(1, -1)                # [1, N2]

    phi_sum = phi1 + phi2                                # [N1, N2]

    overlaps = theta12 <= phi_sum                   # [N1, N2] mask

    cnt1 = overlaps.sum(dim=0).float()             # [N1]
    cnt2 = overlaps.sum(dim=1).float()             # [N2]

    return cnt1, cnt2

    

def cosine_similarity_chunked(a: torch.Tensor,
                              b: torch.Tensor,
                              dim: int = 2,
                              eps: float = 1e-8,
                              chunk_size: int = 100) -> torch.Tensor:
    """
    Compute F.cosine_similarity(a, b, dim, eps) in chunks along dim-1 (i.e. N2).

    Args:
        a, b: [N1, N2, D] tensors (must be same shape).
        dim:   dimension to do similarity over (default 2, the D dimension).
        eps:   numerical stability constant.
        chunk_size: number of columns (in N2) to process at once.

    Returns:
        Tensor of shape [N1, N2] with the cosine similarities.
    """
    N1, N2, D = a.shape
    outputs = []
    # loop over slices of size chunk_size in N2
    for start in range(0, N2, chunk_size):
        end = min(start + chunk_size, N2)
        ai = a[:, start:end, :]      # [N1, chunk, D]
        bi = b[:, start:end, :]      # [N1, chunk, D]
        ci = F.cosine_similarity(ai, bi, dim=dim, eps=eps)  # [N1, chunk]
        outputs.append(ci)
    return torch.cat(outputs, dim=1)  # [N1, N2]


def lunif(x, t=2):
    """
    Computes the uniformity loss as described in "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020).
    Args:
       x (torch.Tensor): Tensor of shape [N, D] with the features.
       t (float): Temperature parameter.
    Returns:
       torch.Tensor: Uniformity loss.

    """
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

class OnlineFeatureMetrics:
    def __init__(self, save_pth):
        # Init save files
        save_folder = os.path.join(save_pth, 'online_feature_metrics')
        os.makedirs(save_folder, exist_ok=True)
        self.save_file_deviation_e = os.path.join(save_folder, 'e_online_deviation.csv')
        with open(self.save_file_deviation_e, 'a') as f:
            f.write('std_deviation_mean,cosine_deviation_mean\n')
        self.save_file_overlap_e = os.path.join(save_folder, 'e_online_overlap.csv')
        with open(self.save_file_overlap_e, 'a') as f:
            f.write('overlap_thresh_mult,ratio_mean_buff_overlaps\n')

        self.save_file_deviation_z = os.path.join(save_folder, 'z_online_deviation.csv')
        with open(self.save_file_deviation_z, 'a') as f:
            f.write('std_deviation_mean,cosine_deviation_mean\n')
        self.save_file_overlap_z = os.path.join(save_folder, 'z_online_overlap.csv')
        with open(self.save_file_overlap_z, 'a') as f:
            f.write('overlap_thresh_mult,ratio_mean_buff_overlaps\n')

    def calculate_stats_online(self, feature_list):
        std_list, mean_list =  [], []
        mean_cos_dist_list, mean_angle_list = [], []

        V = torch.stack(feature_list, dim=0).detach() # first stack current_mbatch_views → (v, b, d)
        per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d) and unbind

        for feature_views in per_sample_views_list:
            # Compute mean and std of e_views
            std_list.append(torch.std(feature_views, correction=0))
            mean_list.append(torch.mean(feature_views, dim=0))

            # Calculate pairwise cosine distance between e_views (obtains mean, std and mean angle of cosine dist)
            cos_dist_mean, _, mean_angle = pairwise_cosine_dist_stats(feature_views)
            mean_cos_dist_list.append(cos_dist_mean)
            mean_angle_list.append(mean_angle)

        return torch.stack(std_list, dim=0), torch.stack(mean_list, dim=0), torch.stack(mean_cos_dist_list, dim=0), torch.stack(mean_angle_list, dim=0)


    def calculate_metrics_online(self, e_stats, z_stats):
        THRESH_MULTIPLIERS = [0.5, 0.7, 0.9, 1]

        # ENCODER FEATURE METRICS
        # Calculate buffer overlap
        e_num_mean_buff_overlaps_list, _, _, _ = calculate_overlap_cosine(torch.stack(e_stats['mean']), torch.stack(e_stats['mean']),
                                                                        torch.stack(e_stats['angle']), torch.stack(e_stats['angle']),
                                                                        thresh_multipliers=THRESH_MULTIPLIERS)
        e_ratio_mean_buff_overlaps_list = [x/len(e_stats['mean']) for x in e_num_mean_buff_overlaps_list]
        
        # Deviation
        e_std_deviation_mean = torch.mean(torch.stack(e_stats['std']))
        e_cosine_deviation_mean = torch.mean(torch.stack(e_stats['cos_dist']))

        with open(self.save_file_deviation_e, 'a') as f:
            f.write(f'{e_std_deviation_mean},{e_cosine_deviation_mean}\n')

        with open(self.save_file_overlap_e, 'a') as f:
            for i, mult in enumerate(THRESH_MULTIPLIERS):
                f.write(f'{mult},{e_ratio_mean_buff_overlaps_list[i]}\n')

           
        # PROJECTOR FEATURE METRICS
        z_num_mean_buff_overlaps_list, _, _, _ = calculate_overlap_cosine(torch.stack(z_stats['mean']), torch.stack(z_stats['mean']),
                                                                        torch.stack(z_stats['angle']), torch.stack(z_stats['angle']),
                                                                        thresh_multipliers=THRESH_MULTIPLIERS)
        z_ratio_mean_buff_overlaps_list  = [x/len(z_stats['mean']) for x in z_num_mean_buff_overlaps_list]
        # Deviation
        z_std_deviation_mean = torch.mean(torch.stack(z_stats['std']))
        z_cosine_deviation_mean = torch.mean(torch.stack(z_stats['cos_dist']))

        with open(self.save_file_deviation_z, 'a') as f:
            f.write(f'{z_std_deviation_mean},{z_cosine_deviation_mean}\n')

        with open(self.save_file_overlap_z, 'a') as f:
            for i, mult in enumerate(THRESH_MULTIPLIERS):
                f.write(f'{mult},{z_ratio_mean_buff_overlaps_list[i]}\n')

