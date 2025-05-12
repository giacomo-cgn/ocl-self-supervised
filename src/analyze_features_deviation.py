import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from einops import rearrange
from typing import Sequence, List, Tuple


from .transforms import get_transforms


class FeatureDeviationAnalyzer():
    def __init__(self,
                 when_features_deviation: str = '50_end',
                 dataset_name: str = 'cifar100',
                 transforms_type: str = 'common',
                 num_views: int = 50,
                 num_current_samples: int = 500,
                 overlap_thresh_multipliers_euclidean: list = [1, 2, 3, 5],
                 overlap_thresh_multipliers_cosine: list = [0.1, 0.3, 1],
                 mb_size: int=10,
                 device: str ='cpu',
                 save_pth: str = None,
                 ):
        
        split_by_underscore = lambda s: s.split('_')

        
        self.when_features_deviation = split_by_underscore(when_features_deviation)
        self.overlap_thresh_multipliers_euc = overlap_thresh_multipliers_euclidean
        self.overlap_thresh_multipliers_cosine = overlap_thresh_multipliers_cosine
        self.mb_size = mb_size
        self.num_current_samples = num_current_samples
        self.device = device

        if transforms_type == 'common':
            self.transforms = get_transforms(dataset=dataset_name, n_crops=num_views, online_transforms=True)
        else:
            raise Exception(f'Transforms type {self.transforms_type} not supported')
        
        if save_pth is not None:
            # Feature std
            feat_analyze_pth = os.path.join(save_pth, 'feature_deviation')
            os.makedirs(feat_analyze_pth, exist_ok=True)
            self.e_save_pth = os.path.join(feat_analyze_pth, 'e_features.csv')
            self.z_save_pth = os.path.join(feat_analyze_pth, 'z_features.csv')
            with open(self.e_save_pth, 'a') as f:
                f.write('exp_idx,tr_step,avg_e_curr_std,avg_e_buff_std,'
                        'avg_e_std_cosine_curr,avg_e_std_cosine_buff,avg_e_mean_cosine_curr,avg_e_mean_cosine_buff\n')
            with open(self.z_save_pth, 'a') as f:
                f.write('exp_idx,tr_step,avg_z_curr_std,avg_z_buff_std,'
                        'avg_z_std_cosine_curr,avg_z_std_cosine_buff,avg_z_mean_cosine_curr,avg_z_mean_cosine_buff\n')
                
            # Overlap
            overlap_cosine_folder = os.path.join(feat_analyze_pth, 'overlap_cosine')
            overlap_euclidean_folder = os.path.join(feat_analyze_pth, 'overlap_euclidean')
            os.makedirs(overlap_cosine_folder, exist_ok=True)
            os.makedirs(overlap_euclidean_folder, exist_ok=True)
            
            self.overlap_e_cosine_file = os.path.join(overlap_cosine_folder, 'overlap_e_features.csv')
            self.overlap_z_cosine_file = os.path.join(overlap_cosine_folder, 'overlap_z_features.csv')
            self.overlap_e_euclidean_file = os.path.join(overlap_euclidean_folder, 'overlap_e_features.csv')
            self.overlap_z_euclidean_file = os.path.join(overlap_euclidean_folder, 'overlap_z_features.csv')

            for file in [self.overlap_e_cosine_file, self.overlap_z_cosine_file, self.overlap_e_euclidean_file, self.overlap_z_euclidean_file]:
                with open(file, 'a') as f:
                    f.write('exp_idx,tr_step,std_mult,avg_overlap_curr,avg_overlap_buff,avg_overlap_c2b,avg_overlap_b2c,'
                            'std_overlap_curr,std_overlap_buff,std_overlap_c2b,std_overlap_b2c\n')
                    
            # Uniformity loss
            uniformity_folder = os.path.join(feat_analyze_pth, 'uniformity')
            os.makedirs(uniformity_folder, exist_ok=True)
            self.uniformity_file = os.path.join(uniformity_folder, 'uniformity.csv')
            with open(self.uniformity_file, 'a') as f:
                f.write('exp_idx,tr_step,uniformity_e_curr,uniformity_z_curr,uniformity_e_buff,uniformity_z_buff\n')

        
            # Save configuration
            with open(save_pth + '/config.txt', 'a') as f:
                f.write('\n')
                f.write('---- ANALYZE FEATURES DEVIATION CONFIG ----\n')
                f.write(f'when (training steps) to analyze: {self.when_features_deviation}\n')
                f.write(f'num analyzed views: {num_views}\n')
                f.write(f'num current samples: {num_current_samples}\n')
                f.write(f'mb size analysis: {self.mb_size}\n')
                f.write(f'overlap std multipliers (for threshold) euclidean: {self.overlap_thresh_multipliers_euc}\n')
                f.write(f'overlap std multipliers (for threshold) cosine: {self.overlap_thresh_multipliers_cosine}\n')

        else:
            self.e_save_pth = None
            self.z_save_pth = None


    


    def analyze_features_deviation(self, encoder, buffer_data, current_data, exp_idx, tr_step, projector=None):

        print(f'>>> Analyzing features deviation at step {tr_step}')

        buffer_dataloader = self.prepare_buffer_data(buffer_data)
        current_dataloader = self.prepare_current_data(current_data)

        encoder.eval()
        if projector is not None:
            projector.eval()

        with torch.no_grad():

            e_std_curr_list, e_mean_curr_list, z_std_curr_list, z_mean_curr_list = [], [], [], []
            e_std_cosine_curr_list, e_mean_cosine_curr_list, z_std_cosine_curr_list, z_mean_cosine_curr_list = [], [], [], []
            e_mean_angle_curr_list, z_mean_angle_curr_list = [], []

            e_std_buff_list, e_mean_buff_list, z_std_buff_list, z_mean_buff_list = [], [], [], []
            e_std_cosine_buff_list, e_mean_cosine_buff_list, z_std_cosine_buff_list, z_mean_cosine_buff_list = [], [], [], []
            e_mean_angle_buff_list, z_mean_angle_buff_list = [], []

            # Extract features from current data
            print('>>> Extracting features from current data, len', len(current_dataloader))
            for _, current_mbatch in enumerate(tqdm(current_dataloader)):
                current_mbatch = current_mbatch.to(self.device)
                current_mbatch_views = self.transforms(current_mbatch)

                # Convert from list of minibatch views to list of per sample views
                V = torch.stack(current_mbatch_views, dim=0) # first stack current_mbatch_views → (v, b, d1, d2, …)
                per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d1, d2, …) and unbind

                for x_views in per_sample_views_list:
                    e_views = encoder(x_views)
                    # compute mean and std of e_views
                    e_std_curr_list.append(torch.mean(torch.std(e_views, dim=0)))
                    e_mean_curr_list.append(torch.mean(e_views, dim=0))

                    # calculate pairwise cosine distance between views ((obtains mean and std of cosine dist)
                    e_cosine_mean, e_cosine_std, e_mean_angle = self.pairwise_cosine_dist_stats(e_views)
                    e_mean_cosine_curr_list.append(e_cosine_mean)
                    e_std_cosine_curr_list.append(e_cosine_std)
                    e_mean_angle_curr_list.append(e_mean_angle)

                    if projector is not None:
                        z_views = projector(e_views)
                        # compute mean and std of z_views
                        z_std_curr_list.append(torch.mean(torch.std(z_views, dim=0)))
                        z_mean_curr_list.append(torch.mean(z_views, dim=0))

                        # calculate pairwise cosine distance between e_views ((obtains mean and std of cosine dist)
                        z_cosine_mean, z_cosine_std, z_mean_angle = self.pairwise_cosine_dist_stats(z_views)
                        z_mean_cosine_curr_list.append(z_cosine_mean)
                        z_std_cosine_curr_list.append(z_cosine_std)
                        z_mean_angle_curr_list.append(z_mean_angle)

            # Extract features from buffer data
            print('>>> Extracting features from buffer data, len', len(buffer_dataloader))
            for _, buffer_mbatch in enumerate(tqdm(buffer_dataloader)):
                buffer_mbatch = buffer_mbatch.to(self.device)
                buffer_mbatch_views = self.transforms(buffer_mbatch)

                # Convert from list of minibatch views to list of per sample views
                V = torch.stack(buffer_mbatch_views, dim=0) # first stack buffer_mbatch_views → (v, b, d1, d2, …)
                per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d1, d2, …) and unbind

                for x_views in per_sample_views_list:
                    e_views = encoder(x_views)
                    # compute mean and std of e_views
                    e_std_buff_list.append(torch.mean(torch.std(e_views, dim=0)))
                    e_mean_buff_list.append(torch.mean(e_views, dim=0))

                    # calculate pairwise cosine distance between z_views ((obtains mean and std of cosine dist)
                    e_cosine_mean, e_cosine_std, e_mean_angle = self.pairwise_cosine_dist_stats(e_views)
                    e_mean_cosine_buff_list.append(e_cosine_mean)
                    e_std_cosine_buff_list.append(e_cosine_std)
                    e_mean_angle_buff_list.append(e_mean_angle)

                    if projector is not None:
                        z_views = projector(e_views)
                        # compute mean and std of z_views
                        z_std_buff_list.append(torch.mean(torch.std(z_views, dim=0)))
                        z_mean_buff_list.append(torch.mean(z_views, dim=0))

                        # calculate pairwise cosine distance between z_views ((obtains mean and std of cosine dist)
                        z_cosine_mean, z_cosine_std, z_mean_angle = self.pairwise_cosine_dist_stats(z_views)
                        z_mean_cosine_buff_list.append(z_cosine_mean)
                        z_std_cosine_buff_list.append(z_cosine_std)
                        z_mean_angle_buff_list.append(z_mean_angle)

            
            # convert lists to tensors
            torch_e_mean_curr_list = torch.stack(e_mean_curr_list)
            torch_e_mean_buff_list = torch.stack(e_mean_buff_list)
            torch_z_mean_curr_list = torch.stack(z_mean_curr_list)
            torch_z_mean_buff_list = torch.stack(z_mean_buff_list)
            torch_e_std_curr_list = torch.stack(e_std_curr_list)
            torch_e_std_buff_list = torch.stack(e_std_buff_list)
            torch_z_std_curr_list = torch.stack(z_std_curr_list)
            torch_z_std_buff_list = torch.stack(z_std_buff_list)
            torch_e_mean_angle_curr_list = torch.stack(e_mean_angle_curr_list)
            torch_e_mean_angle_buff_list = torch.stack(e_mean_angle_buff_list)
            torch_z_mean_angle_curr_list = torch.stack(z_mean_angle_curr_list)
            torch_z_mean_angle_buff_list = torch.stack(z_mean_angle_buff_list)
            
            # ---- UNIFORMITY LOSS ----
            lunif_e_curr = lunif(torch_e_mean_curr_list)
            lunif_e_buff = lunif(torch_e_mean_buff_list)
            if len(torch_z_mean_curr_list) > 0 and len(torch_z_mean_buff_list) > 0:
                lunif_z_curr = lunif(torch_z_mean_curr_list)
                lunif_z_buff = lunif(torch_z_mean_buff_list)
            else:
                lunif_z_curr = 0
                lunif_z_buff = 0
            with open(self.uniformity_file, 'a') as f:
                f.write(f'{exp_idx},{tr_step},{lunif_e_curr},{lunif_z_curr},{lunif_e_buff},{lunif_z_buff}\n')

            # ---- OVERLAPS ----
            # ---- Save and calculate overlaps
            # overlaps for encoder features (e) with cosine distance
            self.get_save_overlaps(torch_e_mean_curr_list, torch_e_mean_buff_list, torch_e_mean_angle_curr_list, torch_e_mean_angle_buff_list,
                                   exp_idx=exp_idx, tr_step=tr_step, save_file=self.overlap_e_cosine_file,
                                   thresh_multipliers=self.overlap_thresh_multipliers_cosine, overlap_func=self.calculate_overlap_cosine)
            # overlaps for projector features (z) with cosine distance
            self.get_save_overlaps(torch_z_mean_curr_list, torch_z_mean_buff_list, torch_z_mean_angle_curr_list, torch_z_mean_angle_buff_list,
                                    exp_idx=exp_idx, tr_step=tr_step, save_file=self.overlap_z_cosine_file,
                                    thresh_multipliers=self.overlap_thresh_multipliers_cosine, overlap_func=self.calculate_overlap_cosine)
            # overlaps for encoder features (e) with euclidean distance
            self.get_save_overlaps(torch_e_mean_curr_list, torch_e_mean_buff_list, torch_e_std_curr_list, torch_e_std_buff_list,
                                   exp_idx=exp_idx, tr_step=tr_step, save_file=self.overlap_e_euclidean_file,
                                   thresh_multipliers=self.overlap_thresh_multipliers_euc, overlap_func=self.calculate_overlap_euclidean)
            # overlaps for projector features (z) with euclidean distance
            self.get_save_overlaps(torch_z_mean_curr_list, torch_z_mean_buff_list, torch_z_std_curr_list, torch_z_std_buff_list,
                                    exp_idx=exp_idx, tr_step=tr_step, save_file=self.overlap_z_euclidean_file,
                                    thresh_multipliers=self.overlap_thresh_multipliers_euc, overlap_func=self.calculate_overlap_euclidean)

            


            # Calculate avg of statistics over all samples and save
            if self.e_save_pth is not None:
                avg_e_std_curr = torch.mean(torch.stack(e_std_curr_list), dim=0)
                avg_e_std_buff = torch.mean(torch.stack(e_std_buff_list), dim=0)
                avg_e_std_cosine_curr = torch.mean(torch.stack(e_std_cosine_curr_list), dim=0)
                avg_e_std_cosine_buff = torch.mean(torch.stack(e_std_cosine_buff_list), dim=0)
                avg_e_mean_cosine_curr = torch.mean(torch.stack(e_mean_cosine_curr_list), dim=0)
                avg_e_mean_cosine_buff = torch.mean(torch.stack(e_mean_cosine_buff_list), dim=0)
                with open(self.e_save_pth, 'a') as f:
                    f.write(f'{exp_idx},{tr_step},{avg_e_std_curr.item()},{avg_e_std_buff.item()}'
                            f',{avg_e_std_cosine_curr.item()},{avg_e_std_cosine_buff.item()},'
                            f'{avg_e_mean_cosine_curr.item()},{avg_e_mean_cosine_buff.item()}\n')

            if projector is not None and self.z_save_pth is not None:
                avg_z_std_curr = torch.mean(torch.stack(z_std_curr_list), dim=0)
                avg_z_std_buff = torch.mean(torch.stack(z_std_buff_list), dim=0)
                avg_z_std_cosine_curr = torch.mean(torch.stack(z_std_cosine_curr_list), dim=0)
                avg_z_std_cosine_buff = torch.mean(torch.stack(z_std_cosine_buff_list), dim=0)
                avg_z_mean_cosine_curr = torch.mean(torch.stack(z_mean_cosine_curr_list), dim=0)
                avg_z_mean_cosine_buff = torch.mean(torch.stack(z_mean_cosine_buff_list), dim=0)
                with open(self.z_save_pth, 'a') as f:
                    f.write(f'{exp_idx},{tr_step},{avg_z_std_curr.item()},{avg_z_std_buff.item()}'
                            f',{avg_z_std_cosine_curr.item()},{avg_z_std_cosine_buff.item()},'
                            f'{avg_z_mean_cosine_curr.item()},{avg_z_mean_cosine_buff.item()}\n')
            
        encoder.train()
        if projector is not None:
            projector.train()


    def prepare_buffer_data(self, buffer_data):
        # If buffer_data is a list, convert it to a tensor, stack along a new dim
        if isinstance(buffer_data, list):
            buffer_data = torch.stack(buffer_data, dim=0)
        buffer_dataloader = DataLoader(buffer_data, batch_size=self.mb_size, shuffle=False)
            
        return buffer_dataloader
    
    def prepare_current_data(self, current_data):
        # Select self.num_current_samples samples from current_data Dataset
        subset_current_data = Subset(current_data, np.random.choice(len(current_data), self.num_current_samples, replace=False))
        current_dataloader = DataLoader(subset_current_data, batch_size=self.mb_size, shuffle=False)

        return current_dataloader
    
    def get_when_to_analyze(self, tr_step, total_steps):
        """
        Determines whether feature deviation analysis should be performed at the current training step.
        Args:
            tr_step (int): Current training step.
            total_steps (int): Total number of training steps.

        Returns:
            bool: True if analysis should be performed at this step, False otherwise.
                 Returns True if:
                 - It's the final step (tr_step == total_steps - 1) and 'end' is in when_features_deviation
                 - The current step number (as string) is in when_features_deviation
        """
        if tr_step == (total_steps-1):
            if 'end' in self.when_features_deviation:
                return True
        if  str(tr_step) in self.when_features_deviation:
            return True
        return False
    
    def pairwise_cosine_dist_stats(self, X: torch.Tensor):
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
        std_dist  = dists.std() 
        
        return mean_dist, std_dist, mean_angle
        

    def get_save_overlaps(self,
                        torch_mean_curr_list, torch_mean_buff_list,
                        torch_radius_curr_list, torch_radius_buff_list,
                        save_file, exp_idx, tr_step, thresh_multipliers,
                        overlap_func):
        # buffer to buffer overlap
        e_mean_overlap_buf_list, _, e_std_overlap_buf_list, _ = overlap_func(
            torch_mean_buff_list, torch_mean_buff_list, torch_radius_buff_list, torch_radius_buff_list,
            thresh_multipliers=thresh_multipliers
        )
        # current to current overlap
        e_mean_overlap_curr_list, _, e_std_overlap_curr_list, _ = overlap_func(
            torch_mean_curr_list, torch_mean_curr_list, torch_radius_curr_list, torch_radius_curr_list,
            thresh_multipliers=thresh_multipliers
        )
        # buffer to current and current to buffer overlap
        e_mean_overlap_b2c_list, e_mean_overlap_c2b_list, e_std_overlap_b2c_list, e_std_overlap_c2b_list = overlap_func(
            torch_mean_buff_list, torch_mean_curr_list, torch_radius_buff_list, torch_radius_curr_list,
            thresh_multipliers=thresh_multipliers
        )
        # Save to file
        for i, mult in enumerate(thresh_multipliers):
            with open(save_file, 'a') as f:
                f.write(f'{exp_idx},{tr_step},{mult},{e_mean_overlap_curr_list[i]},{e_mean_overlap_buf_list[i]},{e_mean_overlap_c2b_list[i]},{e_mean_overlap_b2c_list[i]},'
                        f'{e_std_overlap_curr_list[i]},{e_std_overlap_buf_list[i]},{e_std_overlap_c2b_list[i]},{e_std_overlap_b2c_list[i]}\n')


    def calculate_overlap_euclidean(
        self,
        mean_features_1: torch.Tensor,  # shape [N1, D]
        mean_features_2: torch.Tensor,  # shape [N2, D]
        std_features_1: torch.Tensor,   # shape [N1], per-sample std (averaged over D)
        std_features_2: torch.Tensor,   # shape [N2]
        thresh_multipliers=(1, 2, 3, 5),
    ):
        """
        Calculate the overlap between two sets of features, given the mean and per-sample std, using euclidean distance.

        Args:
            mean_features_1 (Tensor[N1, D]): feature means for set 1
            mean_features_2 (Tensor[N2, D]): feature means for set 2
            std_features_1  (Tensor[N1]):   per-sample std (mean over D) for set 1
            std_features_2  (Tensor[N2]):   per-sample std (mean over D) for set 2
            thresh_multipliers (Sequence[float]): multiples of (std_i + std_j) for overlap threshold

        Returns:
            mean_num_overlap_1 (List[float]): avg # overlaps per sample in set 1 for each k
            mean_num_overlap_2 (List[float]): avg # overlaps per sample in set 2 for each k
            std_num_overlap_1  (List[float]): std of # overlaps in set 1 for each k
            std_num_overlap_2  (List[float]): std of # overlaps in set 2 for each k
        """
        # 1) Compute pairwise euclidean distance matrix
        diff = mean_features_1.unsqueeze(1) - mean_features_2.unsqueeze(0)  # [N1, N2, D]
        dist = diff.norm(dim=2, p=2)                                        # [N1, N2]

        # 2) For each multiplier, count overlaps and compute stats
        mean_num_overlap_1 = []
        mean_num_overlap_2 = []
        std_num_overlap_1  = []
        std_num_overlap_2  = []

        for k in thresh_multipliers:
            # threshold matrix: [N1, N2]
            thresh  = k * (std_features_1.unsqueeze(1) + std_features_2.unsqueeze(0))
            overlap = dist <= thresh

            # counts per sample
            counts_1 = overlap.sum(dim=1).to(torch.float32)  # [N1]
            counts_2 = overlap.sum(dim=0).to(torch.float32)  # [N2]

            # aggregate
            mean_num_overlap_1.append(counts_1.mean().item())
            std_num_overlap_1.append(counts_1.std().item())
            mean_num_overlap_2.append(counts_2.mean().item())
            std_num_overlap_2.append(counts_2.std().item())

        return (
            mean_num_overlap_1,
            mean_num_overlap_2,
            std_num_overlap_1,
            std_num_overlap_2,
        )
    

    def calculate_overlap_cosine(
        self,
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

            cnt1 = overlaps.sum(dim=1).float()             # [N1]
            cnt2 = overlaps.sum(dim=0).float()             # [N2]

            means1.append(cnt1.mean().item())
            stds1.append(cnt1.std().item())
            means2.append(cnt2.mean().item())
            stds2.append(cnt2.std().item())

        return means1, means2, stds1, stds2

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