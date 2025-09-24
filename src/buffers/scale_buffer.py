import numpy as np
from sklearn.metrics import pairwise_distances

import torch

from time import time
import diversipy


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1


class Memory(object):
    def __init__(self,
                 mem_update_type='mo_rdn',
                 mem_size=2000,
                 mem_max_classes=10,
                 mem_max_new_ratio=0.1,
                 device = "cpu",
                 use_ema_embeddings = False,
                 ema_embeddings_decay = 0.5,
                 use_torch_psa = True
                 ):
        """
        Initialize memory.
        
        Args:
            mem_max_classes (int): Maximum number of (pseudo-)classes to store in memory.
            size_per_class (int): Number of samples to store per class in memory. 
            mem_update_type (str): Memory update strategy. Can be one of:
                - 'rdn': Random selection
                - 'mo_rdn': Momentum random selection
                - 'reservoir': Reservoir sampling
                - 'simil': Similarity-based selection
            mem_update_class_based (bool): Whether to cluster and update memory separately per class.
            mem_max_new_ratio (float): Maximum ratio of new samples if 'mo_rdn' update type is used.
            device (str): Device to perform computations on.
            use_ema_embeddings (bool): Whether to use EMA embeddings or recompute embeddings for similarity-based selection.
            ema_embeddings_decay (float): Decay rate for EMA embeddings if used.
            use_torch_psa (bool): Whether to use GPU-accelerated PSA clustering.
    """

        self.max_classes = mem_max_classes
        self.max_size = mem_size
        self.size_per_class = self.max_size // self.max_classes
        self.mem_update_type = mem_update_type
        self.max_new_ratio = mem_max_new_ratio
        self.device = device
        self.use_ema_embeddings = use_ema_embeddings
        self.ema_embeddings_decay = ema_embeddings_decay
        self.use_torch_psa = use_torch_psa

        self.images = []  # A list of numpy arrays
        self.embeddings = []  # A list of numpy arrays
        self.labels_set = []  # Pseud labels assisting memory update
        self.true_labels = []  # Same organization as self.images for true labels record
        self.update_cnt = 0
        self.num_seen_examples = 0

    def sampling(self, lb, old_sz, new_sz, sz_per_lb):
        """
        Implementation of various sampling methods.

        Args:
            lb: int, ground-truth label or pseudo label of the class
            old_sz: int, size of old data samples
            new_sz: int, size of new data samples
            sz_per_lb: int, upperbound on size of samples per label/class,
                take self.size_per_class with class-based sampling,
                take self.max_size without class-based sampling

        Return:
            select_ind: numpy array of the list of indices that are selected in
                the ind th memory bin
        """
        ind = self.labels_set.index(lb)
        select_ind = np.arange(old_sz + new_sz)
        # Memory Update - sample selection
        if old_sz + new_sz > sz_per_lb:
            if self.mem_update_type == 'rdn':
                select_ind = np.random.choice(old_sz + new_sz, sz_per_lb,
                                              replace=False)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.mem_update_type == 'mo_rdn':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                    sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)
                select_ind_old = np.random.choice(old_sz, num_old_samples,
                                                  replace=False)
                select_ind_new = old_sz + np.random.choice(new_sz, num_new_samples,
                                                           replace=False)
                select_ind = np.concatenate((select_ind_old, select_ind_new), axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            elif self.mem_update_type == 'reservoir':
                select_ind = list(np.arange(sz_per_lb))
                cur_ind = np.arange(sz_per_lb)  # Use to record the original index
                for i in range(sz_per_lb, old_sz + new_sz):
                    # i corresponds to the extra portion
                    index = reservoir(self.num_seen_examples, sz_per_lb)
                    if index >= 0:
                        self.images[ind][index] = self.images[ind][i]
                        self.true_labels[ind][index] = self.true_labels[ind][i]
                        select_ind.remove(cur_ind[index])
                        cur_ind[index] = i
                        select_ind.append(i)

                self.images[ind] = self.images[ind][:sz_per_lb]
                self.true_labels[ind] = self.true_labels[ind][:sz_per_lb]
                select_ind = np.array(select_ind)
            elif self.mem_update_type == 'simil':
                num_new_samples = min(new_sz, int(sz_per_lb * self.max_new_ratio))
                num_old_samples = max(int(sz_per_lb * (1 - self.max_new_ratio)),
                                      sz_per_lb - num_new_samples)
                num_old_samples = min(old_sz, num_old_samples)

                simil_sum = np.sum(self.similarity_matrix[ind], axis=1)
                select_ind_old = (-simil_sum[:old_sz]).argsort()[:num_old_samples]
                select_ind_new = old_sz + (-simil_sum[old_sz:]).argsort()[:num_new_samples]

                select_ind = np.concatenate((select_ind_old, select_ind_new),
                                            axis=0)
                self.images[ind] = self.images[ind][select_ind]
                self.true_labels[ind] = self.true_labels[ind][select_ind]
            else:
                raise ValueError(
                    'memory update policy not supported: {}'.format(self.mem_update_type))

        return select_ind

    def update_w_labels(self, new_images, new_labels):
        """
        Update memory samples.
        No need to check the number of classes if labels are provided.
        Args:
            new_images: torch array, new incoming images
            new_labels: torch array, new ground-truth labels
        """
        new_images = new_images.detach().numpy()
        new_labels = new_labels.detach().numpy()
        new_labels_set = set(new_labels)
        self.num_seen_examples += new_images.shape[0]

        for lb in new_labels_set:
            new_ind = (np.array(new_labels) == lb)
            new_sz = np.sum(new_ind)
            if lb in self.labels_set:  # already seen
                ind = self.labels_set.index(lb)
                old_sz = self.images[ind].shape[0]
                self.images[ind] = np.concatenate(
                    (self.images[ind], new_images[new_ind]),
                    axis=0)
                self.true_labels[ind] = np.concatenate(
                    (self.true_labels[ind], new_labels[new_ind]),
                    axis=0)
            else:  # first-time seen labels
                self.labels_set.append(lb)
                old_sz = 0
                self.images.append(new_images[new_ind])
                self.true_labels.append(new_labels[new_ind])

            # Memory update - sample selection
            # The key is transfer lb - the ground-truth label,
            # and sz_per_lb - size upperbound for each class
            self.sampling(lb, old_sz, new_sz, self.size_per_class)

    def update_wo_labels(self, new_images, new_embeddings, model=None):
        """
        Update memory samples.
        Args:
            new_images: torch array, new incoming images
            new_embeddings: torch array, new incoming embeddings
            model: network model being trained, used in kmeans and spectral cluster type

        Return:
            select_indices: numpy array of selected indices in all_images
        """
        new_images = new_images.detach().numpy()
        new_embeddings = new_embeddings.cpu().detach().numpy()
        self.num_seen_examples += new_images.shape[0]

        if len(self.images) > 0:  # Not first-time insertion
            old_images = np.concatenate(self.images)
            old_sz = old_images.shape[0]
            old_images = torch.from_numpy(old_images)

            all_images = np.concatenate((old_images, new_images), axis=0)
        else:  # first-time insertion
            old_sz = 0
            all_images = new_images

        # Create a binary indicator of whether the image is an old or new sample
        old_ind = np.zeros(all_images.shape[0], dtype=bool)
        old_ind[:old_sz] = 1

        # ALTERNATIVE: USE EMA EMBEDDINGS LIKE CLA, INSTEAD OF FULL FORWARD PASS OF BUFFER
        if self.use_ema_embeddings:
            # assert same number of old embeddings as old images
            assert len(self.embeddings) == len(self.images)

            if len(self.embeddings) > 0:  # Not first-time insertion
                old_embeddings = np.concatenate(self.embeddings)
                old_sz = old_embeddings.shape[0]
                all_embeddings = np.concatenate((old_embeddings, new_embeddings), axis=0)
            else:  # first-time insertion
                old_sz = 0
                all_embeddings = new_embeddings

        else:
            # Get latent embeddings
            # feed_images = torch.from_numpy(all_images).to(self.device, non_blocking=True)
            feed_images = torch.from_numpy(all_images).float().div(255).to(self.device, non_blocking=True)

            # ATTENTION! ADDITIONAL FORWARD PASS FOR ALL MEMORY SAMPLES! IS THIS TRICK ILLEGAL?
            all_embeddings = model(feed_images).detach().cpu().numpy()
            # all_embeddings_mean = np.mean(all_embeddings, axis=0, keepdims=True)
            # all_embeddings = (all_embeddings - all_embeddings_mean) * 1e4

        # PSA clustering
        # Clustering
        # simil_matrix = tsne_simil(all_embeddings, metric='cosine')

        # Init selected indices as all indices
        select_indices = np.arange(all_embeddings.shape[0])

        if all_embeddings.shape[0] > self.max_size:  # needs subset selection
            if self.use_torch_psa:
                selected_embeddings = gpu_psa_select(
                    points=all_embeddings,
                    num_selected_points=self.max_size,
                    available_points_indices=None,
                    selection_target="centroid_of_hypercube",
                    tournament_size=0,
                    device=self.device
                )  # gpu_psa_select() returns already selected embeddings, not indices
            else:
                # Use diversipy package
                selected_embeddings = diversipy.subset.psa_select(all_embeddings, self.max_size)  # psa_select() returns already selected embeddings, not indices

            select_indices = np.where(np.all(all_embeddings[:, None, :] == selected_embeddings[None, :, :], axis=-1).any(axis=1))[0]  # convert embeddings to indices
            select_indices.sort()

        self.images = [all_images[select_indices]]
        self.embeddings = [all_embeddings[select_indices]]
        self.labels_set = [0]


        return all_embeddings, select_indices


    def get_mem_samples(self):
        """
        Combine all stored samples and pseudo labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all pseudo labels, (sample #, pseudo label)
        If updated with update_w_labels, the returned labels are the ground-truth labels.
        If updated with update_wo_labels, the returned labels are the pseudo labels.
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = np.repeat(lb, self.images[0].shape[0])
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(labels)

    def get_mem_samples_w_true_labels(self):
        """
        Combine all stored samples and true labels.
        Returns:
            images: numpy array of all images, (sample #, image)
            labels: numpy array of all true labels, (sample #, true label)
        """
        images, labels = None, None
        for lb in self.labels_set:
            ind = self.labels_set.index(lb)
            if images is None:  # First label
                images = self.images[ind]
                labels = self.true_labels[ind]
            else:  # Subsequent labels to be concatenated
                images = np.concatenate((images, self.images[ind]), axis=0)
                labels = np.concatenate((labels, self.true_labels[ind]), axis=0)

        if images is None:  # Empty memory
            return None, None
        else:
            return torch.from_numpy(images), torch.from_numpy(np.array(labels))
        
    # New Sample method
    # Returns None when void buffer, otherwise returns samples
    def sample(self, replay_batch_size):
        if len(self.images) > 0:
            mem_images = np.concatenate(self.images, axis=0)
            mem_len = mem_images.shape[0]
            sample_cnt = min(mem_len, replay_batch_size)
            select_ind = np.random.choice(range(mem_len), sample_cnt, replace=False)

            return torch.from_numpy(mem_images[select_ind]), select_ind
        else:
            return None, None

    def update_embeddings(self, new_embeddings, replay_indices):
        """
        Update EMA embeddings for samples in the buffer.
        Args:
            new_embeddings: torch array, new incoming embeddings
            replay_indices: list of indices of samples in the buffer to update
        """
        new_embeddings = new_embeddings.cpu().detach().numpy()
        assert len(self.embeddings) > 0, "No embeddings to update!"
       
        for i, repl_idx in enumerate(replay_indices):
            self.embeddings[0][repl_idx] = self.ema_embeddings_decay * self.embeddings[0][repl_idx] + (1 - self.ema_embeddings_decay) * new_embeddings[i]

    def end(self):
        return '', ''


def tsne_simil(x, metric='euclidean', sigma=1.0):
    dist_matrix = pairwise_distances(x, metric=metric)
    cur_sim = np.divide(- dist_matrix, 2 * sigma ** 2)
    # print(np.sum(cur_sim, axis=1, keepdims=True))

    # mask-out self-contrast cases
    # the diagonal elements of exp_logits should be zero
    logits_mask = np.ones((x.shape[0], x.shape[0]))
    np.fill_diagonal(logits_mask, 0)
    # print(logits_mask)
    exp_logits = np.exp(cur_sim) * logits_mask
    # print(exp_logits.shape)
    # print(np.sum(exp_logits, axis=1, keepdims=True))

    p = np.divide(exp_logits, np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    p = p + p.T
    p /= 2 * x.shape[0]
    return p


######################################################################################################################################

# GPU implementation of PSA clustering is in diversipy package
import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any
import heapq
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Try to import diversipy for comparison
try:
    from diversipy.subset import psa_select as diversipy_psa_select
    from diversipy.subset import psa_partition as diversipy_psa_partition
    DIVERSIPY_AVAILABLE = True
except ImportError:
    DIVERSIPY_AVAILABLE = False
    print("Warning: diversipy not available. Install with 'pip install diversipy' for comparison.")


@dataclass
class GPUMinBoundingBox:
    """GPU-accelerated minimum bounding box for PSA clustering."""
    
    def __init__(self, points: torch.Tensor, member_indices: torch.Tensor, device: str = 'cuda'):
        self.device = device
        self.member_indices = member_indices
        self.member_points = points[member_indices]
        
        # Calculate bounds
        self.min_bounds = torch.min(self.member_points, dim=0)[0]
        self.max_bounds = torch.max(self.member_points, dim=0)[0]
        
        # Calculate range and find dimension with maximum range
        self.ranges = self.max_bounds - self.min_bounds
        self.dim_index = torch.argmax(self.ranges).item()
        self.max_range = self.ranges[self.dim_index].item()
        
        # Center of hypercube
        self.center = (self.min_bounds + self.max_bounds) / 2.0
        
    def __lt__(self, other):
        """For heap ordering - larger range = higher priority"""
        return self.max_range > other.max_range
        
    def obtain_representative(self, selection_target: str = "centroid_of_hypercube", 
                            tournament_size: int = 0) -> int:
        """Select representative point from cluster."""
        if len(self.member_indices) == 1:
            return self.member_indices[0].item()
            
        if selection_target == "random_uniform":
            idx = torch.randint(0, len(self.member_indices), (1,), device=self.device)
            return self.member_indices[idx].item()
        elif selection_target == "centroid_of_hypercube":
            # Find point closest to center of hypercube
            distances = torch.sum((self.member_points - self.center) ** 2, dim=1)
            best_idx = torch.argmin(distances)
            return self.member_indices[best_idx].item()
        elif selection_target == "center_of_mass":
            # Find point closest to center of mass
            center_of_mass = torch.mean(self.member_points, dim=0)
            distances = torch.sum((self.member_points - center_of_mass) ** 2, dim=1)
            best_idx = torch.argmin(distances)
            return self.member_indices[best_idx].item()
        elif selection_target == "max_dist_from_boundary":
            # Find point with maximum minimum distance to boundary
            min_dists_to_boundary = torch.minimum(
                self.member_points - self.min_bounds,
                self.max_bounds - self.member_points
            )
            min_dists = torch.min(min_dists_to_boundary, dim=1)[0]
            best_idx = torch.argmax(min_dists)
            return self.member_indices[best_idx].item()
        else:
            raise ValueError(f"Unknown selection target: {selection_target}")


def gpu_psa_partition(points: torch.Tensor, num_clusters: int, 
                     available_points_indices: Optional[torch.Tensor] = None,
                     device: str = 'cuda') -> List[GPUMinBoundingBox]:
    """GPU-accelerated PSA partitioning."""
    if available_points_indices is None:
        available_points_indices = torch.arange(len(points), device=device)
    
    assert num_clusters <= len(available_points_indices)
    assert num_clusters > 0
    
    clusters = []
    most_dissimilar_cluster = GPUMinBoundingBox(points, available_points_indices, device)
    
    while len(clusters) + 1 < num_clusters:
        split_index = most_dissimilar_cluster.dim_index
        lower_bound = most_dissimilar_cluster.min_bounds[split_index]
        upper_bound = most_dissimilar_cluster.max_bounds[split_index]
        split_position = (lower_bound + upper_bound) / 2.0
        
        # Vectorized splitting
        member_points_split_dim = points[most_dissimilar_cluster.member_indices, split_index]
        mask = member_points_split_dim < split_position
        
        indices1 = most_dissimilar_cluster.member_indices[mask]
        indices2 = most_dissimilar_cluster.member_indices[~mask]
        
        # Handle edge case where all points go to one side
        if len(indices1) == 0:
            # Move one point from indices2 to indices1
            indices1 = indices2[:1]
            indices2 = indices2[1:]
        elif len(indices2) == 0:
            # Move one point from indices1 to indices2
            indices2 = indices1[:1]
            indices1 = indices1[1:]
        
        cluster1 = GPUMinBoundingBox(points, indices1, device)
        cluster2 = GPUMinBoundingBox(points, indices2, device)
        
        heapq.heappush(clusters, cluster1)
        most_dissimilar_cluster = heapq.heappushpop(clusters, cluster2)
    
    heapq.heappush(clusters, most_dissimilar_cluster)
    return clusters


def gpu_psa_select(points: Union[torch.Tensor, np.ndarray], 
                   num_selected_points: int,
                   available_points_indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
                   selection_target: str = "centroid_of_hypercube",
                   tournament_size: int = 0,
                   device: str = 'cuda') -> Union[torch.Tensor, np.ndarray]:
    """GPU-accelerated PSA selection."""
    # Convert to torch tensor if needed
    original_was_numpy = isinstance(points, np.ndarray)
    if original_was_numpy:
        points_tensor = torch.from_numpy(points).float().to(device)
    else:
        points_tensor = points.to(device)
    
    if available_points_indices is not None:
        if isinstance(available_points_indices, np.ndarray):
            available_points_indices = torch.from_numpy(available_points_indices).long().to(device)
        else:
            available_points_indices = available_points_indices.to(device)
    
    # Perform partitioning
    clusters = gpu_psa_partition(points_tensor, num_selected_points, available_points_indices, device)
    
    # Select representatives
    representative_indices = []
    for cluster in clusters:
        representative_idx = cluster.obtain_representative(selection_target, tournament_size)
        representative_indices.append(representative_idx)
    
    # Get representative points
    representative_indices_tensor = torch.tensor(representative_indices, device=device)
    representatives = points_tensor[representative_indices_tensor]
    
    # Convert back to numpy if input was numpy
    if original_was_numpy:
        return representatives.cpu().numpy()
    else:
        return representatives