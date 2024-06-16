from torch.utils.data import random_split, Subset
import numpy as np
import torch
import random

def get_gradual_subset_increase_exps(dataset, total_tr_steps, start_subset_ratio: float, end_subset_ratio: float, step_ratio: float, seed: int = 42, start_subset: Subset = None):
    subset_lens = [len(dataset)*start_subset_ratio]
    subset_lens.extend([int(len(dataset)*i) for i in np.arange(start_subset_ratio+step_ratio, end_subset_ratio+step_ratio, step_ratio)])

    tr_steps = total_tr_steps // len(subset_lens)
    if start_subset is None:
        start_subset, _ = random_split(dataset, [int(subset_lens[0]), int(len(dataset) - subset_lens[0])],
                                     generator=torch.Generator().manual_seed(seed))
    
    current_subset = start_subset
    exp_list = []
    
    for subset_len in subset_lens:
        # Ensure the subset length is not larger than the dataset
        if subset_len > len(dataset):
            break
        
        # Sample additional data to extend the subset
        additional_len = subset_len - len(current_subset)
        if additional_len > 0:
            remaining_indices = list(set(range(len(dataset))) - set(current_subset.indices))
            random.seed(seed)
            new_indices = random.sample(remaining_indices, additional_len)                           
        else:
            new_indices = []
        
        # Create a subset with the updated indices
        current_subset = Subset(dataset, current_subset.indices + new_indices)
        
        # Store the subset for this step
        exp_list.append((current_subset, tr_steps))

    return exp_list, current_subset