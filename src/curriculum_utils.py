from torch.utils.data import random_split, Subset
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

from sklearn.model_selection import train_test_split

class SubsetSplitterWrapper:
    def __init__(self, type: str = 'random', seed: int = 42):
        self.type = type
        self.seed = seed

    def subset(self, dataset: Dataset, len1: int, len2: int):
        if self.type == 'random':
            subset_1, subset2 = random_split(dataset, [len1, len2], generator=torch.Generator().manual_seed(self.seed))
        elif self.type == 'class_balanced':
            subset_1, subset2 = class_balanced_split(dataset, len1, len2, self.seed)
        else:
            raise ValueError("Invalid splitter type. Choose 'random' or 'class_balanced'.")
        return subset_1, subset2


def get_gradual_subset_increase_exps(dataset, total_tr_steps, start_subset_ratio: float, end_subset_ratio: float,
                                     step_ratio: float, subset_splitter: SubsetSplitterWrapper, start_subset: Subset = None):
    subset_lens = [len(dataset)*start_subset_ratio]
    subset_lens.extend([int(len(dataset)*i) for i in np.arange(start_subset_ratio+step_ratio, end_subset_ratio+step_ratio, step_ratio)])

    tr_steps = total_tr_steps // len(subset_lens)
    if start_subset is None:
        start_subset, _ = subset_splitter.subset(dataset, int(subset_lens[0]), int(len(dataset) - subset_lens[0]))
        
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
            remaining_dataset = Subset(dataset, remaining_indices)
            new_samples, _ = subset_splitter.subset(remaining_dataset, additional_len, len(remaining_dataset) - additional_len)
            current_subset = Subset(dataset, current_subset.indices + new_samples.indices)
        
        # Store the subset for this step
        exp_list.append((current_subset, tr_steps))

    return exp_list, current_subset


def class_balanced_split(dataset: Dataset, len1: int, len2: int, seed: int = 42):
    if len1 + len2 > len(dataset):
        raise ValueError("The sum of len1 and len2 exceeds the total number of samples in the dataset.")
    
    indices = list(range(len(dataset)))
    class_labels = [dataset[idx][1] for idx in indices]

    subset_indices_1, subset_indices_2 = train_test_split(indices, test_size=len2 / (len1 + len2), stratify=class_labels, random_state=seed)

    return Subset(dataset, subset_indices_1), Subset(dataset, subset_indices_2)