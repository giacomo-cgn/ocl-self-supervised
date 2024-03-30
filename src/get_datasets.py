from .transforms import get_dataset_transforms
import random

import torch
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from .benchmark import Benchmark
def get_benchmark(dataset_name, dataset_root, num_exps=20, seed=42, val_ratio=0.1):

    return_task_id = False
    shuffle = True

    if dataset_name == 'cifar100':
        benchmark = SplitCIFAR100(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'cifar10':
        benchmark = SplitCIFAR10(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'imagenet':
        benchmark = SplitImageNet(
                dataset_root=dataset_root,
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'imagenet100':
        # Select 100 random classes from Imagenet
        random.seed(seed) # Seed for getting always same classes
        classes = random.sample(range(0, 1000), 100)
        benchmark = SplitImageNet(
            dataset_root=dataset_root,
            n_experiences=num_exps,
            fixed_class_order = classes,
            return_task_id=return_task_id,
            shuffle=shuffle,
            train_transform=get_dataset_transforms(dataset_name),
            eval_transform=get_dataset_transforms(dataset_name),
            # class_ids_from_zero_from_first_exp=True ## not allowed for Avalanche < 0.4.0
        )

        # Same code as in Avalanche 0.4.0 for enabling "class_ids_from_zero_from_first_exp=True"
        n_original_classes = max(benchmark.classes_order_original_ids) + 1
        benchmark.classes_order = list(range(0, benchmark.n_classes))
        benchmark.class_mapping = [-1] * n_original_classes
        for class_id in range(n_original_classes):
            # This check is needed because, when a fixed class order is
            # used, the user may have defined an amount of classes less than
            # the overall amount of classes in the dataset.
            if class_id in benchmark.classes_order_original_ids:
                benchmark.class_mapping[class_id] = (
                    benchmark.classes_order_original_ids.index(class_id)
                )


    # Create Benchmark object with tr, test (and validation) streams
    tr_stream = []
    valid_stream = []    
    for experience in benchmark.train_stream:
        if val_ratio > 0:
            tr_exp_dataset, val_exp_dataset = class_balanced_split(val_ratio, experience)
            tr_stream.append(tr_exp_dataset)
            valid_stream.append(val_exp_dataset)
        else:
            tr_stream.append(experience.dataset)

    test_stream = []
    for experience in benchmark.test_stream:
        test_stream.append(experience.dataset)

    if val_ratio > 0:
        benchmark = Benchmark(train_stream=tr_stream, test_stream=test_stream, valid_stream=valid_stream)
    else:
        benchmark = Benchmark(train_stream=tr_stream, test_stream=test_stream)

    return benchmark

def get_iid_dataset(benchmark: Benchmark):
     iid_dataset_tr = ConcatDataset([tr_exp_dataset for tr_exp_dataset in benchmark.train_stream])
     return iid_dataset_tr


def class_balanced_split(validation_size, experience):
    """Class-balanced train/validation splits.

    This splitting strategy splits `experience` into two experiences
    (train and validation) of size `validation_size` using a class-balanced
    split. Sample of each class are chosen randomly.

    You can use this split strategy to split a benchmark with::

        validation_size = 0.2
        foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
        bm = benchmark_with_validation_stream(bm, custom_split_strategy=foo)

    :param validation_size: The percentage of samples to allocate to the
        validation experience as a float between 0 and 1.
    :param experience: The experience to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    exp_dataset = experience.dataset

    exp_indices = list(range(len(exp_dataset)))
    exp_classes = experience.classes_in_this_experience

    # shuffle exp_indices
    exp_indices = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
    # shuffle the targets as well
    exp_targets = torch.as_tensor(experience.dataset.targets)[exp_indices]

    train_exp_indices = []
    valid_exp_indices = []
    for cid in exp_classes:  # split indices for each class separately.
        c_indices = exp_indices[exp_targets == cid]
        valid_n_instances = int(validation_size * len(c_indices))
        valid_exp_indices.extend(c_indices[:valid_n_instances])
        train_exp_indices.extend(c_indices[valid_n_instances:])

    result_train_dataset = exp_dataset.subset(train_exp_indices)
    result_valid_dataset = exp_dataset.subset(valid_exp_indices)
    return result_train_dataset, result_valid_dataset

 