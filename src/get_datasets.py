from .transforms import get_dataset_transforms
import random

from torch.utils.data import ConcatDataset

from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy


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


    # Extract validation set too if needed
    if val_ratio > 0:
        class_balanced_split = lambda exp: class_balanced_split_strategy(val_ratio, exp)
        benchmark = benchmark_with_validation_stream(benchmark, custom_split_strategy=class_balanced_split, shuffle=True)

    return benchmark

def get_iid_dataset(benchmark):
     iid_dataset_tr = ConcatDataset([tr_experience.dataset for tr_experience in benchmark.train_stream])
     return iid_dataset_tr
        