from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from transforms import get_dataset_transforms
import random

def get_dataset(dataset_name, dataset_root, num_exps=20):

    first_exp_with_half_classes = False
    return_task_id = False
    shuffle = True

    if dataset_name == 'cifar100':
        benchmark = SplitCIFAR100(
                n_experiences=num_exps,
                seed=42, # Fixed seed for reproducibility
                first_exp_with_half_classes=first_exp_with_half_classes,
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'cifar10':
        benchmark = SplitCIFAR10(
                n_experiences=num_exps,
                seed=42, # Fixed seed for reproducibility
                first_exp_with_half_classes=first_exp_with_half_classes,
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'imagenet':
        benchmark = SplitImageNet(
                dataset_root=dataset_root,
                n_experiences=num_exps,
                seed=42, # Fixed seed for reproducibility
                first_exp_with_half_classes=first_exp_with_half_classes,
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(dataset_name),
                eval_transform=get_dataset_transforms(dataset_name),
            )
        
    elif dataset_name == 'imagenet100':
        # Select 100 random classes from Imagenet
        random.seed(42) # Seed for getting always same classes
        classes = random.sample(range(0, 1000), 100)
        benchmark = SplitImageNet(
            dataset_root=dataset_root,
            n_experiences=num_exps,
            fixed_class_order = classes,
            first_exp_with_half_classes=first_exp_with_half_classes,
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

    return benchmark
        