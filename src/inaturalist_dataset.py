import torchvision
from torch.utils.data import Subset
import pandas as pd

TRAIN_VERSION = '2021_train_mini'
VALID_VERSION = '2021_valid'

ORDERED_CATEGORIES = ["kingdom", "phylum", "class", "order", "family", "genus", "full"]

def INaturalist(root, download=False, train_transform=None, eval_transform=None,
                train_labels_pth=None, eval_labels_pth=None
                ):
        
    dataset_tr = torchvision.datasets.INaturalist(root=root,
                                                version=TRAIN_VERSION,
                                                target_type='full',
                                                download=download,
                                                transform=train_transform)
    train_labels_csv = pd.read_csv(train_labels_pth)
    # Each row correspond to a sample (same order as dataset). Columns are:
    #   - 'include' (sample to be included in dataset),
    #   - 'target_label' (prediction label of the sample)
    #   - 'task_label' (task label for the sample)

    # Include only samples with 'include' == 1, split the dataset according to 'task_label'
    included_samples_idxs = train_labels_csv['include'] == 1
    task_labels = train_labels_csv[included_samples_idxs]['task_label'].unique()
    tr_datasets = {}
    for task in task_labels:
        # Indices of samples for this task
        task_indices = train_labels_csv[included_samples_idxs & (train_labels_csv['task_label'] == task)].index.tolist()
        # Labels of samples for this task
        task_labels = train_labels_csv.loc[task_indices, 'target_label'].tolist()
        # Create subset dataset for this task
        tr_datasets[task] = Subset(dataset_tr, task_indices)
        # Substitute labels in the subset to be the 'target_label' from the csv
        if hasattr(tr_datasets[task], 'targets'):
            tr_datasets[task].targets = task_labels
        elif hasattr(tr_datasets[task], 'labels'):
            tr_datasets[task].labels = task_labels
        else:  
            raise AttributeError("Subset train dataset has no attribute 'targets' or 'labels', here are all attributes: ", dir(tr_datasets[task]))

    # List of task datasets, ordered by task label
    tr_stream = [tr_datasets[task] for task in sorted(task_labels)]
    #  Print how many samples in each task
    for exp_idx, tr_exp in enumerate(tr_stream):
        print(f'Experience {exp_idx}: {len(tr_exp)} samples')

    dataset_test = torchvision.datasets.INaturalist(root=root,
                                                version=VALID_VERSION,
                                                target_type='full',
                                                download=download,
                                                transform=eval_transform)
    test_labels_csv = pd.read_csv(eval_labels_pth)

    # Include only samples with 'include' == 1, split the dataset according to 'task_label'
    included_samples_idxs = test_labels_csv['include'] == 1
    task_labels = test_labels_csv[included_samples_idxs]['task_label'].unique()
    test_datasets = {}
    for task in task_labels:
        # Indices of samples for this task
        task_indices = test_labels_csv[included_samples_idxs & (test_labels_csv['task_label'] == task)].index.tolist()
        # Labels of samples for this task
        task_labels = test_labels_csv.loc[task_indices, 'target_label'].tolist()
        # Create subset dataset for this task
        test_datasets[task] = Subset(dataset_test, task_indices)
        # Substitute labels in the subset to be the 'target_label' from the csv
        if hasattr(test_datasets[task], 'targets'):
            test_datasets[task].targets = task_labels
        elif hasattr(test_datasets[task], 'labels'):
            test_datasets[task].labels = task_labels
        else:  
            raise AttributeError("Subset test dataset has no attribute 'targets' or 'labels', here are all attributes: ", dir(test_datasets[task]))
        
    # List of task datasets, ordered by task label
    test_stream = [test_datasets[task] for task in sorted(task_labels)]

    # Return object with two attributes test_stream and tr_stream
    return type('INaturalistDataset', (), {'test_stream': test_stream, 'train_stream': tr_stream})
