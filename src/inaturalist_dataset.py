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
    included_samples = train_labels_csv[train_labels_csv['include'] == 1]
    task_labels = included_samples['task_label'].unique()
    tr_datasets = {}
    for task in task_labels:
        task_indices = included_samples[included_samples['task_label'] == task].index.tolist()
        tr_datasets[task] = Subset(dataset_tr, task_indices)
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
    included_samples = test_labels_csv[test_labels_csv['include'] == 1]
    task_labels = included_samples['task_label'].unique()
    test_datasets = {}
    for task in task_labels:
        task_indices = included_samples[included_samples['task_label'] == task].index.tolist()
        test_datasets[task] = Subset(dataset_test, task_indices)
    # List of task datasets, ordered by task label
    test_stream = [test_datasets[task] for task in sorted(task_labels)]

    # Return object with two attributes test_stream and tr_stream
    return type('INaturalistDataset', (), {'test_stream': test_stream, 'train_stream': tr_stream})
