import os
import argparse
import re
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset


from src.get_datasets import get_benchmark, get_iid_dataset
from src.backbones import get_encoder
from src.utils import SupervisedDataset

EVAL_MB_SIZE = 256

def parse_config(file_path):
    config = {
        "seed": None,
        "encoder": None,
        "model": None,
        "dataset": None,
        "num_exps": None,
        "val_ratio": None,
        "eval_mb_size": None,
        "iid": None,
        "vit_avg_pooling": False,
        "dataset_seed": None
    }
    
    patterns = {
        "seed": re.compile(r"Seed: (\d+)"),
        "encoder": re.compile(r"Encoder: (\w+)"),
        "model": re.compile(r"Model: (\w+)"),
        "dataset": re.compile(r"Dataset: (\w+)"),
        "num_exps": re.compile(r"Number of Experiences: (\d+)"),
        "val_ratio": re.compile(r"Probing Validation Ratio: ([\d.]+)"),
        "eval_mb_size": re.compile(r"Evaluation MB Size: (\d+)"),
        "iid": re.compile(r"IID pretraining: (\w+)"),
        "vit_avg_pooling": re.compile(r"ViT Average Pooling: (\w+)"),
        "dataset_seed": re.compile(r"Dataset Seed: (\d+)"),
    }
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        for key, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                value = match.group(1)
                if key in ["seed", "num_exps", "eval_mb_size", "dataset_seed"]:
                    config[key] = int(value)
                elif key == "val_ratio":
                    config[key] = float(value)
                elif key in ["iid", "vit_avg_pooling"]:
                    config[key] = value.lower() == 'true'
                else:
                    config[key] = str(value)
    
    return config


def save_activations(args, device):
    # Read config.txt inside saved model to get model infos
    config = parse_config(args.model_config_pth)

    print(f'config: {config}')

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.default_rng(config['seed'])

    # Get dataset
    benchmark, image_size = get_benchmark(dataset_name=config['dataset'],
                              dataset_root=args.dataset_root, 
                              num_exps=config['num_exps'],
                              seed=config['dataset_seed'],
                              val_ratio=config['val_ratio'])
    
    # Make train dataset iid
    iid_tr_dataset = SupervisedDataset(get_iid_dataset(benchmark), config['dataset'])

    # Create joint test and val datasets
    joint_test_dataset = SupervisedDataset(ConcatDataset(benchmark.test_stream), config['dataset'])
    if config['val_ratio'] > 0:
        joint_val_dataset = SupervisedDataset(ConcatDataset(benchmark.valid_stream), config['dataset'])



    # Get encoder with saved weights
    encoder, dim_encoder_features = get_encoder(config['encoder'],
                                                image_size=image_size,
                                                ssl_model_name="-", # not needed because always has to return the feature extractor
                                                vit_avg_pooling=config['vit_avg_pooling'])
    encoder = encoder.to(device)

    saved_weights = torch.load(args.model_pth, map_location=device)
    if config['model'] in ['simsiam', 'barlow_twins', 'emp']:
        encoder_saved_weights = {k[len('encoder.'):]: v for k, v in saved_weights['model_state_dict'].items() if k.startswith('encoder.')}
    elif config['model'] == 'byol':
        encoder_saved_weights = {k[len('online_encoder.'):]: v for k, v in saved_weights.items() if k.startswith('online_encoder.')}
    else:
        raise ValueError(f"Model {config['model']} not supported for activation saving.")
    
    # compare encoder and encoder_saved_weights keys
    encoder_keys = set(encoder.state_dict().keys())
    saved_keys = set(encoder_saved_weights.keys())
    if encoder_keys != saved_keys:
        missing_keys = encoder_keys - saved_keys
        extra_keys = saved_keys - encoder_keys

        if missing_keys:
            print(f"MISSING KEYS: {missing_keys}")
            print('\n')
            print(f"ENCODER KEYS: {encoder_keys}")
            print('\n')
            print(f"SAVED WEIGHTS KEYS: {saved_keys}")
            print('\n')
            print(f"FULL SAVED WEIGHTS: {saved_weights.keys()}")
        if extra_keys:
            print(f"EXTRA KEYS: {extra_keys}")
        raise ValueError("Mismatch between encoder keys and saved weights keys.")
    encoder.load_state_dict(encoder_saved_weights)

    # labels and activations save paths
    labels_save_pth = os.path.join(args.dest_pth, "labels")
    activations_save_pth = os.path.join(args.dest_pth, "activations")
    if not os.path.exists(labels_save_pth):
        os.makedirs(labels_save_pth)
    if not os.path.exists(activations_save_pth):
        os.makedirs(activations_save_pth)

    with torch.no_grad():
        encoder.eval()

        train_loader = DataLoader(dataset=iid_tr_dataset, batch_size=EVAL_MB_SIZE, shuffle=False, num_workers=8)
        test_loader = DataLoader(dataset=joint_test_dataset, batch_size=EVAL_MB_SIZE, shuffle=False, num_workers=8)
        if config['val_ratio'] > 0:
            val_loader = DataLoader(dataset=joint_val_dataset, batch_size=EVAL_MB_SIZE, shuffle=False, num_workers=8)

        # Get encoder activations for train dataloader
        train_activations_list = []
        train_labels_list = []
        for inputs, labels in tqdm(train_loader, desc="Train Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            train_activations_list.append(activations.detach().cpu())
            train_labels_list.append(labels.detach().cpu())
        train_activations = torch.cat(train_activations_list, dim=0).numpy()
        train_labels = torch.cat(train_labels_list, dim=0).numpy()
        # Save labels and activations 
        train_labels_path = os.path.join(labels_save_pth, "train_labels.npy")
        train_activations_path = os.path.join(activations_save_pth, "train_activations.npy")
        np.save(train_labels_path, train_labels)
        np.save(train_activations_path, train_activations)
      
        # Get encoder activations for test dataloader
        test_activations_list = []
        test_labels_list = []
        for inputs, labels in tqdm(test_loader, desc=f"Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            test_activations_list.append(activations.detach().cpu())
            test_labels_list.append(labels.detach().cpu())
        test_activations = torch.cat(test_activations_list, dim=0).numpy()
        test_labels = torch.cat(test_labels_list, dim=0).numpy()
        # Save labels and activations 
        test_labels_path = os.path.join(labels_save_pth, f"test_labels.npy")
        test_activations_path = os.path.join(activations_save_pth, f"test_activations.npy")
        np.save(test_labels_path, test_labels)
        np.save(test_activations_path, test_activations)
            
        if config['val_ratio'] > 0:
            # Get encoder activations for val dataloader
            val_activations_list = []
            val_labels_list = []
            for inputs, labels in tqdm(val_loader, desc=f"Val Set"):
                inputs, labels = inputs.to(device), labels.to(device)
                activations = encoder(inputs)
                val_activations_list.append(activations.detach().cpu())
                val_labels_list.append(labels.detach().cpu())
            val_activations = torch.cat(val_activations_list, dim=0).numpy()
            val_labels = torch.cat(val_labels_list, dim=0).numpy()

            # Save labels and activations 
            val_labels_path = os.path.join(labels_save_pth, f"val_labels.npy")
            val_activations_path = os.path.join(activations_save_pth, f"val_activations.npy")
            np.save(val_labels_path, val_labels)
            np.save(val_activations_path, val_activations)

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saving activations from a pretrained model')
    parser.add_argument('--model-pth', type=str)
    parser.add_argument('--model-config-pth', type=str, default='./model_configs/config.txt')
    parser.add_argument('--dataset-root', type=str, default='/data/cossu/imagenet/imagenet')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dest-pth', type=str, default='./activations')
    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        try:
            device_idx = int(args.device)
            device = torch.device(f'cuda:{device_idx}')
        except ValueError:
            print(f"Invalid device argument: {args.device}. Using the default device.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_activations(args, device)