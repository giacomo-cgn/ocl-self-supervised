import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class LinearProbingSklearn:
    def __init__(self,
                 encoder: nn,
                 device: str = 'cpu',
                 mb_size: int = 1024,
                 save_file: str = None,
                 exp_idx: int = None,
                 tr_samples_ratio: float = 1.0,
                 val_ratio: float = 0.1
                 ):
        """
        Initialize the Linear Probing classifier.

        Args:
        """
        self.encoder = encoder.to(device)
        self.device = device
        self.mb_size = mb_size
        self.save_file = save_file
        self.exp_idx = exp_idx # Task index on which probing is executed, if None, we are in joint probing
        self.tr_samples_ratio = tr_samples_ratio
        self.val_ratio = val_ratio
        
        # Patience on before early stopping
        self.patience = 2

        self.criterion = nn.CrossEntropyLoss()

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                # Write header for probing log file
                if not os.path.exists(self.save_file) or os.path.getsize(self.save_file) == 0:
                    if self.exp_idx is not None:
                        f.write('probing_exp_idx,val_acc,test_acc\n')
                    else:
                        f.write(f'val_acc,test_acc\n')

    def probe(self,
              tr_dataset: Dataset,
              test_dataset: Dataset,
              ):

        # Prepare dataloaders

        # Split train into train and validation
        val_size = int(len(tr_dataset) * self.val_ratio)
        tr_size = len(tr_dataset) - val_size
        tr_dataset, val_dataset = random_split(tr_dataset, [tr_size, val_size],
                                               generator=torch.Generator().manual_seed(42)) # Generator to ensure same splits

        # Select only a random ratio of the train data for probing
        used_ratio_samples = int(len(tr_dataset) * self.tr_samples_ratio)
        tr_dataset, _ = random_split(tr_dataset, [used_ratio_samples, len(tr_dataset) - used_ratio_samples],
                                     generator=torch.Generator().manual_seed(42)) # Generator to ensure same splits
    
        train_loader = DataLoader(dataset=tr_dataset, batch_size=self.mb_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.mb_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.mb_size, shuffle=False)

        # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
        self.encoder.eval()

        # Get encoder activations for tr dataloader
        tr_activations_list = []
        tr_labels_list = []
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            activations = self.encoder(inputs)
            tr_activations_list.append(activations.detach())
            tr_labels_list.append(labels)
        tr_activations = torch.cat(tr_activations_list, dim=0).cpu().numpy()
        tr_labels = torch.cat(tr_labels_list, dim=0).cpu().numpy()

        # Get encoder activations for val dataloader
        val_activations_list = []
        val_labels_list = []
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            activations = self.encoder(inputs)
            val_activations_list.append(activations.detach())
            val_labels_list.append(labels)
        val_activations = torch.cat(val_activations_list, dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels_list, dim=0).cpu().numpy()

        # Get encoder activations for test dataloader
        test_activations_list = []
        test_labels_list = []
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            activations = self.encoder(inputs)
            test_activations_list.append(activations.detach())
            test_labels_list.append(labels)
        test_activations = torch.cat(test_activations_list, dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list, dim=0).cpu().numpy()

        scaler = StandardScaler()

        # Use Logistic Regression to learn weights from activations
        tr_activations = scaler.fit_transform(tr_activations)
        #log_reg = LogisticRegression(max_iter=200).fit(tr_activations, tr_labels)
        log_reg = RidgeClassifier().fit(tr_activations, tr_labels)
        
        # Predict validation
        val_activations = scaler.transform(val_activations)
        val_preds = log_reg.predict(val_activations)
        # Calculate validation accuracy and loss
        val_acc = accuracy_score(val_labels, val_preds)

        # Predict test
        test_activations = scaler.transform(test_activations)
        test_preds = log_reg.predict(test_activations)
        # Calculate test accuracy
        test_acc = accuracy_score(test_labels, test_preds)

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if self.exp_idx is not None:
                    f.write(f'{self.exp_idx},{val_acc},{test_acc}\n')
                else:
                    f.write(f'{val_acc},{test_acc}\n')
