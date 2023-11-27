import os
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


class LinearProbing:
    def __init__(self,
                 encoder: nn,
                 dim_features: int,
                 num_classes: int,
                 lr: float = 2e-3,
                 weight_decay: float = 1e-4,
                 momentum: float = 0,
                 device: str = 'cpu',
                 mb_size: int = 32,
                 save_file: str = None,
                 exp_idx: int = None,
                 tr_samples_ratio: float = 1.0,
                 num_epochs: int = 50,
                 use_val_stop: bool = True,
                 val_ratio: float = 0.1,
                 probe_mb_size: int = 32,
                 ):
        """
        Initialize the Linear Probing classifier.

        Args:

        """
        self.encoder = encoder.to(device)
        self.dim_features = dim_features
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.mb_size = mb_size
        self.save_file = save_file
        self.exp_idx = exp_idx
        self.tr_samples_ratio = tr_samples_ratio
        self.num_epochs = num_epochs
        self.use_val_stop = use_val_stop
        self.val_ratio = val_ratio
        self.probe_mb_size = probe_mb_size
        
        # Patience on before early stopping
        self.patience = 2

        self.probe_layer = nn.Linear(self.dim_features, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.probe_layer.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

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
        # Change mb size to smaller probe_mb_size (32)
        tr_activations = [
        (torch.cat(tr_activations_list[i:i+self.probe_mb_size], dim=0),
         torch.cat(tr_labels_list[i:i+self.probe_mb_size], dim=0))
          for i in range(0, len(tr_activations_list), self.probe_mb_size)
        ]

        # Get encoder activations for val dataloader
        val_activations_list = []
        val_labels_list = []
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            activations = self.encoder(inputs)
            val_activations_list.append(activations.detach())
            val_labels_list.append(labels)
        # Change mb size to smaller probe_mb_size (32)
        val_activations = [
        (torch.cat(val_activations_list[i:i+self.probe_mb_size], dim=0),
         torch.cat(val_labels_list[i:i+self.probe_mb_size], dim=0))
          for i in range(0, len(val_activations_list), self.probe_mb_size)
        ]

        # Get encoder activations for test dataloader
        test_activations_list = []
        test_labels_list = []
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            activations = self.encoder(inputs)
            test_activations_list.append(activations.detach())
            test_labels_list.append(labels)
        # Change mb size to smaller probe_mb_size (32)
        test_activations = [
        (torch.cat(test_activations_list[i:i+self.probe_mb_size], dim=0),
         torch.cat(test_labels_list[i:i+self.probe_mb_size], dim=0))
          for i in range(0, len(test_activations_list), self.probe_mb_size)
        ]

        # For early stopping on validation
        best_val_loss = float('inf')
        patience_counter = 0
        val_acc_list = []
        val_loss_list = []

        for epoch in range(self.num_epochs):
            self.probe_layer.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tr_activations:

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.probe_layer(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                batch_idx += 1

            train_accuracy = 100 * correct / total
            train_loss = running_loss / len(train_loader)

            if self.use_val_stop:
                # Eval the probing clf on validation set at current epoch
                self.probe_layer.eval()
                correct = 0
                total = 0
                val_loss = 0
                with torch.no_grad():
                    for inputs, labels in val_activations:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.probe_layer(inputs)
                        val_loss += self.criterion(outputs, labels).item()

                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                val_acc = 100 * correct / total
                val_loss = val_loss / len(val_loader)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)

                if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        # Stop training
                        break


        # Eval the probing classifier on test set at the end of training
        self.probe_layer.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_activations:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.probe_layer(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100 * correct / total

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if self.exp_idx is not None:
                    f.write(f'{self.exp_idx},{val_acc},{test_acc}\n')
                else:
                    f.write(f'{val_acc},{test_acc}\n')

        # Plot loss accuracy curve over the epochs
        # if self.use_val_stop:
        #     fig = plt.figure(figsize=(16, 8))

        #     plt.subplot(1, 2, 1)
        #     plt.plot(val_acc_list, label='Validation Accuracy')
        #     plt.title('Validation Accuracy')
        #     plt.subplot(1, 2, 2)
        #     plt.plot(val_loss_list, label='Validation Loss', color='orange')
        #     plt.title('Validation Loss')
        #     tr_exp = self.save_file[-5]
        #     pth = os.path.dirname(self.save_file)
        #     plt.savefig(os.path.join(pth, f'val_curve_tr_exp_{tr_exp}_probe_exp_{self.exp_idx}.png'))
        #     plt.clf()

