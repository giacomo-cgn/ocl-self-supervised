import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from avalanche.benchmarks.scenarios import NCExperience

class LinearProbing:
    def __init__(self,
                 encoder: nn,
                 dim_features: nn,
                 num_classes: int,
                 lr: float = 0.01,
                 weight_decay: float = 2e-4,
                 momentum: float = 0,
                 device: str = 'cpu',
                 mb_size: int = 32):
        """
        Initialize the Linear Probing classifier.

        Args:
        - encoder (nn.Module): Pre-trained encoder network
        - num_classes (int): Number of classes for the probing classifier
        """
        self.encoder = encoder.to(device)
        self.dim_features = dim_features
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.mb_size = mb_size

        self.probe_layer = nn.Linear(self.dim_features, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.probe_layer.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    def probe(self,
              tr_experience: NCExperience,
              test_experience: NCExperience,
              num_epochs: int=10):
        """
        Train the probing classifier on a training dataset and test it on a test dataset.

        Args:
        - train_loader (DataLoader): DataLoader for the training dataset
        - test_loader (DataLoader): DataLoader for the test dataset
        - num_epochs (int): Number of training epochs (default is 10)
        """

        # Prepare datasets
        train_loader = DataLoader(dataset=tr_experience.dataset, batch_size=self.mb_size, shuffle=True)
        test_loader = DataLoader(dataset=test_experience.dataset, batch_size=self.mb_size, shuffle=False)


        for epoch in range(num_epochs):
            self.probe_layer.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.probe_layer(self.encoder(inputs))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100 * correct / total
            train_loss = running_loss / len(train_loader)

            # Test the probing classifier
            self.probe_layer.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.probe_layer(self.encoder(inputs))
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_accuracy = 100 * correct / total

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%, Test Accuracy: {test_accuracy:.4f}%')

        return train_loss, train_accuracy, test_accuracy
