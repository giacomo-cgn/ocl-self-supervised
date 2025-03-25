import random
import torch
import numpy as np

class LossAwareBuffer:
    """
    Custom buffer that removes elements based on their loss. Can store batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size, alpha_ema=1.0, alpha_ema_loss=0.0, insertion_policy='random', device='cpu'):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.buffer_loss = torch.empty(0,1).to(device) # Buffer for corresponding sample losses
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.alpha_ema_loss = alpha_ema_loss # 1.0 = do not update stored losses, 0.0 = substitute with new losses
        self.insertion_policy = insertion_policy # 'random', 'loss' or 'fifo', which policy to use to insert new samples.
        # Samples to remove are always chosen based on their loss.
        self.device = device
        self.lifetimes = torch.empty(0, dtype=torch.int) # Buffer for the life of each sample
        self.extractions = torch.empty(0, dtype=torch.int) # Buffer for the number of times each sample has been extracted
        self.finished_lifetimes = []
        self.finished_extractions = []

        self.seen_samples = 0 # Samples seen so far

    # Add a batch of samples, features and losses to the buffer
    def add(self, batch_x, batch_features, batch_loss):
        assert batch_x.size(0) == batch_features.size(0) == batch_loss.size(0)

        # Add +1 to all lifetimes
        self.lifetimes += 1

        batch_x, batch_features, batch_loss = batch_x.to(self.device), batch_features.to(self.device), batch_loss.to(self.device)

        # Initialize empty buffers
        if self.buffer.size(0) == 0:
            # Extend buffer to have same dim of batch_x
            buffer_shape = list(batch_x.size())
            buffer_shape[0] = 0
            self.buffer = torch.empty(buffer_shape, dtype=batch_x.dtype).to(self.device)

            # Extend buffer_features to have same dim of batch_features
            buffer_shape = list(batch_features.size())
            buffer_shape[0] = 0
            self.buffer_features = torch.empty(buffer_shape, dtype=batch_features.dtype).to(self.device)

            # Extend buffer_loss to have same dim of batch_loss
            buffer_shape = list(batch_loss.size())
            buffer_shape[0] = 0
            self.buffer_loss = torch.empty(buffer_shape, dtype=batch_loss.dtype).to(self.device)

        batch_size = batch_x.size(0)

        if self.seen_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.seen_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                self.buffer = torch.cat((self.buffer, batch_x), dim=0)
                self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)
                self.buffer_loss = torch.cat((self.buffer_loss, batch_loss), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                self.seen_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.seen_samples
                self.buffer = torch.cat((self.buffer, batch_x[:remaining_space]), dim=0)
                self.buffer_features = torch.cat((self.buffer_features, batch_features[:remaining_space]), dim=0)
                self.buffer_loss = torch.cat((self.buffer_loss, batch_loss[:remaining_space]), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                self.seen_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/seen_samples
            for i in range(batch_size):
                if self.insertion_policy == 'random':
                    # Like reservoir sampling
                    replace_index = random.randint(0, self.seen_samples + i)

                    if replace_index < self.buffer_size:
                        # Replace sample in buffer with the minimum loss
                        replace_index = self.buffer_loss.argmin().item()

                        self.finished_lifetimes.append(self.lifetimes[replace_index].item())
                        self.lifetimes[replace_index] = 0
                        self.finished_extractions.append(self.extractions[replace_index].item())
                        self.extractions[replace_index] = 0

                        self.buffer[replace_index] = batch_x[i]
                        self.buffer_features[replace_index] = batch_features[i]
                        self.buffer_loss[replace_index] = batch_loss[i]

                elif self.insertion_policy == 'loss':
                    # Concat new batch to buffer
                    self.buffer = torch.cat((self.buffer, batch_x[i].unsqueeze(0)), dim=0)
                    self.buffer_features = torch.cat((self.buffer_features, batch_features[i].unsqueeze(0)), dim=0)
                    self.buffer_loss = torch.cat((self.buffer_loss, batch_loss[i].unsqueeze(0)), dim=0)

                    self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                    self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                    # Find the batch_size samples with minimum loss and remove them
                    indices_to_remove = self.buffer_loss.argsort()[:batch_size].cpu()
                    self.finished_lifetimes += self.lifetimes[indices_to_remove].tolist()
                    self.finished_extractions += self.extractions[indices_to_remove].tolist()

                    indices_to_keep = self.buffer_loss.argsort()[batch_size:].cpu()
                    self.buffer = self.buffer[indices_to_keep]
                    self.buffer_features = self.buffer_features[indices_to_keep]
                    self.buffer_loss = self.buffer_loss[indices_to_keep]
                    self.lifetimes = self.lifetimes[indices_to_keep]
                    self.extractions = self.extractions[indices_to_keep]

                    
                elif self.insertion_policy == 'fifo':
                    # remove batch_size samples from the buffer with the minimum loss
                    indices_to_remove = self.buffer_loss.argsort()[:batch_size].cpu()
                    self.finished_lifetimes += self.lifetimes[indices_to_remove].tolist()
                    self.finished_extractions += self.extractions[indices_to_remove].tolist()

                    indices_to_keep = self.buffer_loss.argsort()[batch_size:].cpu()
                    self.buffer = self.buffer[indices_to_keep]
                    self.buffer_features = self.buffer_features[indices_to_keep]
                    self.buffer_loss = self.buffer_loss[indices_to_keep]
                    self.lifetimes = self.lifetimes[indices_to_keep]
                    self.extractions = self.extractions[indices_to_keep]                    

                    # Concat new batch to buffer
                    self.buffer = torch.cat((self.buffer, batch_x[i].unsqueeze(0)), dim=0)
                    self.buffer_features = torch.cat((self.buffer_features, batch_features[i].unsqueeze(0)), dim=0)
                    self.buffer_loss = torch.cat((self.buffer_loss, batch_loss[i].unsqueeze(0)), dim=0)

                    self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                    self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)

                else:
                    raise Exception(f'Insertion policy {self.insertion_policy} is not supported for LossAwareBuffer')

            self.seen_samples += batch_size

    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)

        self.extractions[indices] += 1

        # Get sample batch from indices
        batch_x = self.buffer[indices]
        batch_features = self.buffer_features[indices]

        return batch_x, batch_features, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, batch_loss, indices):
        assert batch_features.size(0) == len(indices) == batch_loss.size(0)

        batch_features = batch_features.to(self.device)
        batch_loss = batch_loss.to(self.device)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
                # EMA update of loss
                self.buffer_loss[idx] = self.alpha_ema_loss * self.buffer_loss[idx] + (1 - self.alpha_ema_loss) * batch_loss[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]
                self.buffer_loss[idx] = batch_loss[i]

    def end(self):
        for lifetime in self.lifetimes:
            self.finished_lifetimes.append(lifetime)
        for extraction in self.extractions:
            self.finished_extractions.append(extraction)

        avg_lifetime = np.mean(self.finished_lifetimes)
        avg_extraction = np.mean(self.finished_extractions)
        metrics_buffer = f"Average lifetime: {avg_lifetime:.2f}\nAverage extraction: {avg_extraction:.2f}\n"

        csv_buffer = "lifetime,extraction\n"
        for i in range(len(self.finished_lifetimes)):
            csv_buffer += str(self.finished_lifetimes[i]) + "," + str(self.finished_extractions[i]) + "\n"

        return csv_buffer, metrics_buffer