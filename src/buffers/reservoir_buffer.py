import random
import torch
import numpy as np


class ReservoirBuffer:
    """
    Custom reservoir buffer class for batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size, alpha_ema=1.0, device='cpu'):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.device = device

        self.lifetimes = torch.empty(0, dtype=torch.int) # Buffer for the life of each sample
        self.extractions = torch.empty(0, dtype=torch.int) # Buffer for the number of times each sample has been extracted
        self.finished_lifetimes = []
        self.finished_extractions = []

        self.buffer_e_stats = {} # dict containing encoder (e) stats, each is a list
        self.buffer_z_stats = {} # dict containing projector (z) stats, each is a list

        self.seen_samples = 0 # Samples seen so far

    # Add a batch of samples and features to the buffer
    def add(self, batch_x, batch_features, batch_loss=None, e_stats=None, z_stats=None):
        assert batch_x.size(0) == batch_features.size(0)

        # Add +1 to all lifetimes
        self.lifetimes += 1

        batch_x, batch_features = batch_x.to(self.device), batch_features.to(self.device)

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

            # Initialize empty lists for e_stats and z_stats
            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key] = []
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key] = []

        batch_size = batch_x.size(0)

        if self.seen_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.seen_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                self.buffer = torch.cat((self.buffer, batch_x), dim=0)
                self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                if e_stats is not None:
                    for key in e_stats.keys():
                        self.buffer_e_stats[key] += e_stats[key]
                if z_stats is not None:
                    for key in z_stats.keys():
                        self.buffer_z_stats[key] += z_stats[key]
                self.seen_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.seen_samples
                self.buffer = torch.cat((self.buffer, batch_x[:remaining_space]), dim=0)
                self.buffer_features = torch.cat((self.buffer_features, batch_features[:remaining_space]), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                if e_stats is not None:
                    for key in e_stats.keys():
                        self.buffer_e_stats[key] += e_stats[key][:remaining_space]
                if z_stats is not None:
                    for key in z_stats.keys():
                        self.buffer_z_stats[key] += z_stats[key][:remaining_space]
                self.seen_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/seen_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.seen_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = batch_x[i]
                    self.buffer_features[replace_index] = batch_features[i]
                    if e_stats is not None:
                        for key in e_stats.keys():
                            self.buffer_e_stats[key][replace_index] = e_stats[key][i]
                    if z_stats is not None:
                        for key in z_stats.keys():
                            self.buffer_z_stats[key][replace_index] = z_stats[key][i]

                    self.finished_lifetimes.append(self.lifetimes[replace_index].item())
                    self.lifetimes[replace_index] = 0
                    self.finished_extractions.append(self.extractions[replace_index].item())
                    self.extractions[replace_index] = 0
            
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
    def update_features(self, batch_features, indices, batch_loss=None, e_stats=None, z_stats=None):
        assert batch_features.size(0) == len(indices)

        batch_features = batch_features.to(self.device)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]

            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key][idx] = self.alpha_ema * self.buffer_e_stats[key][idx] + (1 - self.alpha_ema) * e_stats[key][i]
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key][idx] = self.alpha_ema * self.buffer_z_stats[key][idx] + (1 - self.alpha_ema) * z_stats[key][i]
                    
    def end(self):
        results_lifetimes = []
        results_extractions = []

        results_lifetimes += [lifetime.item() if hasattr(lifetime, 'item') else lifetime for lifetime in self.lifetimes]
        results_extractions += [extraction.item() if hasattr(extraction, 'item') else extraction for extraction in self.extractions]

        results_lifetimes += [lifetime.item() if hasattr(lifetime, 'item') else lifetime for lifetime in self.finished_lifetimes]
        results_extractions += [extraction.item() if hasattr(extraction, 'item') else extraction for extraction in self.finished_extractions]

        avg_lifetime = np.mean(results_lifetimes)
        avg_extraction = np.mean(results_extractions)
        metrics_buffer = f"Average lifetime: {avg_lifetime:.2f}\nAverage extraction: {avg_extraction:.2f}\n"

        csv_buffer = "lifetime,extraction\n"
        for i in range(len(results_lifetimes)):
            csv_buffer += str(results_lifetimes[i]) + "," + str(results_extractions[i]) + "\n"

        return csv_buffer, metrics_buffer
    
    def get_curr_len(self):
        return len(self.buffer)
    
    def get_buffer_data(self):
        return self.buffer