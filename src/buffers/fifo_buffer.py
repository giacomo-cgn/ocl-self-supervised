import random
import torch
import numpy as np


class FIFOBuffer:
    """
    Custom FIFO buffer class for batches of samples without labels, but with encoder features.
    FIFO means that oldest samples are replaced with new ones on each add call, sampling remains random.
    """
    def __init__(self, buffer_size, alpha_ema=1.0, alpha_ema_loss=0.0):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = [] # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # Buffer for corresponding sample features
        self.buffer_loss = [] # Buffer for corresponding sample losses
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.alpha_ema_loss = alpha_ema_loss # 1.0 = do not update stored losses, 0.0 = substitute with new losses

        self.buffer_e_stats = {} # dict containing encoder (e) stats, each is a list
        self.buffer_z_stats = {} # dict containing projector (z) stats, each is a list

        self.lifetimes = [] # Buffer for the life of each sample
        self.extractions = [] # Buffer for the number of times each sample has been extracted
        self.finished_lifetimes = []
        self.finished_extractions = []


    # Add a batch of samples and features to the buffer
    def add(self, batch_x, batch_features, batch_loss, e_stats=None, z_stats=None):
        assert batch_x.size(0) == batch_features.size(0)
        # Adds batch with a FIFO strategy

        self.lifetimes = [lifetime + 1 for lifetime in self.lifetimes]

        self.buffer.extend(batch_x)
        self.buffer_features.extend(batch_features)
        self.buffer_loss.extend(batch_loss)
        self.lifetimes.extend([0]*batch_x.size(0))
        self.extractions.extend([0]*batch_x.size(0))


        # Add e_stats and z_stats to the buffer
        if e_stats is not None:
            for key in e_stats.keys():
                if key not in self.buffer_e_stats:
                    self.buffer_e_stats[key] = []
                self.buffer_e_stats[key].extend(e_stats[key])
        if z_stats is not None:
            for key in z_stats.keys():
                if key not in self.buffer_z_stats:
                    self.buffer_z_stats[key] = []
                self.buffer_z_stats[key].extend(z_stats[key])

        if len(self.buffer) > self.buffer_size:
            # Remove oldest samples
            self.buffer = self.buffer[-self.buffer_size:]
            self.buffer_features = self.buffer_features[-self.buffer_size:]
            self.buffer_loss = self.buffer_loss[-self.buffer_size:]
            self.finished_lifetimes += self.lifetimes[:-self.buffer_size]
            self.finished_extractions += self.extractions[:-self.buffer_size]
            self.lifetimes = self.lifetimes[-self.buffer_size:]
            self.extractions = self.extractions[-self.buffer_size:]

            # Remove oldest e_stats and z_stats
            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key] = self.buffer_e_stats[key][-self.buffer_size:]
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key] = self.buffer_z_stats[key][-self.buffer_size:]

    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]
        features = [self.buffer_features[i] for i in indices]

        for i in indices:
            self.extractions[i] += 1

        # Reconstruct batch from samples
        batch_x = torch.stack([sample for sample in samples])
        batch_features = torch.stack([feature for feature in features])

        return batch_x, batch_features, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, batch_loss, indices, e_stats=None, z_stats=None):
        assert batch_features.size(0) == len(indices)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx] + (1 - self.alpha_ema) * batch_features[i]
                # EMA update of loss
                self.buffer_loss[idx] = self.alpha_ema_loss * self.buffer_loss[idx] + (1 - self.alpha_ema_loss) * batch_loss[i]
            else:
                # No features stored yet, store newly passed features and loss
                self.buffer_features[idx] = batch_features[i]
                self.buffer_loss[idx] = batch_loss[i]

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
        return torch.stack(self.buffer)
