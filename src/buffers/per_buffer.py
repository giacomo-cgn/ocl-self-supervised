import random
import torch
import numpy as np


class PERBuffer:
    """
    Prioritized Experience Replay (PER) Buffer. Extraction is softly prioritized based on sample loss.
    It is based on a FIFO buffer for storage.
    It can store batches of samples without labels, but with encoder features.
    """

    def __init__(self, buffer_size, alpha_ema=1.0, alpha_ema_loss=0.0, alpha_per=0.5, epsilon_per=0.01, rank_based_per=False):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = [] # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # Buffer for corresponding sample features
        self.buffer_loss = [] # Buffer for corresponding sample losses
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.alpha_ema_loss = alpha_ema_loss # 1.0 = do not update stored losses, 0.0 = substitute with new losses

        # PER hyperparameters
        self.alpha_per = alpha_per  # Determines how much prioritization is used (0 = uniform, 1 = full prioritization)
        self.epsilon_per = epsilon_per  # Small constant to ensure non-zero probability for all samples
        self.rank_based_per = rank_based_per  # Whether to use rank-based prioritization

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

    # Sample batch_size samples from the buffer with prioritization based on loss
    # Returns samples and indices
    def sample(self, batch_size, use_prioritization=True):
        assert batch_size <= len(self.buffer)

        if not use_prioritization or self.alpha_per == 0:
            # Uniform sampling (original behavior)
            indices = random.sample(range(len(self.buffer)), batch_size)
            weights = torch.ones(batch_size)  # Uniform weights
        else:
            # Prioritized sampling based on loss
            losses = torch.tensor([self.buffer_loss[i].item() if torch.is_tensor(self.buffer_loss[i]) 
                                   else self.buffer_loss[i] for i in range(len(self.buffer))])
            
            if self.rank_based_per:
                # Rank-based prioritization: p_i = 1 / rank(i)
                # Sort by loss (descending) and get ranks
                sorted_indices = torch.argsort(torch.abs(losses), descending=True)
                ranks = torch.zeros(len(self.buffer))
                ranks[sorted_indices] = torch.arange(1, len(self.buffer) + 1).float()

                # Calculate priorities: p_i = 1 / rank(i)^alpha_per
                priorities = (1.0 / ranks) ** self.alpha_per
            else:
                # Proportional prioritization: p_i = (|loss_i| + epsilon_per)^alpha_per
                priorities = (torch.abs(losses) + self.epsilon_per) ** self.alpha_per

            # Calculate sampling probabilities: P(i) = p_i / sum(p_k)
            probabilities = priorities / priorities.sum()
            
            # Sample indices according to probabilities
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, 
                                      p=probabilities.numpy())
            indices = indices.tolist()

        # Extract samples and features
        samples = [self.buffer[i] for i in indices]
        features = [self.buffer_features[i] for i in indices]

        # Update extraction counts
        for i in indices:
            self.extractions[i] += 1

        # Reconstruct batch from samples
        batch_x = torch.stack([sample for sample in samples])
        batch_features = torch.stack([feature for feature in features])

        return batch_x, batch_features, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices, batch_loss, e_stats=None, z_stats=None):
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