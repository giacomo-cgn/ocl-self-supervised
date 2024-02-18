import random
import torch


class ReservoirBuffer:
    """
    Custom reservoir buffer class for batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size, alpha_ema=1.0):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = [] # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # Buffer for corresponding sample features
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features

        self.stored_samples = 0 # Samples seen so far

    # Add a batch of samples and features to the buffer
    def add(self, batch_x, batch_features):
        assert batch_x.size(0) == batch_features.size(0)

        batch_size = batch_x.size(0)

        if self.stored_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.stored_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                self.buffer.extend(batch_x)
                self.buffer_features.extend(batch_features)
                self.stored_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.stored_samples
                self.buffer.extend(batch_x[:remaining_space])
                self.buffer_features.extend(batch_features[:remaining_space])
                self.stored_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/stored_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.stored_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = batch_x[i]
                    self.buffer_features[replace_index] = batch_features[i]
            
            self.stored_samples += batch_size

    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]
        features = [self.buffer_features[i] for i in indices]

        # Reconstruct batch from samples
        batch_x = torch.stack([sample for sample in samples])
        batch_features = torch.stack([feature for feature in features])

        return batch_x, batch_features, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices):
        assert batch_features.size(0) == len(indices)

        for i, idx in enumerate(indices):
            if self.buffer_features[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features[idx] = self.alpha_ema * self.buffer_features[idx][1] + (1 - self.alpha_ema) * batch_features[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features[idx] = batch_features[i]
