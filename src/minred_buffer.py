import random
import torch



class MinRedBuffer:
    """
    MinRed buffer class for batches of samples without labels.
    """
    def __init__(self, buffer_size, alpha_ema=0.5):
        self.buffer_size = buffer_size
        self.alpha_ema = alpha_ema
        self.buffer = [] # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # Buffer for corresponding sample features
        self.stored_samples = 0

    # Add a batch of samples to the buffer
    def add(self, batch_x):
        batch_size = batch_x.size(0)
        n_excess = self.stored_samples + batch_size - self.buffer_size

        # Remove n_excess samples
        if n_excess > 0:
            # Buffer is full
            for _ in range(n_excess):
                non_none_indices = [idx for idx, value in enumerate(self.buffer_features) if value is not None]

                stacked_features = torch.stack([self.buffer_features[i] for i in non_none_indices], dim=0)
                # Cosine distance = 1 - cosine similarity
                tensor_normalized = torch.nn.functional.normalize(stacked_features, p=2, dim=1)
                d = 1- torch.mm(tensor_normalized, tensor_normalized.t())
                # Set d diagonal to 1 (maximum distance for cosine distance)
                d = d.fill_diagonal_(1.0)

                # Nearest neighbor for each sample
                nearneigh, _ = torch.min(d, dim=1)
                # Minimum distance in d matrix
                _, min_indices = torch.min(nearneigh, dim=0)
                
                # Get index to remove from non-None list of indices
                idx_to_remove = non_none_indices[min_indices.item()]

                # Remove sample with smallest distance
                self.buffer.pop(idx_to_remove)
                self.buffer_features.pop(idx_to_remove)
                self.stored_samples -= 1

        # Add samples to buffer
        samples_to_add = [batch_x[i] for i in range(batch_size)]
        self.buffer.extend(samples_to_add)
        self.buffer_features.extend([None for i in range(batch_size)])
        self.stored_samples += batch_size


    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= self.stored_samples

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = [self.buffer[i] for i in indices]

        # Reconstruct batch from samples
        batch_x = torch.stack([sample for sample in samples])

        return batch_x, indices
    
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

