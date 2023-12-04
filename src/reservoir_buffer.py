import random
import torch

class ReservoirBufferUnlabeled:
    """
    Custom reservoir buffer class for batches of samples without labels.
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.stored_samples = 0

    # Add a batch of samples to the buffer
    def add(self, batch_x):
        batch_size = batch_x.size(0)

        if self.stored_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.stored_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                samples = [batch_x[i] for i in range(batch_size)]
                self.buffer.extend(samples)
                self.stored_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.stored_samples
                samples = [batch_x[i] for i in range(batch_size)]
                self.buffer.extend(samples[:remaining_space])
                self.stored_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/stored_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.stored_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = batch_x[i]
            
            self.stored_samples += batch_size

    # Sample batch_size samples from the buffer
    def sample(self, batch_size):
        assert batch_size <= self.stored_samples

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        # Reconstruct batch from samples and return
        return torch.stack([self.buffer[i] for i in indices])



class ReservoirBufferUnlabeledFeatures:
    """
    Custom reservoir buffer class for batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.stored_samples = 0

    # Add a batch of samples and features to the buffer
    def add(self, batch_x, batch_features):
        assert batch_x.size(0) == batch_features.size(0)

        batch_size = batch_x.size(0)

        if self.stored_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.stored_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                samples = [(batch_x[i], batch_features[i]) for i in range(batch_size)]
                self.buffer.extend(samples)
                self.stored_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.stored_samples
                samples = [(batch_x[i], batch_features[i]) for i in range(batch_size)]
                self.buffer.extend(samples[:remaining_space])
                self.stored_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/stored_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.stored_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = (batch_x[i], batch_features[i])
            
            self.stored_samples += batch_size

    # Sample batch_size samples from the buffer
    def sample(self, batch_size):
        assert batch_size <= self.stored_samples

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        samples = samples = [self.buffer[i] for i in indices]


        # Stack each element of the tuples, each sample is a tuple of (x, features)
        batch_x = torch.stack([sample[0] for sample in samples])
        batch_features = torch.stack([sample[1] for sample in samples])

        return batch_x, batch_features
