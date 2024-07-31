import random
import torch



class HybridMinRedFIFOBuffer:
    """
    Hybrid MinRed-FIFO buffer class for batches of samples without labels.
    """
    def __init__(self, fifo_buffer_size, minred_buffer_size, alpha_ema=0.5, device='cpu'):

        self.fifo_buffer_size = fifo_buffer_size
        self.minred_buffer_size = minred_buffer_size
        self.total_buffer_size = fifo_buffer_size + minred_buffer_size
        self.alpha_ema = alpha_ema
        self.buffer_fifo = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features_fifo = torch.empty(0,1).to(device) # Buffer for corresponding sample features

        self.buffer_minred = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features_minred = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.device = device
        self.using_only_fifo = True

    # Add a batch of samples to the buffer
    def add(self, batch_x, batch_features, aligner = None):
        assert batch_x.size(0) == batch_features.size(0)

        # Initialize empty buffers
        if self.buffer_fifo.size(0) == 0:
            # Extend buffer to have same dim of batch_x
            buffer_shape = list(batch_x.size())
            buffer_shape[0] = 0
            self.buffer_fifo = torch.empty(buffer_shape).to(self.device)
            # Extend buffer_features to have same dim of batch_features
            buffer_shape = list(batch_features.size())
            buffer_shape[0] = 0
            self.buffer_features_fifo = torch.empty(buffer_shape).to(self.device)

        # Store new samples in the FIFO buffer
        self.buffer_fifo = torch.cat((batch_x, self.buffer_fifo), dim=0)
        self.buffer_features_fifo = torch.cat((batch_features, self.buffer_features_fifo), dim=0)

        if self.using_only_fifo and self.buffer_fifo.size(0) > self.total_buffer_size:
        # If just surpassed max number of samples:
        # stop using FIFO only policy and initialize MinRed buffer with the oldest samples of the FIFO buffer
            self.using_only_fifo = False

            self.buffer_minred = self.buffer_fifo[-self.minred_buffer_size:].clone().detach()
            self.buffer_features_minred = self.buffer_features_fifo[-self.minred_buffer_size:].clone().detach()

            self.buffer_fifo = self.buffer_fifo[:-self.minred_buffer_size]
            self.buffer_features_fifo = self.buffer_features_fifo[:-self.minred_buffer_size]

        if not self.using_only_fifo:
            for i in range(self.buffer_fifo.size(0) - self.fifo_buffer_size):
                # Update the buffers, keep in the MinRed buffer only decorrelated samples

                if aligner is not None:
                    aligner.eval()
                    with torch.no_grad():
                        aligned_fifo = aligner(self.buffer_features_fifo)
                    aligner.train()
                else:
                    aligned_fifo = self.buffer_features_fifo
                
                # Cosine distance = 1 - cosine similarity
                fifo_norm = torch.nn.functional.normalize(aligned_fifo, p=2, dim=1)
                minred_norm = torch.nn.functional.normalize(self.buffer_features_minred, p=2, dim=1)
                d_fifo_minred = 1 - torch.mm(fifo_norm, minred_norm.t())
                d_minred = 1 - torch.mm(minred_norm, minred_norm.t())
                # Set d diagonal to 1 (maximum distance for cosine distance)
                d_minred = d_minred.fill_diagonal_(1.0)
                # Nearest neighbor among the MinRed buffer for MinRed sample
                nearneigh_minred, _ = torch.min(d_minred, dim=1)
                # Nearest MinRed nieghbor of each FIFO sample
                nearneigh_fifo, _ = torch.min(d_fifo_minred, dim=1)

                # Select minimum distance between minred samples and maximum distance among fifo samples
                max_dist_fifo, max_indices_fifo = torch.min(nearneigh_fifo, dim=0)
                min_dist_minred, min_indices_minred = torch.min(nearneigh_minred, dim=0)

                if max_dist_fifo > min_dist_minred:
                    # Get index of sample to remove
                    idx_to_remove_fifo = max_indices_fifo.item()
                    idx_to_remove_minred = min_indices_minred.item()

                    # Substitute sample in MinRed buffer with sample in FIFO buffer
                    self.buffer_minred[idx_to_remove_minred] = self.buffer_fifo[idx_to_remove_fifo]
                    self.buffer_features_minred[idx_to_remove_minred] = self.buffer_features_fifo[idx_to_remove_fifo]

                    # Remove sample from FIFO buffer
                    self.buffer_fifo = torch.cat((self.buffer_fifo[:idx_to_remove_fifo], self.buffer_fifo[idx_to_remove_fifo + 1:]), dim=0)
                    self.buffer_features_fifo = torch.cat((self.buffer_features_fifo[:idx_to_remove_fifo], self.buffer_features_fifo[idx_to_remove_fifo + 1:]), dim=0)

                else:
                    break

            if self.buffer_fifo.size(0) > self.fifo_buffer_size:
                self.buffer_fifo = self.buffer_fifo[:self.fifo_buffer_size]
                self.buffer_features_fifo = self.buffer_features_fifo[:self.fifo_buffer_size]



    def sample_fifo(self, batch_size_fifo):
    # Sample batch_size samples from FIFO buffer, 
    # returns samples and indices of extracted samples (for feature update)
        assert batch_size_fifo <= len(self.buffer_fifo)

        fifo_indices = random.sample(range(self.buffer_fifo.size(0)), batch_size_fifo)
        batch_x_fifo = self.buffer_fifo[fifo_indices]
        batch_features_fifo = self.buffer_features_fifo[fifo_indices]

        return batch_x_fifo, batch_features_fifo, fifo_indices

    def sample_minred(self, batch_size_minred):
    # Sample batch_size samples from MinRed buffer,
    # returns samples and indices of extracted samples (for feature update)
        assert batch_size_minred <= len(self.buffer_minred) and self.using_only_fifo == False

        minred_indices = random.sample(range(len(self.buffer_minred)), batch_size_minred)
        batch_x_minred = self.buffer_minred[minred_indices]
        batch_features_minred = self.buffer_features_minred[minred_indices]

        return batch_x_minred, batch_features_minred, minred_indices
    

    # Update features of buffer samples at given indices
    def update_features_fifo(self, batch_features_fifo, fifo_indices):
        assert batch_features_fifo.size(0) == len(fifo_indices)

        batch_features_fifo = batch_features_fifo.to(self.device)

        for i, idx in enumerate(fifo_indices):
            if self.buffer_features_fifo[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features_fifo[idx] = self.alpha_ema * self.buffer_features_fifo[idx] + (1 - self.alpha_ema) * batch_features_fifo[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features_fifo[idx] = batch_features_fifo[i]

    def update_features_minred(self, batch_features_minred, minred_indices):
        assert batch_features_minred.size(0) == len(minred_indices)

        batch_features_minred = batch_features_minred.to(self.device)

        for i, idx in enumerate(minred_indices):
            if self.buffer_features_minred[idx] is not None:
                # There are already features stored for that sample
                # EMA update of features
                self.buffer_features_minred[idx] = self.alpha_ema * self.buffer_features_minred[idx] + (1 - self.alpha_ema) * batch_features_minred[i]
            else:
                # No features stored yet, store newly passed features
                self.buffer_features_minred[idx] = batch_features_minred[i]