import torch
import numpy as np

class MetricsAwareBufferMean:
    """
    Custom buffer that removes elements based on feature metrics and loss. Can store batches of samples without labels, but with encoder features.
    """
    def __init__(self, buffer_size, alpha_ema=1.0, alpha_ema_loss=0.0, 
                 device='cpu',
                 gamma_loss=0.5, gamma_extraction=0.5,
                 gamma_loss_out=0.5, gamma_extraction_out=0.5
                 ):
        
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.buffer_loss = torch.empty(0,1).to(device) # Buffer for corresponding sample losses
        self.alpha_ema = alpha_ema # 1.0 = do not update stored features, 0.0 = substitute with new features
        self.alpha_ema_loss = alpha_ema_loss # 1.0 = do not update stored losses, 0.0 = substitute with new losses

        self.gamma_loss = gamma_loss # how much weight is given to the normalized loss when selecting samples to remove
        self.gamma_extraction = gamma_extraction # how much weight is given to the normalized num of extractions when selecting samples to remove

        self.gamma_loss_out = gamma_loss_out # how much weight is given to the normalized loss when selecting samples to extract
        self.gamma_extraction_out = gamma_extraction_out # how much weight is given to the normalized num of extractions when selecting samples to extract

        self.buffer_e_stats = {} # dict containing encoder (e) stats, each is a list
        self.buffer_z_stats = {} # dict containing projector (z) stats, each is a list

        self.device = device
        self.lifetimes = torch.empty(0, dtype=torch.int) # Buffer for the life of each sample
        self.extractions = torch.empty(0, dtype=torch.int) # Buffer for the number of times each sample has been extracted
        self.finished_lifetimes = []
        self.finished_extractions = []

        self.seen_samples = 0 # Samples seen so far

    # Add a batch of samples, features and losses to the buffer
    def add(self, batch_x, batch_features, batch_loss, e_stats=None, z_stats=None):
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

            # Initialize buffer_e_stats and buffer_z_stats
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
                self.buffer_loss = torch.cat((self.buffer_loss, batch_loss), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)
                self.seen_samples += batch_size
                if e_stats is not None:
                    for key in e_stats.keys():
                        self.buffer_e_stats[key].extend(e_stats[key])
                if z_stats is not None:
                    for key in z_stats.keys():
                        self.buffer_z_stats[key].extend(z_stats[key])
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.seen_samples
                self.buffer = torch.cat((self.buffer, batch_x[:remaining_space]), dim=0)
                self.buffer_features = torch.cat((self.buffer_features, batch_features[:remaining_space]), dim=0)
                self.buffer_loss = torch.cat((self.buffer_loss, batch_loss[:remaining_space]), dim=0)
                self.lifetimes = torch.cat((self.lifetimes, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                self.extractions = torch.cat((self.extractions, torch.zeros(remaining_space, dtype=torch.int)), dim=0)
                self.seen_samples += remaining_space
                if e_stats is not None:
                    for key in e_stats.keys():
                        self.buffer_e_stats[key].extend(e_stats[key][:remaining_space])
                if z_stats is not None:
                    for key in z_stats.keys():
                        self.buffer_z_stats[key].extend(z_stats[key][:remaining_space])
        else:
            # Loss-based insertion: add the batch then keep the top-scored buffer_size samples.
            self.buffer = torch.cat((self.buffer, batch_x), dim=0)
            self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)
            self.buffer_loss = torch.cat((self.buffer_loss, batch_loss), dim=0)
            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key] += e_stats[key]
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key] += z_stats[key]

            self.lifetimes = torch.cat((self.lifetimes, torch.zeros(batch_size, dtype=torch.int)), dim=0)
            self.extractions = torch.cat((self.extractions, torch.zeros(batch_size, dtype=torch.int)), dim=0)
            # Remove both low- and high-score samples.
            sorted_indices = self.calculate_scores().argsort().cpu()
            num_remove_low = batch_size // 2
            num_remove_high = batch_size - num_remove_low
            low_indices = sorted_indices[:num_remove_low]
            high_indices = sorted_indices[-num_remove_high:]
            indices_to_remove = torch.cat((low_indices, high_indices), dim=0)
            self.finished_lifetimes += self.lifetimes[indices_to_remove].tolist()
            self.finished_extractions += self.extractions[indices_to_remove].tolist()

            keep_mask = torch.ones(sorted_indices.size(0), dtype=torch.bool)
            keep_mask[indices_to_remove] = False
            indices_to_keep = torch.arange(sorted_indices.size(0))[keep_mask]
            self.buffer = self.buffer[indices_to_keep]
            self.buffer_features = self.buffer_features[indices_to_keep]
            self.buffer_loss = self.buffer_loss[indices_to_keep]
            self.lifetimes = self.lifetimes[indices_to_keep]
            self.extractions = self.extractions[indices_to_keep]

            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key] = [self.buffer_e_stats[key][j] for j in indices_to_keep]
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key] = [self.buffer_z_stats[key][j] for j in indices_to_keep]

            self.seen_samples += batch_size


    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size):
        assert batch_size <= len(self.buffer)

        # Loss-stochastic extraction.
        scores = self.calculate_scores_out()
        probabilities = scores.softmax(dim=0)
        indices = torch.multinomial(probabilities, batch_size, replacement=False)


        self.extractions[indices] += 1

        # Get sample batch from indices
        batch_x = self.buffer[indices]
        batch_features = self.buffer_features[indices]

        return batch_x, batch_features, indices
    
    # Update features of buffer samples at given indices
    def update_features(self, batch_features, indices, batch_loss, e_stats=None, z_stats=None):
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

            if e_stats is not None:
                for key in e_stats.keys():
                    self.buffer_e_stats[key][idx] = self.alpha_ema * self.buffer_e_stats[key][idx] + (1 - self.alpha_ema) * e_stats[key][i]
            if z_stats is not None:
                for key in z_stats.keys():
                    self.buffer_z_stats[key][idx] = self.alpha_ema * self.buffer_z_stats[key][idx] + (1 - self.alpha_ema) * z_stats[key][i]


    def calculate_scores(self):
        # Calculate elimination scores for all samples in the buffer
        # Prefer maintaining samples with high loss and low extraction count
    
        # 0-1 normalization of metrics
        norm_loss = ((self.buffer_loss - self.buffer_loss.min()) / (self.buffer_loss.max() - self.buffer_loss.min()).clamp(min=1e-6)).cpu()
        norm_extraction = 1 - ((self.extractions - self.extractions.min())  / (self.extractions.max() - self.extractions.min()).clamp(min=1e-6)).cpu()

        scores = self.gamma_loss * norm_loss + self.gamma_extraction * norm_extraction
        return scores
    
    def calculate_scores_out(self):
        # Calculate scores to select which sample to extract from the buffer
        # Use distance from mean for normalized loss and extraction.
    
        # 0-1 normalization of metrics
        norm_loss = ((self.buffer_loss - self.buffer_loss.min()) / (self.buffer_loss.max() - self.buffer_loss.min()).clamp(min=1e-6)).cpu()
        norm_extraction = 1 - ((self.extractions - self.extractions.min())  / (self.extractions.max() - self.extractions.min()).clamp(min=1e-6)).cpu()

        loss_distance_from_mean = (norm_loss - norm_loss.mean()).abs()
        extraction_distance_from_mean = (norm_extraction - norm_extraction.mean()).abs()

        scores = self.gamma_loss_out * loss_distance_from_mean + self.gamma_extraction_out * extraction_distance_from_mean
        return scores

        

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