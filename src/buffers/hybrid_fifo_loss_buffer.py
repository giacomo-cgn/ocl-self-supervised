import random
import torch

from .loss_aware_buffer import LossAwareBuffer
from .fifo_buffer import FIFOBuffer



class HybridFIFOLossBuffer():
    """
    Hybrid FIFO-Loss Aware buffer class for batches of samples without labels.
    """
    def __init__(self, fifo_buffer_size, total_buffer_size, loss_aware_batch_size,
                 alpha_ema_loss=0.0, insertion_policy='random', extraction_policy='random', gamma_extraction=0.5, # Loss Aware buffer params
                 alpha_ema=0.5, device='cpu'):
        
        self.loss_aware_buffer = LossAwareBuffer(buffer_size=(total_buffer_size - fifo_buffer_size), alpha_ema_loss=alpha_ema_loss,
                                                 insertion_policy=insertion_policy, extraction_policy=extraction_policy,
                                                 gamma_extraction=gamma_extraction,  device=device)
        self.fifo_buffer = FIFOBuffer(fifo_buffer_size, alpha_ema=alpha_ema)
        self.loss_aware_batch_size = loss_aware_batch_size
        self.alpha_ema = alpha_ema
        self.device = device

       
        self.seen_samples = 0 # Samples seen so far


    # Add a batch of samples, features and losses to the buffer
    def add(self, batch_x, batch_features, batch_loss):
        assert batch_x.size(0) == batch_features.size(0) == batch_loss.size(0)

        # Add to FIFO buffer if there is space
        fifo_remaining_space = self.fifo_buffer.buffer_size - len(self.fifo_buffer.buffer)
        if batch_x.size(0) <= fifo_remaining_space:
            self.fifo_buffer.add(batch_x, batch_features, batch_loss)

        else:
            # Extract the oldest samples from the FIFO buffer
            needed_space = batch_x.size(0) - fifo_remaining_space
            fifo_oldest_samples = self.fifo_buffer.buffer[:needed_space]
            fifo_oldest_features = self.fifo_buffer.buffer_features[:needed_space]
            fifo_oldest_losses = self.fifo_buffer.buffer_loss[:needed_space]
            # Delete the oldest samples from the FIFO buffer
            self.fifo_buffer.buffer = self.fifo_buffer.buffer[needed_space:]
            self.fifo_buffer.buffer_features = self.fifo_buffer.buffer_features[needed_space:]
            self.fifo_buffer.buffer_loss = self.fifo_buffer.buffer_loss[needed_space:]
            # Save the lifetimes and extractions of the oldest samples
            self.fifo_buffer.finished_lifetimes = self.fifo_buffer.finished_lifetimes[:needed_space]
            self.fifo_buffer.finished_extractions = self.fifo_buffer.finished_extractions[:needed_space]
            # Delete the lifetimes and extractions of the oldest samples
            self.fifo_buffer.lifetimes = self.fifo_buffer.lifetimes[needed_space:]
            self.fifo_buffer.extractions = self.fifo_buffer.extractions[needed_space:]
            
            # Convert the oldest samples to tensors
            fifo_oldest_samples  = torch.stack(fifo_oldest_samples,  dim=0)
            fifo_oldest_features = torch.stack(fifo_oldest_features, dim=0)
            fifo_oldest_losses   = torch.stack(fifo_oldest_losses,   dim=0)

            # Add the oldest samples to the Loss Aware buffer
            self.loss_aware_buffer.add(fifo_oldest_samples, fifo_oldest_features, fifo_oldest_losses)

            # Add the new samples to the FIFO buffer
            self.fifo_buffer.add(batch_x, batch_features, batch_loss)


    def sample(self, batch_size):
        # How many samples to take from loss aware buffer
        self.loss_aware_curr_batch_size = min(self.loss_aware_batch_size, len(self.loss_aware_buffer.buffer))
        # How many samples to take from FIFO buffer
        fifo_curr_batch_size = min(batch_size - self.loss_aware_curr_batch_size, len(self.fifo_buffer.buffer))
       
        if len(self.loss_aware_buffer.buffer) > 0:
            # Sample from Loss Aware buffer if it is not empty
            loss_aware_x, loss_aware_features, loss_aware_indices = self.loss_aware_buffer.sample(self.loss_aware_curr_batch_size)
            if fifo_curr_batch_size > 0:
                # Sample from FIFO buffer
                fifo_x, fifo_features, fifo_indices = self.fifo_buffer.sample(fifo_curr_batch_size)
                # Concatenate samples from both buffers
                batch_x = torch.cat((loss_aware_x, fifo_x), dim=0)
                batch_features = torch.cat((loss_aware_features, fifo_features), dim=0)
                indices = loss_aware_indices + fifo_indices
            else:
                # Only sample from Loss Aware buffer
                batch_x = loss_aware_x
                batch_features = loss_aware_features
                indices = loss_aware_indices
        else:
            # Only sample from FIFO buffer
            fifo_x, fifo_features, fifo_indices = self.fifo_buffer.sample(fifo_curr_batch_size)
            batch_x = fifo_x
            batch_features = fifo_features
            indices = fifo_indices

        return batch_x, batch_features, indices
    
     # Update features of buffer samples at given indices
    def update_features(self, batch_features, batch_loss, indices):
        assert batch_features.size(0) == len(indices)

        loss_aware_features = batch_features[:self.loss_aware_curr_batch_size]
        loss_aware_loss = batch_loss[:self.loss_aware_curr_batch_size]
        loss_aware_indices = indices[:self.loss_aware_curr_batch_size]
        fifo_features = batch_features[self.loss_aware_curr_batch_size:]
        fifo_loss = batch_loss[self.loss_aware_curr_batch_size:]
        fifo_indices = indices[self.loss_aware_curr_batch_size:]
        self.loss_aware_buffer.update_features(loss_aware_features, loss_aware_loss, loss_aware_indices)
        self.fifo_buffer.update_features(fifo_features, fifo_loss, fifo_indices)


    def end(self):
        fifo_csv_buffer, fifo_metrics_buffer = self.fifo_buffer.end()
        loss_aware_csv_buffer, loss_aware_metrics_buffer = self.loss_aware_buffer.end()

        # concatenate (from 2 cols to 4) the two csv buffers
        csv_buffer = "lifetime_fifo,extraction_fifo,lifetime_loss_aware,extraction_loss_aware\n"
        for i in range(1, len(fifo_csv_buffer.split("\n")) - 1):
            csv_buffer += fifo_csv_buffer.split("\n")[i] + "," + loss_aware_csv_buffer.split("\n")[i] + "\n"


        metrics_buffer = f'FIFO buffer:\n{fifo_metrics_buffer}\nLoss-aware buffer:\n{loss_aware_metrics_buffer}'
       
        return csv_buffer, metrics_buffer
    
    def get_curr_len(self):
        return self.fifo_buffer.get_curr_len() + self.loss_aware_buffer.get_curr_len()
    
    def get_buffer_data(self):
        if self.loss_aware_buffer.get_curr_len() == 0:
            return self.fifo_buffer.get_buffer_data()
        return torch.cat((self.fifo_buffer.get_buffer_data(), self.loss_aware_buffer.get_buffer_data()))