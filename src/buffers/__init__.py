from .reservoir_buffer import ReservoirBuffer
from .scale_buffer import Memory
from .minred_buffer import MinRedBuffer
from .fifo_buffer import FIFOBuffer
from .fifo_last_buffer import FIFOLastBuffer
from .augmented_representations_buffer import AugmentedRepresentationsBuffer
from .hybrid_minred_fifo_buffer import HybridMinRedFIFOBuffer
from .loss_aware_buffer import LossAwareBuffer
from .hybrid_fifo_loss_buffer import HybridFIFOLossBuffer
from .metrics_aware_buffer import MetricsAwareBuffer
from .lars_buffer import LARSBuffer
from .per_buffer import PERBuffer

def get_buffer(buffer_type: str,
               mem_size: int = 2000,
               alpha_ema: int = 0.5,
               device: str = 'cpu',
               fifo_buffer_ratio: float = 0.25, # only for hybrid buffer
               alpha_ema_loss: float = 0.5, # only for loss aware buffer
               insertion_policy: str = 'loss', # only for loss aware buffer,
               extraction_policy: str = 'loss', # only for loss aware buffer,
               gamma_extraction: float = 0.5, # only for loss/metrics aware buffer
               gamma_loss: float = 0.5, # only for metrics aware buffer
               gamma_overlap: float = 0.5, # only for metrics aware buffer
               gamma_std_deviation: float = 0.5, # only for metrics aware buffer
               gamma_cosine_deviation: float = 0.5, # only for metrics aware buffer
               gamma_loss_out=0.5, # only for metrics aware buffer
               gamma_extraction_out=0.5, # only for metrics aware buffer
               gamma_overlap_out=0.5, # only for metrics aware buffer
               gamma_std_deviation_out=0.5, # only for metrics aware buffer
               gamma_cosine_deviation_out=0.5, # only for metrics aware buffer
               fifo_buffer_size: int = 200, # only for hybrid fifo loss-aware buffer
               loss_aware_batch_size: int = 128, # only for hybrid fifo loss-aware buffer
               scale_use_ema_embeddings: bool = False, # only for scale buffer
               scale_ema_embeddings_decay: float = 0.5, # only for scale buffer
               scale_use_torch_psa: bool = False, # only for scale buffer
               alpha_per: float = 0.6, # only for PER buffer
               epsilon_per: float = 1e-6, # only for PER buffer
               rank_based_per: bool = False # only for PER buffer

               ):
    
    if buffer_type == 'reservoir':
        return ReservoirBuffer(mem_size, alpha_ema, device=device)
    elif buffer_type == 'fifo':
        return FIFOBuffer(mem_size, alpha_ema)
    elif buffer_type == 'fifo_last':
        return FIFOLastBuffer(mem_size, alpha_ema)
    elif buffer_type == 'minred':
        return MinRedBuffer(mem_size, alpha_ema, device=device)
    elif buffer_type == 'augmented_representations':
        return AugmentedRepresentationsBuffer(mem_size, device=device)
    elif buffer_type == 'scale':
        return Memory(mem_size=mem_size, device=device, use_ema_embeddings=scale_use_ema_embeddings,
                      ema_embeddings_decay=scale_ema_embeddings_decay, use_torch_psa=scale_use_torch_psa)
    elif buffer_type == 'aug_rep':
        return AugmentedRepresentationsBuffer(mem_size, device=device)
    elif buffer_type == 'hybrid_minred_fifo':
        fifo_buffer_size = int(mem_size * fifo_buffer_ratio)
        minred_buffer_size = mem_size - fifo_buffer_size
        return HybridMinRedFIFOBuffer(fifo_buffer_size=fifo_buffer_size, minred_buffer_size=minred_buffer_size,
                                      alpha_ema=alpha_ema, device=device)
    elif buffer_type == 'loss_aware':
        return LossAwareBuffer(mem_size, alpha_ema, alpha_ema_loss=alpha_ema_loss, insertion_policy=insertion_policy, 
                               extraction_policy=extraction_policy, device=device, gamma_extraction=gamma_extraction)
    elif buffer_type == 'metrics_aware':
        return MetricsAwareBuffer(mem_size, alpha_ema, alpha_ema_loss=alpha_ema_loss, insertion_policy=insertion_policy, 
                               extraction_policy=extraction_policy, device=device, gamma_extraction=gamma_extraction,
                               gamma_loss=gamma_loss, gamma_overlap=gamma_overlap, gamma_std_deviation=gamma_std_deviation,
                               gamma_cosine_deviation=gamma_cosine_deviation,
                               gamma_loss_out=gamma_loss_out, gamma_extraction_out=gamma_extraction_out,
                               gamma_overlap_out=gamma_overlap_out, gamma_std_deviation_out=gamma_std_deviation_out,
                               gamma_cosine_deviation_out=gamma_cosine_deviation_out)
    
    elif buffer_type == 'hybrid_fifo_loss':
        return HybridFIFOLossBuffer(fifo_buffer_size=fifo_buffer_size, total_buffer_size=mem_size, loss_aware_batch_size=loss_aware_batch_size,
                                     alpha_ema=alpha_ema, alpha_ema_loss=alpha_ema_loss, insertion_policy=insertion_policy, 
                               extraction_policy=extraction_policy, device=device, gamma_extraction=gamma_extraction)
    elif buffer_type == 'lars':
        return LARSBuffer(mem_size, alpha_ema=alpha_ema, device=device)
    elif buffer_type == 'per':
        return PERBuffer(mem_size, alpha_ema=alpha_ema, alpha_ema_loss=alpha_ema_loss,
                         alpha_per=alpha_per, epsilon_per=epsilon_per, rank_based_per=rank_based_per)
    
    else:
        raise Exception(f'Buffer type {buffer_type} is not supported')