from torch.optim.lr_scheduler import CosineAnnealingLR


def init_scheduler(scheduler_name: str, optimizer, total_tr_steps: int):
    if scheduler_name == None:
        return None
    elif scheduler_name == 'cosine':
        CosineAnnealingLR(optimizer, T_max=total_tr_steps, eta_min=0)
    else:
        raise ValueError(f"Invalid scheduler {scheduler_name}")