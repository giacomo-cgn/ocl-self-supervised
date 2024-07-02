import torch

class AbstractStrategy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def before_experience(self):
        pass

    def before_mb_passes(self, stream_mbatch):
        return stream_mbatch

    def before_forward(self, batch):
        pass

    def before_forward(self, stream_mbatch) -> torch.Tensor:
        return stream_mbatch

    def after_transforms(self, x_views_list) -> list[torch.Tensor]:
        return x_views_list

    def after_forward(self, x_views_list, loss, z_list, e_list):
        return loss
    
    def after_backward(self):
        pass

    def after_mb_passes(self):
        pass
    
    def get_params(self) -> list:
        return []

    def get_name(self):
        return None