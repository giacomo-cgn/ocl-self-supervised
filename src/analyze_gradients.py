import os
import torch

class GradientAnalyzer:
    def __init__(self, save_pth=None):

        if save_pth is not None:
            analyze_gradient_folder = os.path.join(save_pth, 'analyze_gradients')
            os.makedirs(analyze_gradient_folder, exist_ok=True)
            self.analyze_gradients_file = os.path.join(analyze_gradient_folder, 'gradients_norm.csv')
            with open(self.analyze_gradients_file, 'a') as f:
                f.write('exp_idx,tr_step,buffer_loss_sum,stream_loss_sum,buffer_loss_mean,stream_loss_mean,buffer_grad_sum,stream_grad_sum,buffer_grad_mean,stream_grad_mean\n')
            
        

    def analyze_gradients(self, buffer_losses, stream_losses, params, exp_idx, tr_step):

        trainable_params = [p for p in params if p.requires_grad]

        buffer_loss_sum = buffer_losses.sum()
        stream_loss_sum = stream_losses.sum()
        buffer_loss_mean = buffer_losses.mean()
        stream_loss_mean = stream_losses.mean()

        grads_buffer_sum = torch.autograd.grad(buffer_loss_sum, trainable_params, retain_graph=True)
        grads_stream_sum = torch.autograd.grad(stream_loss_sum, trainable_params, retain_graph=True)
        grads_buffer_mean = torch.autograd.grad(buffer_loss_mean, trainable_params, retain_graph=True)
        grads_stream_mean = torch.autograd.grad(stream_loss_mean, trainable_params, retain_graph=True)

        # Compute L2-norms of the gradients
        norm_grad_buffer_sum  = net_grad_norm(grads_buffer_sum)
        norm_grad_stream_sum  = net_grad_norm(grads_stream_sum)
        norm_grad_buffer_mean = net_grad_norm(grads_buffer_mean)
        norm_grad_stream_mean = net_grad_norm(grads_stream_mean)

        with open(self.analyze_gradients_file, 'a') as f:
            f.write(f'{exp_idx},{tr_step},{buffer_loss_sum},{stream_loss_sum},{buffer_loss_mean},{stream_loss_mean},{norm_grad_buffer_sum},{norm_grad_stream_sum},{norm_grad_buffer_mean},{norm_grad_stream_mean}\n')        
     



def net_grad_norm(grads):
    """ Compute L2-norms of the summed gradients 

    Args:
        grads (list): List of gradients
    Returns:
        float: L2-norm of the summed gradients    
    """
    sq = 0.0
    for g in grads:
        sq += g.flatten().pow(2).sum().item()
    return sq ** 0.5


