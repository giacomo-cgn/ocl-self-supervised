from torchvision import models
from torch import nn
from typing import Tuple

from .custom_resnets import ResNet18VariableWidth, ResNet9VariableWidth
from .vit import ViT

def get_encoder(encoder_name, image_size, ssl_model_name, vit_avg_pooling, save_pth=None) -> Tuple[nn.Module, int]:
    """Returns an initialized encoder without the last clf layer and the encoder feature dimensions."""

    if encoder_name == 'resnet18':
        encoder = models.resnet18(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        if image_size == 32:
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()

    elif encoder_name == 'resnet34':
        encoder = models.resnet34(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        if image_size == 32:
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()

    elif encoder_name == 'resnet50':
        encoder = models.resnet50(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        if image_size == 32:
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()

    elif encoder_name == 'resnet9':
        encoder = ResNet9VariableWidth(num_base_features=64)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'wide_resnet18':
        encoder = ResNet18VariableWidth(zero_init_residual=True, nf=128)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        if image_size == 32:
            encoder.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()

    elif encoder_name == 'slim_resnet18':
        encoder = ResNet18VariableWidth(zero_init_residual=True, nf=20)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        if image_size == 32:
            encoder.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity()


    elif encoder_name == 'wide_resnet9':
        encoder = ResNet9VariableWidth(num_base_features=128)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'slim_resnet9':
        encoder = ResNet9VariableWidth(num_base_features=20)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'vit_tiny':
        if image_size == 32:
            encoder = ViT(image_size=image_size, patch_size=2, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 64:
            encoder = ViT(image_size=image_size, patch_size=4, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 224:
            encoder = ViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        elif image_size == 256:
            encoder = ViT(image_size=image_size, patch_size=16, return_avg_pooling=vit_avg_pooling, save_pth=save_pth)
        else:
            raise Exception(f'Invalid image size for ViT backbone: {image_size}')
        dim_encoder_features = encoder.emb_dim
        if not ssl_model_name == 'mae':
            print("strategy name:", ssl_model_name)
            # Wrap to return only 1 feature tensor
            encoder = encoder.return_features_wrapper()
            print("Restituisce wrapper!!!!")          
        
    else:
        raise Exception(f'Invalid encoder: {encoder_name}')
    
    # Print number of parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params_in_millions = total_params / 1e6
    print(f'NUM PARAMS for {encoder_name}: {total_params_in_millions:.1f}M')
    
    return encoder, dim_encoder_features