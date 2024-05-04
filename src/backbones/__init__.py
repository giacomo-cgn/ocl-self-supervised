from torchvision import models
from torch import nn
from typing import Tuple

from .custom_resnets import ResNet18VariableWidth, ResNet9VariableWidth

def get_encoder(encoder_name) -> Tuple[nn.Module, int]:
    """Returns an initialized encoder without the last clf layer and the encoder feature dimensions."""

    if encoder_name == 'resnet18':
        encoder = models.resnet18(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'resnet34':
        encoder = models.resnet34(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'resnet50':
        encoder = models.resnet50(zero_init_residual=True)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'resnet9':
        encoder = ResNet9VariableWidth(num_base_features=64)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'wide_resnet18':
        encoder = ResNet18VariableWidth(zero_init_residual=True, nf=128)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'slim_resnet18':
        encoder = ResNet18VariableWidth(zero_init_residual=True, nf=20)
        dim_encoder_features = encoder.fc.weight.shape[1]
        print('DIM_ENCODER_FEATURES SHAPE:', encoder.fc.weight.shape)
        encoder.fc = nn.Identity()


    elif encoder_name == 'wide_resnet9':
        encoder = ResNet9VariableWidth(num_base_features=128)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()

    elif encoder_name == 'slim_resnet9':
        encoder = ResNet9VariableWidth(num_base_features=20)
        dim_encoder_features = encoder.fc.weight.shape[1]
        encoder.fc = nn.Identity()
        
    else:
        raise Exception(f'Invalid encoder {encoder_name}')
    
    return encoder, dim_encoder_features