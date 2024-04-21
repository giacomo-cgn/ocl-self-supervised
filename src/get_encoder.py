from torchvision import models

def get_encoder(encoder_name):
    if encoder_name == 'resnet18':
        encoder = models.resnet18
    elif encoder_name == 'resnet34':
        encoder = models.resnet34
    elif encoder_name == 'resnet50':
        encoder = models.resnet50
    else:
        raise Exception(f'Invalid encoder {encoder_name}')
    
    return encoder