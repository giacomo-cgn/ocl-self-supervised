import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from .abstract_ssl_model import AbstractSSLModel


class MAE(torch.nn.Module, AbstractSSLModel):
    def __init__(self,
                 image_size: int = 32,
                 patch_size: int = 2,
                 emb_dim: int = 192,
                 encoder_layer: int = 12,
                 encoder_head: int = 3,  
                 decoder_layer: int = 4,
                 decoder_head: int = 3,
                 mask_ratio: float = 0.75,
                 eval_avg_pooling=False,
                 save_pth: str = None,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

        self.mask_ratio = mask_ratio
        self.eval_avg_pooling = eval_avg_pooling
        self.save_pth = save_pth

        self.model_name = 'mae'

        def mae_loss(predicted_img, img, mask):
            return torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
        
        self.criterion = mae_loss

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'image_size: {image_size}\n')
                f.write(f'patch_size: {patch_size}\n')
                f.write(f'emb_dim: {emb_dim}\n')
                f.write(f'encoder_layer: {encoder_layer}\n')
                f.write(f'encoder_head: {encoder_head}\n')
                f.write(f'decoder_layer: {decoder_layer}\n')
                f.write(f'decoder_head: {decoder_head}\n')
                f.write(f'mask_ratio: {mask_ratio}\n')
                f.write(f'eval_avg_pooling: {eval_avg_pooling}\n')

    def forward(self, x_views_list):
        img = x_views_list[0]

        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        loss = self.criterion(predicted_img, img, mask)
        return loss, [features], [features]
    
    def get_encoder(self): 
       return self.encoder
    
    def get_encoder_for_eval(self):
        class MAE_EncoderForEval(torch.nn.Module):
            def __init__(self, encoder, return_avg_pooling=False):
                super().__init__()
                self.encoder = encoder
                # Return avg pooling of transformer token outputs, otherwise only return clf token
                self.return_avg_pooling = return_avg_pooling
            def forward(self, x):
                features, _ = self.encoder(x)
                features = rearrange(features, 't b c -> b t c')
                if self.return_avg_pooling:
                    return features.mean(dim=1)
                else:
                    return features[:,0]
                    
        return MAE_EncoderForEval(self.encoder, return_avg_pooling=self.eval_avg_pooling)
    
    def get_projector(self):
        # No projector head
        return torch.nn.Identity()
    
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        # No projector head
        self.get_embedding_dim()
    
    def get_criterion(self):
        return self.criterion, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    # Default ViT encoder is ViT-Tiny
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
    
def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes