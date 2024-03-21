from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.pmc_oa.timm_model import TimmModel
# from timm_model import TimmModel

from torch.utils.checkpoint import checkpoint
from models.pmc_oa.blocks import Bottleneck, AttentionPool2d, ResNet, ModifiedResNet, LayerNorm, QuickGELU 
# from blocks import Bottleneck, AttentionPool2d, ResNet, ModifiedResNet, LayerNorm, QuickGELU

@dataclass
class CLIPVisionCfg:
    backbone: str = 'ModifiedRN50'  # ['RN50', 'ModifiedRN50', 'MAE']
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 64
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 512
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    patch_dropout: float = 0.0  # patch dropout rate, no dropout by default
    drop_attention_rate: float = 0.  # Transformer Dropout



class PMC_CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg = CLIPVisionCfg,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        vision_cfg.layers = [3,4,6,3]
        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            VisualBackbone = {
                "RN50": ResNet,
                "ModifiedRN50": ModifiedResNet,
            }[vision_cfg.backbone]
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width

            self.visual = VisualBackbone(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        self.init_parameters()

    def init_parameters(self):
        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, image):
        image_features = self.encode_image(image)
        # image_features = F.normalize(image_features['image_features'], dim=-1)  # torch.Size([2, 2048, 16, 16])
        return image_features
    
