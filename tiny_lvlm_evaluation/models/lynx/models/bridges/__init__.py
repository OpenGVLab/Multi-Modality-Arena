# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import torch.nn as nn


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


class MLPBridge(nn.Module):
    def __init__(self, config, vision_width, text_width, add_vision_ln: bool):
        super(MLPBridge, self).__init__()

        self.ln_vision = nn.LayerNorm(vision_width) if add_vision_ln else None

        if config['bridge'] == 'affine':
            self.mlp = nn.Linear(vision_width, text_width)
        elif config['bridge'] == 'mlp':
            self.mlp = build_mlp(vision_width, text_width)
        else:
            raise NotImplementedError("bridge: ", config['bridge'])

    def forward(
        self,
        vision_embeds=None,
        vision_atts=None,
    ):

        if self.ln_vision is not None:
            vision_embeds = self.ln_vision(vision_embeds)

        v2t_feats = self.mlp(vision_embeds)

        return v2t_feats, vision_atts

