# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
            self,
            vision_width,
            text_width,
            dim_head=64,
            heads=8
    ):
        super().__init__()

        self.vision_width = vision_width
        self.text_width = text_width

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(vision_width)
        self.norm_latents = nn.LayerNorm(text_width)

        self.to_q = nn.Linear(text_width, inner_dim, bias=False)
        self.to_kv = nn.Linear(vision_width, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, text_width, bias=False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            vision_width,
            text_width,
            depth,
            dim_head=64,
            heads=8,
            num_latents=64,
            ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, text_width))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(vision_width=vision_width, text_width=text_width, dim_head=dim_head, heads=heads),
                FeedForward(dim=text_width, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(text_width)

    def forward(self, vision_embeds=None, vision_atts=None):
        x = vision_embeds

        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        latents = repeat(self.latents, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        v2t_feats = self.norm(latents).squeeze(dim=1)  # for image, squeeze dim=1
        v2t_atts = torch.ones(v2t_feats.shape[:2], dtype=torch.long, device=v2t_feats.device)

        return v2t_feats, v2t_atts
