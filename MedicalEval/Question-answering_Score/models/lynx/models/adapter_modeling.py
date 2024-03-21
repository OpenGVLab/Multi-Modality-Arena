# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import torch
from torch import nn

from transformers.activations import ACT2FN


class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    """
    def __init__(
        self,
        input_size,
        down_sample=None,
    ):
        super().__init__()

        self.input_size = input_size

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        self.adapter_norm_before = nn.LayerNorm(self.input_size)
        self.adapter_down = nn.Linear(self.input_size, self.down_sample)
        self.non_linearity = ACT2FN["silu"]

        # seq_list = []
        # seq_list.append(self.adapter_norm_before)
        # seq_list.append(nn.Linear(self.input_size, self.down_sample))
        # seq_list.append(self.non_linearity)
        # self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # Additional scaling factor (from He et al. (2021))
        self.scaling = nn.Parameter(torch.ones(1))   

        self.adapter_down.apply(self._init_weights)
        self.adapter_up.apply(self._init_weights)

    def forward(self, x, residual_input):  # , residual_input=None):

        down = self.non_linearity(self.adapter_down(self.adapter_norm_before(x)))

        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        output = output + residual_input

        return output

    @staticmethod
    def _init_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


