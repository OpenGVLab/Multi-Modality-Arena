import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf
from .cheetah.common.config import Config
from .cheetah.common.registry import registry
from .cheetah.conversation.conversation import Chat, CONV_VISION

from .cheetah.models import *
from .cheetah.processors import *


class TestCheetah:
    def __init__(self, device=None) -> None:
        cfg_path = 'models/cheetah/cheetah_eval_vicuna.yaml'
        config = OmegaConf.load(cfg_path)
        cfg = Config.build_model_config(config)
        model_cls = registry.get_model_class(cfg.model.arch)
        model = model_cls.from_config(cfg.model).to('cuda')

        vis_processor_cfg = cfg.preprocess.vis_processor.eval
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda')

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=64):
        prompt = f"<Img><HereForImage></Img> {question}"
        image = [image]
        output = self.chat.answer(image, prompt, max_new_tokens=max_new_tokens)
        return output

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=512):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output

