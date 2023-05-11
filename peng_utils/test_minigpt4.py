import torch

from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .minigpt4.models import *
from .minigpt4.processors import *

CFG_PATH = 'peng_utils/minigpt4/minigpt4_eval.yaml'


class TestMiniGPT4:
    def __init__(self, cfg_path=CFG_PATH):
        cfg = Config(cfg_path)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.chat.device = device
            self.model = self.model.to(device, dtype=dtype)
            self.chat.move_stopping_criteria_device(device, dtype=dtype)
        else:
            dtype = torch.float32
            device = 'cpu'
            self.chat.device = 'cpu'
            self.model = self.model.to('cpu', dtype=dtype)
            self.chat.move_stopping_criteria_device('cpu', dtype=dtype)
        
        return device, dtype
    
    def generate(self, text, image=None, device=None, keep_in_device=False):
        try:
            device, dtype = self.move_to_device(device)
            chat_state = CONV_VISION.copy()
            img_list = []
            if image is not None:
                self.chat.upload_img(image, chat_state, img_list)
            self.chat.ask(text, chat_state)
            llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]
            
            if not keep_in_device:
                self.chat.device = 'cpu'
                self.model = self.model.to('cpu', dtype=torch.float32)
                self.chat.move_stopping_criteria_device('cpu', dtype=torch.float32)

            return llm_message
        except Exception as e:
            return getattr(e, 'message', str(e))