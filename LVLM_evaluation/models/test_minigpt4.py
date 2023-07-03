import torch
from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .minigpt4.models import *
from .minigpt4.processors import *

from . import get_image, DATA_DIR

CFG_PATH = 'models/minigpt4/minigpt4_eval.yaml'


class TestMiniGPT4:
    def __init__(self):
        cfg = Config(CFG_PATH, DATA_DIR)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.move_to_device()

    def move_to_device(self):
        if torch.cuda.is_available():
            self.dtype = torch.float16
            self.device = 'cuda'
            self.chat.device = 'cuda'
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.chat.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        chat_state = CONV_VISION.copy()
        img_list = []
        if image is not None:
            image = get_image(image)
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=max_new_tokens)[0]

        return llm_message

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        image_list = [get_image(image) for image in image_list]
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        batch_outputs = self.chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=max_new_tokens)
        return batch_outputs