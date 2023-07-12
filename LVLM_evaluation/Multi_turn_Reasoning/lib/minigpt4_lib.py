import torch

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.models import *
from minigpt4.processors import *
from PIL import Image
import cv2

CFG_PATH = './minigpt4/minigpt4_eval.yaml'

class MiniGPT4():
    def __init__(self, model_type="minigpt4", device="cuda"):
        self.device = device
        self.dtype = torch.float16
        self.model_type = model_type    
        cfg = Config(CFG_PATH)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        #model = model_cls.from_config(model_config).to(self.device, dtype=self.dtype)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to(self.device, dtype=self.dtype)
        self.chat = Chat(model, vis_processor, device=self.device)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)


    def ask(self, img_path, question):
        chat_state = CONV_VISION.copy()
        img_list = []        
        image = Image.open(img_path).convert('RGB')
        self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]
        llm_message = llm_message.strip()
        return llm_message


    def caption(self, img_path, max_gen_len=64, temperature=0.1, top_p=0.75):
        # TODO: Multiple captions
        chat_state = CONV_VISION.copy()
        img_list = []     
        image = Image.open(img_path).convert('RGB')
        self.chat.upload_img(image, chat_state, img_list)
        question = ""
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]
        llm_message = llm_message.replace('\n', ' ').strip()  # trim caption
        return llm_message

