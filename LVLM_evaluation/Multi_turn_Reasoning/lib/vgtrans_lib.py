import torch

from vpgtrans.common.config import Config
from vpgtrans.common.registry import registry
#from minigpt4.conversation.conversation import Chat, CONV_VISION
from vpgtrans.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from vpgtrans.models import *
from vpgtrans.processors import *
import cv2
from PIL import Image

DATA_DIR = './VLP_web_data'
CFG_PATH = './vpgtrans/vpgtrans_demo.yaml'

class VPGTrans():
    def __init__(self, model_type="llava", device="cuda"):
        cfg = Config(CFG_PATH,DATA_DIR)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.device = device
        self.chat.device = self.device
        self.dtype = torch.float16
        self.model_type = model_type
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)


    
    def ask(self, image, question):
        chat_state = CONV_VISION.copy()
        img_list = [] 
        image = Image.open(image).convert("RGB")
        self.chat.upload_img(image, chat_state, img_list)  
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]

        return llm_message
    
    def caption(self, image):
        chat_state = CONV_VISION.copy()
        img_list = []  
        image = Image.open(image).convert("RGB")
        self.chat.upload_img(image, chat_state, img_list)      
        question='a photo of'
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list)[0]
        llm_message = llm_message.replace('\n', ' ').strip() 
        return llm_message


