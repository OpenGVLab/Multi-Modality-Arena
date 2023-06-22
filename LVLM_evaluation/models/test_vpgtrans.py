import torch
import random
from types import MethodType
from .vpgtrans.common.config import Config
from .vpgtrans.common.registry import registry
from .vpgtrans.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from .vpgtrans.models import *
from .vpgtrans.processors import *

from . import get_image, DATA_DIR

CFG_PATH = 'models/vpgtrans/vpgtrans_demo.yaml'


@torch.no_grad()
def forward_lm(self, samples):
    image = samples["image"]
    image = image.to(self.device)
    img_embeds, atts_img = self.encode_img(image)
    if hasattr(samples, 'question_split'):  # VQA dataset
        print('VQA Batch')
        vqa_prompt = '###Human: <Img><ImageHere></Img> '
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
    elif self.prompt_list:
        prompt = random.choice(self.prompt_list)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

    self.llama_tokenizer.padding_side = "right"

    to_regress_tokens = self.llama_tokenizer(
        [samples["prompt"]] * image.size(0),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=self.max_txt_len,
        add_special_tokens=False
    ).to(image.device)

    targets = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
    )

    start_loc = samples["start_loc"]
    targets[0, :start_loc] = -100
    targets[0, start_loc + 1:] = -100

    empty_targets = (
        torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    batch_size = img_embeds.shape[0]
    bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.llama_model.model.embed_tokens(bos)
    atts_bos = atts_img[:, :1]

    to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

    with self.maybe_autocast():
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    return outputs


class TestVPGTrans:
    def __init__(self, device=None):
        cfg = Config(CFG_PATH, DATA_DIR)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        self.model.forward_lm = MethodType(forward_lm, self.model)

        # print(f'Check the number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        # print(f'Check the number of whole parameters: {sum(p.numel() for p in self.model.parameters())}')

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.chat.device = self.device
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

    def forward_lm(self, image, prompt, start_loc):
        image = get_image(image)
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
        chat_state = CONV_VISION.copy()
        chat_state.append_message(chat_state.roles[0], "<Img><ImageHere></Img>")
        self.chat.ask(prompt, chat_state)
        prompt = chat_state.get_prompt()
        outputs = self.model.forward_lm({"image": image, "prompt": prompt, "start_loc": start_loc})
        loss = outputs.loss

        return loss