import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from cheetah.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
                
            ret = ret.lstrip(self.sep)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system='',
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)



class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([2]).to(self.device)]  
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def batch_answer(self, batch_raw_img_list, batch_context, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                    repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000,
                    update_layer=16):
        embeds_list = []
        all_imgs = []
        for raw_img_list in batch_raw_img_list:
            images = []
            for raw_image in raw_img_list:
                raw_image = Image.open(raw_image).convert('RGB') 
                img = self.vis_processor(raw_image).unsqueeze(1).to(self.device)
                images.append(img)
            images = torch.cat(images, 1)
            all_imgs.append(images)
        all_imgs = torch.stack(all_imgs, 0)

        img_list, vit_list, att_list = [], [], []
        for j in range(all_imgs.size(2)):
            image = all_imgs[:,:,j,:,:]
            image_emb, image_att, vit_emb = self.model.encode_img(image)
            img_list.append(image_emb)
            vit_list.append(vit_emb)
            att_list.append(image_att)
        
        conv_list = []
        for context in batch_context:
            chat_state = CONV_VISION.copy()
            img_embd_list = []
            for i, text in enumerate(context.split("<ImageHere>")):
                if text != '' and text.strip()!='':
                    self.ask(text, chat_state)
                if i < len(raw_img_list):
                    if len(chat_state.messages)>0:
                        chat_state.messages[-1][1] = ' '.join([chat_state.messages[-1][1], "<Img><HereForImage></Img>"])
                    else:
                        chat_state.append_message(chat_state.roles[0], "<Img><HereForImage></Img>")
            chat_state.append_message(chat_state.roles[1], None)
            conv_list.append(chat_state)
        
        split_prompt = []
        for conv in conv_list:
            prompt = conv.get_prompt()
            cur_split_prompt = prompt.split('<HereForImage>')
            assert len(cur_split_prompt) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
            split_prompt.append(cur_split_prompt)
        embs, attention_mask, img_position_list, input_part_targets_len = self.batch_get_context_emb(split_prompt, img_list, att_list)
        
        assert embs.shape[1] + max_new_tokens < max_length
        with self.model.maybe_autocast():
            outputs = self.model.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                # new add
                update_layer = update_layer,
                image_position_list = img_position_list,
                input_part_targets_len = input_part_targets_len,
                all_image_embeds = torch.stack(vit_list,dim=1)
            )

        batch_outputs = []
        for output_token in outputs:
            if output_token[0] == 0:  
                output_token = output_token[1:]
            if output_token[0] == 1:  
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('</s>')[0]  # remove the stop sign '</s>'
            batch_outputs.append(output_text)
        return batch_outputs
                    

    def answer(self, raw_img_list, context, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000,
               update_layer=16):
        chat_state = CONV_VISION.copy()
        img_embd_list = []
        vit_list = []
        for i, text in enumerate(context.split("<Img><HereForImage></Img>")):
            if text != '' and text.strip()!='':
                self.ask(text, chat_state)
            if i < len(raw_img_list):
                self.upload_img(raw_img_list[i], chat_state, img_embd_list, vit_list)
        chat_state.append_message(chat_state.roles[1], None)
        embs, img_position_list, input_part_targets_len = self.get_context_emb(chat_state, img_embd_list)

        assert embs.shape[1] + max_new_tokens < max_length
        
        with self.model.maybe_autocast():
            outputs = self.model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                # new add
                update_layer = update_layer,
                image_position_list = img_position_list,
                input_part_targets_len = input_part_targets_len,
                all_image_embeds = torch.stack(vit_list,dim=1)
            )
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0]  # remove the stop sign '</s>'
        return output_text

    def upload_img(self, image, conv, img_list, vit_list, att_list=None):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, image_att, vit_emb = self.model.encode_img(image)
        img_list.append(image_emb)
        vit_list.append(vit_emb)
        if att_list is not None:
            att_list.append(image_att)
        if isinstance(conv, list):
            for c in conv:
                if len(c.messages)>0:
                    c.messages[-1][1] = ' '.join([c.messages[-1][1], "<Img><HereForImage></Img>"])
                else:
                    c.append_message(c.roles[0], "<Img><HereForImage></Img>")
        else:
            if len(conv.messages)>0:
                conv.messages[-1][1] = ' '.join([conv.messages[-1][1], "<Img><HereForImage></Img>"])
            else:
                conv.append_message(conv.roles[0], "<Img><HereForImage></Img>")
        msg = "Received."
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        # print(prompt)
        prompt_segs = prompt.split('<HereForImage>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        
        mixed_embs = []
        img_position_list = []
        img_start = 0
        for i in range(len(prompt_segs)):
            mixed_embs.append(seg_embs[i])
            if i != len(img_list):
                mixed_embs.append(img_list[i])
                img_start += seg_embs[i].size(1)
                img_end = img_start + img_list[i].size(1)
                img_position_list.append((img_start, img_end))
                img_start = img_end
        
        mixed_embs = torch.cat(mixed_embs, dim=1)
        
        input_part_targets_len = []
        for i in range(mixed_embs.size(0)):
            input_part_targets_len.append(mixed_embs.size(1)-1)
        input_part_targets_len = torch.tensor(input_part_targets_len)
        return mixed_embs, img_position_list, input_part_targets_len
    
    def batch_get_context_emb(self, split_prompt, img_list, img_attns):
        prompt_segs = []
        for i in range(len(img_list) + 1):
            prompt_segs.append([p[i] for p in split_prompt])
        
        self.model.llama_tokenizer.padding_side = "left"
        
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", padding=True, add_special_tokens=i == 0).to(self.device)
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t.input_ids) for seg_t in seg_tokens]
        seg_attns = [seg_t.attention_mask for seg_t in seg_tokens]
        
        mixed_embs = []
        mixed_attns = []
        img_position_list = []
        img_start = 0
        for i in range(len(prompt_segs)):
            mixed_embs.append(seg_embs[i])
            mixed_attns.append(seg_attns[i])
            if i != len(img_list):
                mixed_embs.append(img_list[i])
                mixed_attns.append(img_attns[i])
                img_start += seg_embs[i].size(1)
                img_end = img_start + img_list[i].size(1)
                img_position_list.append((img_start, img_end))
                img_start = img_end
        
        mixed_embs = torch.cat(mixed_embs, dim=1)
        mixed_attns = torch.cat(mixed_attns, dim=1)
        
        input_part_targets_len = []
        for i in range(mixed_embs.size(0)):
            input_part_targets_len.append(mixed_embs.size(1)-1)
        input_part_targets_len = torch.tensor(input_part_targets_len)
        
        return mixed_embs, mixed_attns, img_position_list, input_part_targets_len

    