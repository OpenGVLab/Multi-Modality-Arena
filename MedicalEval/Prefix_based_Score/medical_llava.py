import os
import json
import pandas
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel
from llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from torch.nn import CrossEntropyLoss

from PIL import Image
from io import BytesIO

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name


def get_conv(model_name):
    if "llava" in model_name.lower():
        if "v1" in model_name.lower():
            template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt_multimodal"
        else:
            template_name = "multimodal"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "koala" in model_name: # Hardcode the condition
        template_name = "bair_v1"
    elif "v1" in model_name:    # vicuna v1_1/v1_2
        template_name = "vicuna_v1_1"
    else:
        template_name = "v1"
    return conv_templates[template_name].copy()


def load_model(model_path, model_name, dtype=torch.float16, device='cpu'):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'llava' in model_name.lower():
        if 'mpt' in model_name.lower():
            model = LlavaMPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    elif 'mpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    # get image processor
    image_processor = None
    if 'llava' in model_name.lower():
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device=device)
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device=device, dtype=dtype)
        
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, image_processor, context_len


class LLaVA():
    def __init__(self, model_type="llava", device="cuda"):
        model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        model_name = get_model_name(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name)
        self.conv = get_conv(model_name)
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        self.dtype = torch.float16
        self.device = device
        self.model_type = model_type
        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

    
    def ask(self, image_path, question):
        imgs = [Image.open(image_path).convert('RGB')]
        imgs = [self.img_transform(x) for x in imgs]
        image = torch.stack(imgs, dim=0).to(self.device)  
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = self.do_generate(prompt, image, stop_str=stop_str, dtype=self.dtype)

        return output

    def do_generate(self, prompt, image, dtype=torch.float16, temperature=0.2, max_new_tokens=512, stop_str=None, keep_aspect_ratio=False):
        images = [image]
        assert len(images) == prompt.count(DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"

        if keep_aspect_ratio:
            new_images = []
            for image_idx, image in enumerate(images):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))
                # replace the image token with the image patch token in the prompt (each occurrence)
                cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:
            images = self.image_processor(images, return_tensors='pt')['pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = self.tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor([input_ids]).to(self.model.device),
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = self.model(input_ids=torch.as_tensor([[token]], device=self.model.device),
                            use_cache=True,
                            attention_mask=torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_idx is not None and token == stop_idx:
                break
            elif token == self.tokenizer.eos_token_id:
                break
            elif i == max_new_tokens - 1:
                break
   
        output = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
        if stop_str is not None:
            pos = output.rfind(stop_str)
            if pos != -1:
                output = output[:pos]
        
        return output
    


def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return answer_list


def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]




def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img


@torch.no_grad()
def test(model, dataset=None, model_type='llava', prompt_idx=4, save_path=''):
    with open(dataset) as f:
        data_all = json.load(f)
    cnt = 0
    correct = 0
    
    res = []
    for data in data_all:
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = data['gt_answer']
        img_path = data['image_path']  
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.tokenizer(prefix)
        start_loc = len(prefix_tokens.input_ids)
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        images = model.image_processor(raw_image, return_tensors='pt')['pixel_values']
        images = images.to(model.model.device, dtype=model.dtype)
        for candidate in candidates:
            max_new_tokens = 512
            prompt = prefix + " {}.".format(candidate)
            input_ids = model.tokenizer(prompt).input_ids
            max_src_len = model.context_len - max_new_tokens - 8
            input_ids = input_ids[-max_src_len:]
            input_ids = torch.as_tensor([input_ids]).to(model.model.device)
            out = model.model(input_ids,use_cache=True,images=images)
            logits = out.logits
            targets =  input_ids
            
            prompt = prefix + " {}.".format(candidate)
            prompt_tokens = model.tokenizer(prompt, return_tensors="pt")
            lang_t = prompt_tokens["input_ids"]
            
            prefix_tokens = model.tokenizer(prefix, return_tensors="pt")  
            lang_t1 = prefix_tokens["input_ids"]
            lang_diff = lang_t.shape[1] - lang_t1.shape[1]
            
            
            targets[0,:start_loc]=-100
            targets[0,start_loc+lang_diff:]=-100
            shift_logits = logits[...,:-1,:].contiguous()
            shift_labels = targets[...,1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1,50282),shift_labels.view(-1))
            
            candidate_scores.append(loss.item())
        data['confidence'] =  str(candidate_scores)
        candidate_scores = softmax(np.reciprocal(candidate_scores))
        pred = candidates[np.argmax(candidate_scores)]
        print(candidates, candidate_scores)
        data['model_pred'] = pred
        
        data['is_correct'] = 'yes' if pred == answer else 'no'
        if pred == answer:
            correct += 1
        res.append(data)
        
    acc = correct / cnt
    print("Accuracy: ", acc)
        
    final_res = {'model_name': model_type, 'dataset_name': dataset, 'correct_precentage': acc, 'pred_dict': res}
    
    
    with open('{}/{}.json'.format(save_path, dataset.replace('/', '_')), 'w') as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)
        

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--dataset_path", type=str, default='/path/to/datset')
    parser.add_argument("--answer_path", type=str, default="output_res")
    args = parser.parse_args()
    return args

def run(args):
    model_type = 'llava'
    vqa_model = LLaVA(device=torch.device("cuda")) 
    
    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    sub_dataset = args.dataset_path
    test(vqa_model, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)

if __name__ == "__main__":
    args = parse_args()
    run(args)