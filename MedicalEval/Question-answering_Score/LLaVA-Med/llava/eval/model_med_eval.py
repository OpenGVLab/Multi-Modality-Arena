import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import random
import json
from tqdm import tqdm
from copy import deepcopy
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import difflib
from PIL import Image
import math
def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args, question_file, answers_base_path):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        
        print(model_name)
        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
            model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)


        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    # import pdb; pdb.set_trace()
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.join(answers_base_path, 'tmp', os.path.dirname(question_file).replace('/', '_') + 'pred.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    print('start inference...', flush=True)
    res = []
    for i, line in enumerate(tqdm(questions)):
        try:
            question, gt_ans, image = preprocess_input(line)

            #try:
            #    question = line["conversations"][0] # ['value'].split('\n')[0]
            #    gt_ans = line["conversations"][1] # ['value']        
            #except:
            #    question = line["conversatons"][0] # ['value'].split('\n')[0]
            #    gt_ans = line["conversatons"][1] # ['value']    

            qs = question['value']

            qs = qs.replace('<image>', '').strip()
            cur_prompt = qs

            #image_file = line["image"]
            #image = Image.open(os.path.join(args.image_folder, image_file))
            
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            cur_prompt = cur_prompt + '\n' + '<image>'

            if args.conv_mode == 'simple_legacy':
                qs += '\n\n### Response:'
            assert gt_ans['from'] == 'gpt'
            # conv = default_conversation.copy()
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=256,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            if args.conv_mode == 'simple_legacy':
                while True:
                    cur_len = len(outputs)
                    outputs = outputs.strip()
                    for pattern in ['###', 'Assistant:', 'Response:']:
                        if outputs.startswith(pattern):
                            outputs = outputs[len(pattern):].strip()
                    if len(outputs) == cur_len:
                        break

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()

            # prompt for answer
            if True: #args.answer_prompter:
                outputs_reasoning = outputs
                inputs = tokenizer([prompt + outputs_reasoning + ' ###\nANSWER:'])

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                keywords = ['###']
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        do_sample=True,
                        temperature=0.7,
                        max_new_tokens=64,
                        stopping_criteria=[stopping_criteria])

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

                try:
                    index = outputs.index(conv.sep)
                except ValueError:
                    outputs += conv.sep
                    index = outputs.index(conv.sep)

                outputs = outputs[:index].strip()
                # outputs = outputs_reasoning + '\n The answer is ' + outputs

            save_dict = deepcopy(line)
            save_dict['model_pred'] = outputs
            save_dict['prompt_question'] = cur_prompt

            ans_file.write(json.dumps(save_dict) + "\n")
            res.append(save_dict)
            ans_file.flush()
        except:
            continue
    ans_file.close()
    
    pred_dict, correct_precentage = MedicalEval(res)
    final_save_dict = {
        "model_name": model_name,
        "dataset_name": question_file,
        "correct_precentage" :correct_precentage,
        "pred_dict" : pred_dict
    }
    
    with open(os.path.join(answers_base_path, os.path.dirname(question_file).replace('/', '_') + '.json'), 'w') as f:
        json.dump(final_save_dict, f, indent=4, ensure_ascii=False)


def preprocess_input(entity) -> tuple:
    a,b,c,d = entity.get('option_A'), entity.get('option_B'), entity.get('option_C'), entity.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    q_str = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(answer_list)+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!\n<image>'
    ans_str = entity['gt_answer']
    
    question = {
        'from': 'human',
        'value': q_str
    }
    
    answer = {
        "from": "gpt",
        "value": ans_str
    }
    
    image_url = entity.get('image_path')
    image = read_img_from_url(image_url)
    return question, answer, image
    

from PIL import Image
import sys
from io import BytesIO

def read_img_from_url(url):
    img = Image.open(url)
    return img


def MedicalEval(pred_dict: list) -> tuple:
    tot = len(pred_dict)
    succ = 0
    for data in pred_dict:
        try:
            a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            
            if answer_list[find_most_similar_index(answer_list, data['model_pred'])] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except:
            continue
        
    return pred_dict, succ/tot
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VLP_web_data/llava_med/llava_med_in_text_60k")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple_legacy")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", default=True, action="store_true")
    parser.add_argument('--answers_base_path', type=str, default="llava_med_output")
    args = parser.parse_args()

    print('start', flush=True)
    os.makedirs(args.answers_base_path, exist_ok=True)
    eval_model(args, args.question_file, args.answers_base_path)
    print('finish', flush=True)