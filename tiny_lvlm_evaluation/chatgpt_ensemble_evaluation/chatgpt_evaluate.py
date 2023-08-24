import pdb, os, json, glob,shutil, copy, time
import numpy as np
import openai
import copy
from prompt_lib import messages1
import argparse
import requests

root_path = '/path_to/original_item'
save_root = '/path_to/save_root_dir'
model_pool = ['LLaMA-Adapter-v2', 'VPGTrans', 'MiniGPT-4', 'BLIP2', 'InstructBLIP', 'Otter', 'Otter-Image', 'LLaVA', 'OFv2_4BI', 'mPLUG-Owl', 'PandaGPT', 'Bard']
dataset_pool = ['VCR1_MCI', 'CTW', 'IIIT5K', 'CIFAR100', 'IC13', 'VCR1_OC', 'WHOOPSWeird', 'OCRVQA', 
                'Flickr', 'WHOOPSCaption', 'IconQA', 'POIE', 'VQAv2', 'AOKVQAOpen', 'SVT', 'SROIE', 
                'DocVQA', 'Flowers102', 'HOST', 'CIFAR10', 'GQA', 'CUTE80', 'NoCaps', 'OKVQA', 'STVQA', 
                'MSCOCO_OC', 'WHOOPSVQA', 'MSCOCO_caption_karpathy', 'MSCOCO_pope_random', 
                'MSCOCO_pope_adversarial', 'AOKVQAClose', 'VSR', 'MSCOCO_MCI', 'HatefulMemes', 'MSCOCO_pope_popular', 
                'FUNSD', 'WordArt', 'IC15', 'OxfordIIITPet', 'ImageNet', 'VizWiz', 'COCO-Text', 'TextVQA', 'SVTP', 
                'Total-Text', 'WOST', 'ScienceQAIMG', 'ImageNetVC_others', 'ImageNetVC_material', 'ImageNetVC_shape', 'ImageNetVC_component', 'ImageNetVC_color', 'RSVQALR_MCI', 'COD10K', 'ImageNetC', 'ImageNetC_blur','ImageNetC_digital', 'ImageNetC_noise', 'ImageNetC_weather', 'ImageNetC_extra']
save_dir = '/path_to/save_dir'
save_root_dir = os.path.join(save_root, save_dir)

API_KEY = "Input your api-key"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

messages_template = messages1


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--begin_model', default=0, type=int,
                        help='From which model gpt begin to work')   
    parser.add_argument('--begin_dataset', default=0, type=int,
                        help='From which dataset gpt begin to work')  
    parser.add_argument('--begin_index', default=0, type=int,
                        help='From which number gpt begin to work')
    return parser
args = get_args_parser()
args = args.parse_args()

def generate_chat_completion(messages, model="gpt-3.5-turbo", temperature=0.5, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")



for j in range(args.begin_model, len(model_pool)):
    model_now = model_pool[j]
    model_path_now = os.path.join(root_path, model_now)
    for k in range(args.begin_dataset, len(dataset_pool)):
        dataset_now = dataset_pool[k]
        path_now = os.path.join(model_path_now, dataset_now + '.json')
        save_path = os.path.join(save_root_dir, model_now, dataset_now)
        if os.path.exists(path_now):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(path_now, 'r') as f:
                data = json.load(f)
            data_new = copy.deepcopy(data)
            for i in range(args.begin_index, len(data)):
                messages = copy.deepcopy(messages_template)                    
                data_now = data[i]
                question = data_now['question']
                student_answer = data_now['answer']
                gt_answer = data_now['gt_answers']
                question_id = data_now['question_id']
                save_json_path = os.path.join(save_path, model_now + '_' + dataset_now + '_' + question_id + '.json')
                input_info = "The Question is: " + question + "\n" + "The Correct Answer is: " + gt_answer + "\n" + "The Student's Answer is: " + student_answer
                messages[1]['content'] = messages[1]['content'] + input_info
                # Get the response text
                finish_flag = False
                while not finish_flag:
                    try:
                        response_text = generate_chat_completion(messages)
                        finish_flag = True
                    except:
                        print('Error, try again')
                        time.sleep(10)

                data_new[i]['chatgpt_judgement'] = response_text
                ff = json.dumps(data_new[i])
                with open(save_json_path, 'w') as f:
                    json.dump(data_new[i], f, indent=4)
                print(model_now, dataset_now, i)