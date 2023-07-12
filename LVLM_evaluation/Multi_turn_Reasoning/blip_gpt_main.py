import os
import yaml
import argparse
import torch

import openai
from tqdm import tqdm
import pdb

from data import VCRSampler
from data import VESampler

from chat import VCRConversationTwoAgent
from chat import VEConversationTwoAgent
import random



def IdealGPT(vqa_model, dataset, data_ids, model, save_path='', max_n_rounds=5, print_mode='no', prompt_setting='v1a', temp_gpt=0.0):
    """
    Conduct IdealGPT conversation

    Args:
        vqa_model : vqa model.
        dataset: the dataset used to caption
        data_ids (list): a list of sample ids in the dataset
        model (str or Blip2): the model name used to ask quetion. Valid values are 'gpt3', 'chatgpt', and their concrete model names 
                    including 'text-davinci-003', 'davinci,' and 'gpt-3.5-turbo'.
                    If passing a Blip2 instance, will use its backend LLM.
        save_path (str): the path to save caption results. If it is empty, results are not being saved.
        max_n_rounds (int): the max number of chat rounds
        n_blip2_context (int): how many previous QA rounds can blip2 see. negative value means blip2 can see all 
        print_mode (str): print mode. 'chat' for printing everying. 'bar' for printing everthing but the chat process. 'no' for no printing
    """
    if model == 'chatgpt':
        model = 'gpt-3.5-turbo'
    elif model =='gpt4':
        model = 'gpt-4'

    all_predict_answer = []
    all_answer_label = []
    all_round_number = 0
    for data_id in tqdm(data_ids, disable=print_mode!='no'):
        result_path = os.path.join(save_path, 'result', '{}.yaml'.format(data_id))
        # Skip if the result file exist.
        if os.path.isfile(result_path):
            continue
        if print_mode != 'no':
            print('Data ID {}'.format(data_id))

        if type(dataset) == VCRSampler:
            image_path, qa = dataset.fetch_data(data_id)
            info = {'setting':
                        {
                        'id': data_id,
                        'question_id': qa['question_id'] if 'question_id' in qa else None,
                        'question': qa['question'].strip(),
                        'answer_choices':[answer_i.strip() for answer_i in qa['answer_choices']] if 'answer_choices' in qa else None,
                        'answer_label': str(qa['answer_label']) if 'answer_label' in qa else None,
                        'max_n_rounds': max_n_rounds,
                        'img_path': qa['img_path'] if 'img_path' in qa else None
                        }
                }
            if 'caption' in qa:
                caption = qa['caption']
            else:
                caption = None
        elif type(dataset) == VESampler:
            image_path, ve_info = dataset.fetch_data(data_id)
            info = {'setting':
                        {
                        'id': data_id,
                        'hypothesis': ve_info['hypothesis'].strip(),
                        'answer_label': str(ve_info['answer_label']) if 'answer_label' in ve_info else None,
                        'max_n_rounds': max_n_rounds,
                        'img_path': ve_info['img_path'] if 'img_path' in ve_info else None
                        }
                }
            if 'caption' in ve_info:
                caption = ve_info['caption']
            else:
                caption = None
        results = {}
        # Initialize VQA Instance.
        if type(dataset) == VCRSampler:
            chat = VCRConversationTwoAgent(img=image_path,
                                vqa_model=vqa_model,
                                model=model,
                                question=info['setting']['question'],
                                answer_choices=info['setting']['answer_choices'],
                                prompt_setting=prompt_setting,
                                caption=caption,
                                temp_gpt=temp_gpt,
                                data_id=data_id,)
        elif type(dataset) == VESampler:
            chat = VEConversationTwoAgent(img=image_path,
                                vqa_model=vqa_model,
                                model=model,
                                question=info['setting']['hypothesis'],
                                answer_choices=['entailment', 'neutral', 'contradiction'],
                                prompt_setting=prompt_setting,
                                caption=caption,
                                temp_gpt=temp_gpt,
                                data_id=data_id)


        used_round = chat.chatting(max_n_rounds, print_mode=print_mode)
        results['predict_answer'] = chat.answer_predict
        results['sub_questions'] = chat.sub_questions
        results['sub_answers'] = chat.sub_answers
        results['chat_history'] = chat.chat_history
        results['total_tokens'] = chat.total_tokens
        results['caption'] = chat.catpion
        results['used_round'] = used_round

        info['result'] = results

        all_predict_answer.append(chat.answer_predict)
        all_answer_label.append(str(info['setting']['answer_label']))
        all_round_number += results['used_round']
        
        if save_path:
            with open(result_path, 'w') as f:
                yaml.dump(info, f)

    # Evaluation:
    if type(dataset) == VCRSampler or type(dataset) == VESampler:
        # Evaluate VCR and SNLI-VE by acc.
        total_correct = 0
        total_exceed_round = 0
        for predict_i, gt_i in zip(all_predict_answer, all_answer_label):
            if predict_i == gt_i:
                total_correct += 1
            if predict_i is None:
                total_exceed_round += 1
        acc = (total_correct*1.0) / len(data_ids)
        print('Acc:{}%'.format(acc*100))
        print('Average number of rounds:{}'.format(all_round_number*1.0/len(data_ids)))
        exceed_round_ratio = (total_exceed_round*1.0) / len(data_ids)
        print('Unknown Ratio:{}%'.format(exceed_round_ratio*100))


def parse():
    parser = argparse.ArgumentParser(description='IdealGPT Args.')
    parser.add_argument('--data_root', type=str, default='/home/haoxuan/data/vcr1/', 
                        help='root path to the dataset')
    parser.add_argument('--save_root', type=str, default='./exp_result/', 
                        help='root path for saving results')
    parser.add_argument("--data_subset", type=str, default=None, help="specify the subset of the dataset.")
    parser.add_argument('--data_partition', type=str, default=None,
                        help='range of data used, in the format of numberA_numberB, A<=B')
    parser.add_argument('--exp_tag', type=str, required=True, 
                        help='tag for this experiment. caption results will be saved in save_root/exp_tag')
    parser.add_argument('--dataset', type=str, default='vcr_val',
                        help='Names of the dataset to use in the experiment. Valid datasets include vcr_val, ve_dev. Default is vcr_val')
    parser.add_argument('--max_n_rounds', type=int, default=4,
                        help='Nax Number of QA rounds between GPT and BLIP-2. Default is 4.')
    parser.add_argument('--model', type=str, default='chatgpt', choices=['chatgpt', 'gpt4'],
                        help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
    parser.add_argument('--vqa_model', type=str, default='blip2_t5_xxl', choices=['blip2_t5_xxl', 'blip2_t5_xl',  'blip2_opt_6.7b', 'blip2_opt_2.7b', 'llava', 'minigpt4','otter','vgtrans','mplugowl','instructblip','llama_adapter'],
                        help='model as Answerer.')
    parser.add_argument('--device_id', type=int, default=0, 
                        help='Which GPU to use.')
    parser.add_argument('--prompt_setting', type=str,  default='v1a', 
                        help='Prompt Setting Version')
    parser.add_argument('--openai_key', type=str,  default='', 
                        help='OpenAI Key for GPT-3.5/4 API')
    parser.add_argument('--caption_path', type=str,  default=None, 
                        help='Caption path for images')
    parser.add_argument('--temp_gpt', type=float,  default=0.0, 
                        help='Temperature for GPT')
    parser.add_argument('--temp_vqa', type=float,  default=0.001, 
                        help='Temperature for VQA model (LLaVA and MiniGPT4), must be positive')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    args = parser.parse_args()
    return args
    
    
def main(args):
    # Set OpenAI
    OPENAI_API_KEY = args.openai_key
    openai.api_key = OPENAI_API_KEY

    random.seed(args.seed)

    # load the dataset
    if 'vcr' in args.dataset:
        dataset = VCRSampler(dataset_root=args.data_root, 
                             dataset_name=args.dataset, 
                             data_subset=args.data_subset, 
                             data_partition=args.data_partition, 
                             caption_path=args.caption_path)
    elif 've' in args.dataset:
        dataset = VESampler(dataset_root=args.data_root,
                               dataset_name=args.dataset, 
                               data_subset=args.data_subset,
                               data_partition=args.data_partition, 
                               caption_path=args.caption_path)
    print('Finish loading data')

    print('Start loading VQA model')
    if 'blip2' in args.vqa_model:
        from lib.blip2_lib import Blip2Lavis
        if 't5' in args.vqa_model and '_xl' in args.vqa_model:
            vqa_model = Blip2Lavis(name="blip2_t5", model_type="pretrain_flant5xl", device=torch.device("cuda:{}".format(args.device_id)))

        elif 't5' in args.vqa_model and '_xxl' in args.vqa_model:
            vqa_model = Blip2Lavis(name="blip2_t5", model_type="pretrain_flant5xxl", device=torch.device("cuda:{}".format(args.device_id)))

        elif 'opt' in args.vqa_model and '6.7b' in args.vqa_model:
            vqa_model = Blip2Lavis(name="blip2_opt", model_type="pretrain_opt6.7b", device=torch.device("cuda:{}".format(args.device_id)))

        elif 'opt' in args.vqa_model and '2.7b' in args.vqa_model:
            vqa_model = Blip2Lavis(name="blip2_opt", model_type="pretrain_opt2.7b", device=torch.device("cuda:{}".format(args.device_id)))
        else:
            raise NotImplemented(f'{args.vqa_model} not supported')
    elif 'llava' in args.vqa_model:
        from lib.llava_lib import LLaVA
        vqa_model = LLaVA(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))
    elif 'minigpt4' in args.vqa_model:
        from lib.minigpt4_lib import MiniGPT4
        vqa_model = MiniGPT4(model_type="minigpt4", device=torch.device("cuda:{}".format(args.device_id)))
    elif 'llama_adapter' in args.vqa_model:
        from lib.LLama_lib import LLama
        vqa_model = LLama(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))
    elif 'otter' in args.vqa_model:
        from lib.otter_lib import Otter
        vqa_model = Otter(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))           
    elif 'instructblip' in args.vqa_model:
        from lib.instructblip_lib import InstructBLIP
        vqa_model = InstructBLIP(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))         
    elif 'mplugowl' in args.vqa_model:
        from lib.mplugowl_lib import MplugOwl
        vqa_model = MplugOwl(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))
    elif 'vgtrans' in args.vqa_model:
        from lib.vgtrans_lib import VPGTrans
        vqa_model = VPGTrans(model_type="llava", device=torch.device("cuda:{}".format(args.device_id)))        
    print('Finish loading VQA model {}'.format(args.vqa_model))

    question_model = args.model

    # preparing the folder to save results
    save_path = os.path.join(args.save_root, f'{args.dataset}_{args.exp_tag}')
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'result'))
    with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # start Conversation
    IdealGPT(vqa_model,
                dataset, 
                dataset.ids, 
                save_path=save_path, 
                max_n_rounds=args.max_n_rounds, 
                model=question_model,
                print_mode='no',
                prompt_setting=args.prompt_setting,
                temp_gpt=args.temp_gpt)
    

if __name__ == '__main__':
    args = parse()
    main(args)