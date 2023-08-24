import os, glob, json, time, copy
import numpy as np
from collections import Counter


dir2prompt = {  
    'p1_dir_name': 'prompt1',
    'p2_dir_name' : 'prompt2',
    'p3_dir_name' : 'prompt3',
    'p4_dir_name' : 'prompt4',
    'p5_dir_name' : 'prompt5',
}

prompt2key = {
    'prompt1': 'Most Likely Score',
    'prompt2': 'Final Score',
    'prompt3': 'Final Assessment Score',
    'prompt4': 'Final Score',
    'prompt5': 'Most Likely Score',
}

round_pool = [
    'p1_dir_name', 
    'p2_dir_name',
    'p3_dir_name',
    'p4_dir_name',
    'p5_dir_name',
    ]

root_path = '/path_to/evaluation_results'
base_pure_path = '/path_to/original_item'
model_pool = ['LLaMA-Adapter-v2', 'VPGTrans', 'MiniGPT-4', 'BLIP2', 'InstructBLIP', 'Otter', 'Otter-Image', 'LLaVA', 'OFv2_4BI', 'mPLUG-Owl', 'PandaGPT', 'Bard']
dataset_pool = ['VCR1_MCI', 'CTW', 'IIIT5K', 'CIFAR100', 'IC13', 'VCR1_OC', 'WHOOPSWeird', 'OCRVQA', 
                'Flickr', 'WHOOPSCaption', 'IconQA', 'POIE', 'VQAv2', 'AOKVQAOpen', 'SVT', 'SROIE', 
                'DocVQA', 'Flowers102', 'HOST', 'CIFAR10', 'GQA', 'CUTE80', 'NoCaps', 'OKVQA', 'STVQA', 
                'MSCOCO_OC', 'WHOOPSVQA', 'MSCOCO_caption_karpathy', 'MSCOCO_pope_random', 
                'MSCOCO_pope_adversarial', 'AOKVQAClose', 'VSR', 'MSCOCO_MCI', 'HatefulMemes', 'MSCOCO_pope_popular', 
                'FUNSD', 'WordArt', 'IC15', 'OxfordIIITPet', 'ImageNet', 'VizWiz', 'COCO-Text', 'TextVQA', 'SVTP', 
                'Total-Text', 'WOST', 'ScienceQAIMG', 'ImageNetVC_others', 'ImageNetVC_material', 'ImageNetVC_shape', 'ImageNetVC_component', 'ImageNetVC_color', 'RSVQALR_MCI', 'COD10K', 'ImageNetC', 'ImageNetC_blur','ImageNetC_digital', 'ImageNetC_noise', 'ImageNetC_weather', 'ImageNetC_extra']
save_root_path = '/path_to/ensemble_accuracy'
save_root_path1 = '/path_to/ensemble_accuracy_for_each_item'
if not os.path.exists(save_root_path):
    os.makedirs(save_root_path)
for model_now in model_pool:
    save_json_path = os.path.join(save_root_path, model_now + '.json')
    output_content = []
    print(model_now)
    for dataset_now in dataset_pool:
        base_json_path = os.path.join(base_pure_path, model_now, dataset_now + '.json')
        if os.path.exists(base_json_path):
            save_items_path = os.path.join(save_root_path1, model_now)
            if not os.path.exists(save_items_path):
                os.makedirs(save_items_path)
            with open(base_json_path, 'r') as f:
                base_json = json.load(f)
            new_json_data = copy.deepcopy(base_json)
            length_now = len(base_json)
            gpt_judge_paths = []
            all_items_final_scores = []
            for i in range(len(round_pool)):
                gpt_judge_paths.append(os.path.join(root_path, round_pool[i], model_now, dataset_now))
            for i in range(length_now):
                question_id = base_json[i]['question_id']
                gpt_scores = []
                for j in range(len(round_pool)):
                    gpt_json_path = os.path.join(gpt_judge_paths[j], model_now + '_' + dataset_now + '_' + str(question_id).zfill(6) + '.json')
                    round_now = round_pool[j]
                    prompt_now = dir2prompt[round_now]
                    key_now = prompt2key[prompt_now]
                    with open(gpt_json_path, 'r') as f:
                        gpt_json = json.load(f)
                        gpt_statement = gpt_json['chatgpt_judgement']
                        if key_now not in gpt_statement:
                            if key_now.lower() in gpt_statement:
                                gpt_statement = gpt_statement.replace(key_now.lower(), key_now) 
                        if key_now not in gpt_statement:
                            gpt_score = -1
                            gpt_statement_condition = False
                        else:
                            gpt_statement_condition = True
                        if gpt_statement_condition:
                            gpt_score_statement = gpt_statement.split(key_now)[1]
                            if '0' in gpt_score_statement:
                                gpt_score = 0
                            elif '1' in gpt_score_statement:
                                gpt_score = 1
                            else:
                                gpt_score = -1
                        gpt_scores.append(gpt_score)
                gpt_scores_counter = Counter(gpt_scores)
                gpt_score_ensemble = gpt_scores_counter.most_common(1)[0][0]
                all_items_final_scores.append(gpt_score_ensemble)
                new_json_data[i]['gpt_scores'] = gpt_scores
                new_json_data[i]['gpt_score_ensemble'] = gpt_score_ensemble
            accuracy = all_items_final_scores.count(1) / len(all_items_final_scores)
            bug_items_num = all_items_final_scores.count(-1)
            accuracy_without_bug = all_items_final_scores.count(1) / (len(all_items_final_scores) - bug_items_num)
            output_content.append('The ensemble accuracy of {} on {} is {}'.format(model_now, dataset_now, accuracy))
            output_content.append('The ensemble accuracy of {} on {} exclude the bug item is {}'.format(model_now, dataset_now, accuracy_without_bug))
            output_content.append('There are {} bug items in {} on {}'.format(bug_items_num, model_now, dataset_now))
            with open(save_json_path, 'w') as f:
                json.dump(output_content, f, indent=4)

    ###############save for each item################
            save_json_items_path = os.path.join(save_items_path, dataset_now + '.json')
            with open(save_json_items_path, 'w') as f:
                json.dump(new_json_data, f, indent=4)
    #################################################