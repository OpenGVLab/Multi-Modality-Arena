from tqdm import tqdm
import os
import yaml
import pdb
import argparse
import re
import pdb

def parse():
    parser = argparse.ArgumentParser(description='IdealGPT in test datasets.')
    parser.add_argument('--result', type=str, default='/home/haoxuan/code/SubGPT/exp_result/vcr_04262356/vcr_val/result', 
                        help='VCR predicted result path')
    parser.add_argument('--print_multiround', action='store_true', 
                        help='Whether print the annoid of multiple rounds and correct samples.')
    parser.add_argument('--print_outlier', action='store_true', 
                        help='Whether print the annoid of multiple rounds and correct samples.')
    args = parser.parse_args()
    return args


args = parse()
blip_gpt_folder = args.result

answer_map = ['1', '2', '3', '4']
total_correct_count = 0
total_unknown_count = 0
total_count = 0
total_round = 0
total_tokens = 0
total_outlier_count = 0
count_round = False

for i in tqdm(os.listdir(blip_gpt_folder), desc='Analyzing BLIP_GPT result ({})'.format(blip_gpt_folder)):
    cur_dir = os.path.join(blip_gpt_folder, i)
    with open(cur_dir, "r") as file:
        data_i = yaml.safe_load(file)

    if 'used_round' in data_i['result']:
        count_round = True
        total_round += data_i['result']['used_round']
        total_tokens += data_i['result']['total_tokens']

    if data_i['result']['predict_answer'] is None:
        total_unknown_count += 1
        if args.print_outlier:
            print('Samples with unknown answer prediction:{}'.format(data_i['setting']['id']))
    else:
        if str(data_i['result']['predict_answer']).strip() == str(data_i['setting']['answer_label']).strip():
            total_correct_count += 1
            if args.print_multiround and 'used_round' in data_i['result']:
                if data_i['result']['used_round'] > 1:
                    print('Samples with multiple rounds and correct:{}'.format(data_i['setting']['id']))
        elif str(data_i['result']['predict_answer']).strip() not in answer_map:
            total_outlier_count += 1
            if args.print_outlier:
                print('Samples with outlier answer prediction:{}'.format(data_i['setting']['id']))

    total_count += 1

print('Accuracy of {} samples: {}%'.format(total_count, (total_correct_count*1.00*100/total_count)))
print('Ratio of unknown samples: {}%'.format(total_unknown_count*1.00*100/total_count))
print('Ratio of outlier samples: {}%'.format(total_outlier_count*1.00*100/total_count))
if count_round:
    print('Average rounds used: {}'.format(total_round*1.00/total_count))
    print('Average tokens used: {}'.format(total_tokens*1.00/total_count))
