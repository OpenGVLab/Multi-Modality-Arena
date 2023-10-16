# Copyright (2023) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json

import torch
import torch.backends.cudnn as cudnn

import utils
from utils import write_jsonl
from dataset import create_dataset, create_loader


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    result = []

    for n, (idx, vision_input, input_ids, input_atts) in enumerate(data_loader):
        vision_input = vision_input.to(device, non_blocking=True)
        input_ids = input_ids.to(device)
        input_atts = input_atts.to(device)

        text_outputs = model.generate(
            vision_input=vision_input,
            input_ids=input_ids, input_atts=input_atts,
            use_nucleus_sampling=config.get('use_nucleus_sampling', False),
            apply_lemmatizer=config['apply_lemmatizer'],
            num_beams=config['num_beams'],
            min_length=config['min_length'],
            length_penalty=config.get('length_penalty', 1.0),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
            top_p=config.get('top_p', 0.9),
            top_k=config.get('top_k', 3),
            max_new_tokens=config.get('max_new_tokens', 64))

        for i, output in zip(idx, text_outputs):
            result.append({"index": i, "text_output": output.strip()})

    return result


def main(args, config):
    print("### Evaluating", flush=True)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("config:", json.dumps(config), flush=True)
    print("output_path, ", args.output_path, flush=True)

    print("### Creating model", flush=True)
    from models.lynx import LynxBase
    model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)

    model = model.to(device)

    for _, param in model.named_parameters():
        param.requires_grad = False

    model.eval()

    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("### Creating datasets", flush=True)
    test_dataset = create_dataset('eval', config)

    start_time = time.time()
    print("### Start evaluating", flush=True)

    test_loader = create_loader([test_dataset],
                                batch_size=[config['batch_size_test']],
                                num_workers=[4],
                                collate_fns=[test_dataset.collate_fn])[0]

    predictions = evaluation(model, test_loader, device, config)

    write_jsonl(predictions, args.output_path)
    print("### Prediction Results Save To: ", args.output_path, flush=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True, help="path of outputfile")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)
