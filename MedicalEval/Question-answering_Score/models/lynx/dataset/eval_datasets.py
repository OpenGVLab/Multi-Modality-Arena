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

import os
import io
from base64 import b64decode
import json

import numpy as np

import torch

from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import sample_frame_ids

from dataset import get_image_transform, get_video_transform
from dataset.tokenizers import build_tokenizer


class LynxEvalDatasetBase(Dataset):
    def __init__(self, config):

        self.VISION_PROMPT_DICT = config.get("vision_prompt_dict", "image").strip()
        self.OUTPUT_PROMPT_DICT = config.get("output_prompt_dict", "answer").strip()

        self.prompt = config.get("prompt", "").strip()

        if len(self.prompt):
            print("### Use prompt: ", self.prompt, flush=True)

        self.data = self.read_ann_file(config['test_files'])
        print("### num_test: ", len(self.data), flush=True)

        self.image_rdir = config['image_rdir']

        self.use_left_pad = config['use_left_pad']
        self.tokenizer, _ = build_tokenizer(config['LLM'], use_left_pad=self.use_left_pad)
        self.max_input_tokens = config['max_input_tokens']

        self.lower_text = config['lower_text']

        _, self.img_transform = get_image_transform(config)

        self.num_frames = config['data']['num_frames']
        print("### num_frames: ", self.num_frames, flush=True)

        _, self.video_transform = get_video_transform(config)

    def read_ann_file(self, ann_file):
        if isinstance(ann_file, str):
            ann_file = [ann_file]
        else:
            assert isinstance(ann_file, list)

        data = []
        for rpath in ann_file:
            assert os.path.exists(rpath)

            if rpath.endswith(".jsonl"):
                with open(rpath, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise ValueError("type of rpath not json/jsonl: ", rpath)

        return data

    def __len__(self):
        return len(self.data)

    def _get_frames(self, vision_inp):
        assert isinstance(vision_inp, list) or isinstance(vision_inp, np.ndarray)
        assert isinstance(vision_inp[0], str)

        # sample frames
        assert self.num_frames > 0
        ids = sample_frame_ids(len(vision_inp), self.num_frames, training=True)

        frames = []
        for i in ids:
            img = Image.open(io.BytesIO(b64decode(vision_inp[i]))).convert("RGB")
            frames.append(self.video_transform(img))

        return frames

    def load_vision_inp(self, vision_inp):
        if vision_inp is None:
            return None

        elif isinstance(vision_inp, list) or isinstance(vision_inp, np.ndarray):
            return self._get_frames(vision_inp)

        elif isinstance(vision_inp, str):

            if os.path.exists(vision_inp):
                if vision_inp.endswith('.json'):
                    with open(vision_inp, 'r') as f:
                        data = json.load(f)

                        if isinstance(data, list) or isinstance(data, np.ndarray):
                            return self._get_frames(data)
                        else:
                            image = Image.open(io.BytesIO(b64decode(data))).convert("RGB")

                else:
                    image = Image.open(vision_inp).convert('RGB')

            else:  # base64 encoding
                try:
                    image = Image.open(io.BytesIO(b64decode(vision_inp))).convert("RGB")
                except Exception as e:
                    raise ValueError(f"check whether it is a rpath (and not exist)?: {vision_inp} {e}")

        else:
            raise ValueError(f"vision_inp type of {type(vision_inp)}")

        image = self.img_transform(image)

        return image

    def get_vision_input(self, ann):
        assert isinstance(ann, dict)

        vision_key = self.VISION_PROMPT_DICT.strip()
        vision_inp = ann[vision_key]

        return self.load_vision_inp(vision_inp)

    def get_text_input(self, ann):
        assert isinstance(ann, dict)

        text_input = self.prompt.format_map(ann).strip()

        return text_input.lower() if self.lower_text else text_input

    def __getitem__(self, index):
        ann = self.data[index]
        assert isinstance(ann, dict)

        idx = ann['index']

        vision_input = self.get_vision_input(ann)
        text_input = self.get_text_input(ann)

        input_ids = [self.tokenizer.bos_token] + self.tokenizer.tokenize(text_input)
        if self.max_input_tokens > 0:
            input_ids = input_ids[-self.max_input_tokens:]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)

        return idx, vision_input, input_ids

    def pad_batch_data(self, input_ids):
        max_length = max([len(ids) for ids in input_ids])

        # pad + get atts
        input_ids_pad = []
        input_atts_pad = []

        for ids in input_ids:
            n_tokens = len(ids)
            n_pads = max_length - n_tokens

            if self.use_left_pad:  # recommend
                input_ids_pad.append([self.tokenizer.pad_token_id] * n_pads + ids)
                input_atts_pad.append([0] * n_pads + [1] * n_tokens)

            else:  # right pad
                input_ids_pad.append(ids + [self.tokenizer.pad_token_id] * n_pads)
                input_atts_pad.append([1] * n_tokens + [0] * n_pads)

        input_ids_pad = torch.LongTensor(input_ids_pad)
        input_atts_pad = torch.LongTensor(input_atts_pad)

        return input_ids_pad, input_atts_pad

    def collate_fn(self, batch):
        inputs = []
        for x in zip(*batch):
            inputs.append(x)

        idx, vision_input, input_ids = inputs

        if isinstance(vision_input[0], list):  # video
            batch_size = len(vision_input)
            vision_input = torch.stack(sum(vision_input, []))  # flatten
            _, c, h, w = vision_input.shape

            if self.num_frames > 1:
                vision_input = vision_input.reshape([batch_size, self.num_frames, c, h, w])

        else:  # image
            vision_input = torch.stack(vision_input, dim=0)

        input_ids_pad, input_atts_pad = self.pad_batch_data(input_ids)

        return idx, vision_input, input_ids_pad, input_atts_pad
