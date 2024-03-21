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

import json
import numpy as np


def sample_frame_ids(num_frames: int, num_segments: int, training: bool):
    nf = num_frames
    ns = num_segments
    out_indices = np.zeros(ns)
    if nf <= ns:
        if training:
            out_indices = np.concatenate((np.arange(nf), np.random.randint(nf, size=ns - nf)), axis=0)
        else:
            out_indices = np.array([(idx % nf) for idx in range(ns)])
        out_indices = np.sort(np.array(out_indices))
    else:
        stride = nf // ns  # at least 1
        strides = np.array([stride] * ns)
        offsets = np.array([1] * (nf - ns * stride) + [0] * (ns * (stride + 1) - nf))
        if training:
            np.random.shuffle(offsets)
        strides += offsets
        cursor = 0
        for idx, each_stride in enumerate(strides):
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if training:
                out_indices[idx] = np.random.randint(left, right)
            else:
                out_indices[idx] = left

    return [int(i) for i in out_indices]

def write_jsonl(result: list, wpath: str):
    if wpath.startswith('hdfs'):
        with hopen(wpath, 'w') as f:
            for res in result:
                to_write = json.dumps(res, ensure_ascii=False) + '\n'
                f.write(to_write.encode())
    else:
        with open(wpath, 'wt') as f:
            for res in result:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')


def read_jsonl(rpath: str):
    result = []
    if rpath.startswith('hdfs'):
        with hopen(rpath, 'r') as f:
            for line in f:
                result.append(json.loads(line.decode().strip()))
    else:
        with open(rpath, 'rt') as f:
            for line in f:
                result.append(json.loads(line.strip()))

    return result
