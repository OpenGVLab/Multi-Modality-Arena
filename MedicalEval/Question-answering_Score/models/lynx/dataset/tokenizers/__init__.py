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

DEFAULT_PAD_TOKEN = "[PAD]"
TOKEN_NONE_FLAG = "[NONE]"


def build_tokenizer(LLM: str, use_left_pad: bool):
    # rpath = os.path.join("data/", LLM)
    rpath = LLM
    assert os.path.exists(rpath)

    num_new_tokens = 0

    if 'vicuna-7b' in LLM:
        from ...models.llms.llama.tokenization_llama import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(rpath)

        if tokenizer.pad_token is None:
            num_new_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": DEFAULT_PAD_TOKEN,
                }
            )

    else:
        raise NotImplementedError("Tokenizer for LLM: ", LLM)

    print("-" * 40)
    print("### Vocab Size: ", len(tokenizer), flush=True)

    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None

    if tokenizer.bos_token is None:
        print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
        tokenizer.bos_token = TOKEN_NONE_FLAG

    else:
        print("bos_token, ", tokenizer.bos_token)
        print("bos_token_id, ", tokenizer.bos_token_id)

    if use_left_pad:
        tokenizer.padding_side = "left"

    print("Left Pad: ", use_left_pad, flush=True)

    print("eos_token, ", tokenizer.eos_token)
    print("eos_token_id, ", tokenizer.eos_token_id)

    print("pad_token, ", tokenizer.pad_token)
    print("pad_token_id, ", tokenizer.pad_token_id)

    print("unk_token, ", tokenizer.unk_token, flush=True)
    print("unk_token_id, ", tokenizer.unk_token_id, flush=True)
    print("-" * 40)

    return tokenizer, num_new_tokens
