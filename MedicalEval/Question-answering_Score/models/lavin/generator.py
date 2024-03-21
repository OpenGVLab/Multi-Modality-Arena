# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from .tokenizer import Tokenizer
from .eval_model import Transformer
from  torch.cuda.amp import autocast

class LaVIN_Generator:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # self.backbone = clip.load('ViT-B/16', device='cpu')[0]

    def insert_image_embeds(self,examples,image_embeds,prefix_img,prefix_nonimg,img_indicators):
        _bsz, seqlen,_ = examples.shape
        new_examples=[]
        for i, example in enumerate(examples):
            if img_indicators[i]>0.:
                new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
                new_example = new_example[:seqlen]
            else:
                new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
                new_example = new_example[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        return new_examples

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        images: torch.Tensor,
        indicators: List[int],
        max_gen_len: int,
        n_feats: int=3,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        prefix_img_token = self.tokenizer.encode("Image: ", bos=True, eos=False)
        non_prefix_img_token= self.tokenizer.encode("Image: N/A", bos=True, eos=False)

        images=images.cuda()
        self.model.backbone.cuda()

        image_embeds = self.model.backbone.encode_image(images)
        image_embeds = self.model.adapter_proj(image_embeds)


        prompt_tokens=[]
        for i,x in enumerate(prompts):
            if indicators[i]==1:
                token_idx=prefix_img_token+[0]*n_feats+self.tokenizer.encode(x, bos=False, eos=False)
            else:
                token_idx = non_prefix_img_token + self.tokenizer.encode(x, bos=False, eos=False)
            prompt_tokens.append(token_idx)


        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        input_text_mask=torch.zeros_like(tokens).bool()

        for k, t in enumerate(prompt_tokens):
            t=t[:total_len]
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k,:len(t)]=True

        token_embeds=self.model.tok_embeddings(tokens)
        indicators=torch.Tensor(indicators).cuda().long()
        modality_embedding=self.model.adapter_modality_embedding(indicators).unsqueeze(1)

        for i in range(len(token_embeds)):
            if indicators[i]==1:
                pos=len(prefix_img_token)
                #insert image emebedding into the sequence
                image_token_embed=torch.cat([token_embeds[i,:pos],image_embeds[i],token_embeds[i,pos+n_feats:]],0)
                token_embeds[i]=image_token_embed



        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):

            if prev_pos==0:
                h=torch.cat([modality_embedding,token_embeds[:,prev_pos:cur_pos]],1)
            else:
                h=token_embeds[:,prev_pos:cur_pos]
            logits = self.model.forward(h, prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated

            next_token_embeds = torch.where(
                input_text_mask[:, cur_pos,None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token)
            )
            token_embeds[:,cur_pos]=next_token_embeds

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))


        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
