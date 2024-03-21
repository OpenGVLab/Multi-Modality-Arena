import os
import json
import pandas
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax
import requests
from lavis.models import load_model_and_preprocess
import torch
import pdb
from types import MethodType
from PIL import Image
from io import BytesIO

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def forward_lm(self, samples):
    image = samples["image"]
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
    # print(samples["text_output"])
    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(
            samples["text_input"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            samples["text_output"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}




def load_candidates_medical(data):
    a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return answer_list


def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]

def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img

 
 
@torch.no_grad()
def test(model, vis_processors, dataset=None, model_type='blip2', prompt_idx=4, save_path=''):
    data_all = json.load(open(dataset))
    cnt = 0
    correct = 0
    
    res = []
    for data in data_all:
        cnt += 1
        question = data['question']
        candidates = load_candidates_medical(data)
        answer = data['gt_answer']
        img_path = data['image_path']  
        
        prefix = load_prompt(question, prompt_idx)
        prefix_tokens = model.t5_tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)
        start_loc = prefix_tokens.input_ids.size(1)
        
        candidate_scores = []  # pred scores of candidates
        raw_image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        
        for candidate in candidates:
            
            prompt = prefix + " {}.".format(candidate)
            outputs = model.forward_lm({"image": image, "text_input": prefix, "text_output": candidate, "start_loc": start_loc})
            loss = outputs["loss"]
            candidate_scores.append(loss.item())
        data['confidence'] =  str(candidate_scores)
        candidate_scores = softmax(np.reciprocal(candidate_scores))
        pred = candidates[np.argmax(candidate_scores)]
        print(candidates, candidate_scores)
        data['model_pred'] = pred
        
        data['is_correct'] = 'yes' if pred == answer else 'no'
        if pred == answer:
            correct += 1
        res.append(data)
        
    acc = correct / cnt
    print("Accuracy: ", acc)
        
    final_res = {'model_name': model_type, 'dataset_name': dataset, 'correct_precentage': acc, 'pred_dict': res}
    
    
    with open('{}/{}.json'.format(save_path, dataset.replace('/', '_')), 'w') as f:
        json.dump(final_res, f, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--dataset_path", type=str, default='/path/to/datset')
    parser.add_argument("--answer_path", type=str, default="output_res")
    args = parser.parse_args()
    return args

def run(args):
    model_type = 'blip2'
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    model.forward_lm = MethodType(forward_lm, model)

    answer_path = f'{args.answer_path}/{model_type}'
    os.makedirs(answer_path, exist_ok=True)
    
    sub_dataset = args.dataset_path
    test(model, vis_processors, dataset=sub_dataset, model_type=model_type, prompt_idx=4, save_path=answer_path)


if __name__ == "__main__":
    args = parse_args()
    run(args)

