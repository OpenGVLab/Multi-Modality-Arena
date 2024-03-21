

from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig
)
import torch

from model.blip2.modeling_blip_2 import Blip2ForConditionalGeneration
from model.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration,
    "instructblip": InstructBlipForConditionalGeneration,

}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )


    for param in model.parameters():
        param.requires_grad = False

    for param in model.language_projection.parameters():
        param.requires_grad = True

    if model_args.backbone_model == 'flan-t5':
        for block in model.language_model.encoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True

        for block in model.language_model.decoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True
            block.layer[1].EncDecAttention.q.requires_grad=True
            block.layer[1].EncDecAttention.v.requires_grad=True
    else:# vicuna
        print(f"vicuna layer:{len(model.language_model.model.layers)}")
        for block in model.language_model.model.layers:
            block.self_attn.q_proj.weight.requires_grad=True
            block.self_attn.v_proj.weight.requires_grad=True
    
    all_param = 0
    trained_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad ==True:
            trained_param+=param.numel()
    total_param = all_param 

    print('***** total param is {} *****'.format(total_param))
    print('***** total trained param is {} *****'.format(trained_param))
    return model
