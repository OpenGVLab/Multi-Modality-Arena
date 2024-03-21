
import numpy as np
from einops import rearrange
from typing import Tuple, Union,Optional
from dataclasses import dataclass, field

import torch
import transformers
import torch.nn.functional as F
import torchvision.models as models
# from torchvision.models.models import resnet50, ResNet50_Weights

from torch import nn
# from torch.nn import TransformerEncoder
from transformers import AutoModel,BertConfig,AutoTokenizer,CLIPVisionModel,CLIPVisionConfig,LlamaTokenizer
from models.llama.blocks import Transformer
from models.pmc_oa.blocks import Transformer,AttentionPool2d
from models.pmc_oa.pmc_clip import PMC_CLIP
import pdb

# from blocks import Transformer

from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)   

@dataclass
class CLIPTextCfg:
    bert_model_name: str = 'base'
    context_length: int = 77
    vocab_size: int = 32000
    width: int = 768
    heads: int = 8
    layers: int = 12
    fusion_layers: int = 1  # layers of fusion_module
    MOMENTUM: float = 0.5  # 0.99

@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)
    
def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config



class Binary_VQA_Model(nn.Module):
    def __init__(self,config): 
        super(Binary_VQA_Model, self).__init__()
        embed_dim = config.embed_dim
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(config.pretrained_tokenizer)
        self.llama_model = transformers.LlamaModel.from_pretrained(config.pretrained_model)
        peft_config = get_peft_config(PEFTArguments)
        self.text_encoder = get_peft_model(self.llama_model, peft_config)
        self.text_embed = nn.Sequential(nn.Linear(4096, embed_dim))
        
        # self.cls_id = 2  # [CLS]'s token id is 2, while it varies from tokenizers
        self.context_length = 256
        
        if config.image_encoder == "CLIP":
            self.image_encoder_name = "CLIP"
            configuration = CLIPVisionConfig(image_size=512)
            if config.clip_pretrained == "openai/clip-vit-base-patch32":
                self.image_encoder = CLIPVisionModel(configuration).from_pretrained(config.clip_pretrained)
                self.image_encoder.vision_model.embeddings = transformers.models.clip.modeling_clip.CLIPVisionEmbeddings(configuration)
            else:
                self.image_encoder = CLIPVisionModel(configuration)
        elif config.image_encoder == "PMC_CLIP":
            self.image_encoder_name = "PMC_CLIP"
            self.image_encoder = PMC_CLIP(embed_dim=768)
            checkpoint = torch.load(config.pmcclip_pretrained)
            state_dict = checkpoint['state_dict']
            state_dict.pop('module.visual.attnpool.positional_embedding')
            self.image_encoder.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()},strict=False)
            self.image_embed = nn.Sequential(nn.Linear(2048, embed_dim))
            
        self.qformer_query = nn.Parameter(torch.empty(32, 768))
        self.qformer_decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead=4, dim_feedforward=768, dropout=0.1, activation='relu',norm_first=True)
        self.qformer_decoder_norm = nn.LayerNorm(embed_dim)
        self.qformer_decoder = nn.TransformerDecoder(self.qformer_decoder_layer, 12, self.qformer_decoder_norm)
        
        text_cfg = CLIPTextCfg
        self.transformer_width = text_cfg.width
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        
        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        
        self.mlm_projection = nn.Parameter(torch.empty(text_cfg.width, text_cfg.vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_module = Transformer(
            width=text_cfg.width,
            layers=text_cfg.fusion_layers,
            heads=text_cfg.heads,
        )
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        self.init_parameters()
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def init_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
        if self.mlm_projection is not None:
            nn.init.normal_(self.mlm_projection, std=self.transformer_width ** -0.5)
            
    
    def forward(self,image,encoded_input_ids,encoded_attention_mask):
        batchsize,_ = encoded_input_ids.shape
        # image [8, 3, 512, 512]
        if self.image_encoder_name == "CLIP":
            image_features = self.image_encoder(image).last_hidden_state[:,1:,:] #as key and value [8, 256, 768]
        elif self.image_encoder_name == "PMC_CLIP":
            #pdb.set_trace() #image: [8, 3, 512, 512]
            image_features = self.image_encoder(image)  #[8, 2048, 16, 16]
            image_features = rearrange(image_features,'b n h w -> (b h w) n') #[2048, 2048]
            image_features = self.image_embed(image_features) #[2048, 768]
            image_features = rearrange(image_features,'(b n) d -> b n d', b=batchsize) #[8, 256, 768]
            
        # pdb.set_trace()
        image_query_features = self.qformer_query.unsqueeze(0).expand(batchsize, -1, -1) # [32, 768] --> [8, 32, 768]
        image_features = self.qformer_decoder(image_query_features.transpose(0,1), image_features.transpose(0,1)).transpose(0,1) # [8, 32, 768]
        question_features  = self.text_encoder(input_ids =encoded_input_ids,attention_mask = encoded_attention_mask)[0] #[8, 256, 4096]
        question_features = rearrange(question_features,'b n d -> (b n) d') #[2048, 4096]
        question_features = self.text_embed(question_features) #[2048, 4096]
        x = rearrange(question_features,'(b n)d -> b n d', b=batchsize) #[8, 256, 4096]
        #torch.Size([1, 256, 768])
        
        B, _len, _dim = x.shape #[8, 256, 768]
        # pdb.set_trace()
        img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [128, 1, embed_dim]  [1, 1, 768]-->[8, 1, 768]
        x = torch.cat([x, img_special_tokens, image_features], dim=1) # [8, 256, 768]  [8, 1, 768]  [8, 32, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_module(x) # [289, 8, 768]
        x = x.permute(1, 0, 2)  # LND -> NLD [8, 289, 768]
        x = x[:, :-33, :]  # Remove token [img_special_token, img] [8, 256, 768]
        out = self.softmax(x @ self.mlm_projection)  # [batch_size=128, n_ctx=77, vocab_size=49409] [8, 256, 32000]
        return out 

