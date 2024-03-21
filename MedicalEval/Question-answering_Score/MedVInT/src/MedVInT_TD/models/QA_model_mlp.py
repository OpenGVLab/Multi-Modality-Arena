#from .transformer import *
from dataclasses import dataclass, field
import tqdm.auto as tqdm
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange
import transformers
from transformers import CLIPVisionConfig
from .blocks import ModifiedResNet,PMC_CLIP_cfg
import torchvision.models as models
import json
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
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
  
class QA_model(nn.Module):
    def __init__(self, model_args):  
        super(QA_model, self).__init__()  
        self.hidden_dim = model_args.hidden_dim
        self.voc_size = model_args.voc_size
        self.img_tokens = model_args.img_token_num
        self.H = model_args.H 
        self.N = model_args.N 
        self.Vision_module = model_args.Vision_module
        
        ###################################
        ''' Visual Model'''
        ###################################

        if self.Vision_module == 'PMC-CLIP':
            #vision_cfg = json.load(open(model_args.visual_model_config,'r'))['vision_cfg']
            vision_cfg = PMC_CLIP_cfg()
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            vision_model = ModifiedResNet(
                layers=vision_cfg.layers,
                heads=vision_heads,
                output_dim = 768,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
            vision_model = self.vision_load_pretrain(vision_model,model_args.visual_model_path)
            self.vision_model = nn.Sequential(*list(vision_model.children())[:-2])
            num_ftrs = 1024
            
        if self.Vision_module == "CLIP":
            self.vision_model = transformers.CLIPVisionModel.from_pretrained(model_args.visual_model_path,ignore_mismatched_sizes=True)
            num_ftrs = 768
        if self.Vision_module == 'Scratch':
            self.vision_model = transformers.CLIPVisionModel(config=CLIPVisionConfig(image_size=512))
            num_ftrs = 768
        
        ###################################
        ''' Query Decoder'''
        ###################################

        # self.query_embed = nn.Embedding(self.img_tokens, num_ftrs) 
        
        # decoder_layer = TransformerDecoderLayer(num_ftrs, self.H, 1024,
        #                                 0.1, 'relu',normalize_before=True)
        # decoder_norm = nn.LayerNorm(num_ftrs)
        # self.decoder = TransformerDecoder(decoder_layer, self.N , decoder_norm,
        #                           return_intermediate=False)

        ###################################
        ''' FC '''
        ###################################
        
        self.fc_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2 = nn.Linear(num_ftrs, self.hidden_dim)
        
        ###################################
        ''' Large Langugae Model'''
        ###################################
        self.llamacasual = self.Setup_model(model_args)
        
    def vision_load_pretrain(self,resnet,model_path):
        checkpoint = torch.load(model_path, map_location='cpu') 
        state_dict = checkpoint['state_dict'] 
        state_dict = {k.replace('module.visual.',''): v for k, v in state_dict.items() if '.visual' in k}
        resnet.load_state_dict(state_dict)
        return resnet
                
    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")    

    def Setup_model(self, model_args):
        print("Setup Model")
        model = transformers.LlamaForCausalLM.from_pretrained(
           model_args.model_path,
        )
        if model_args.checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
        if model_args.is_lora:
            print("Setup PEFT")
            peft_config = get_peft_config(peft_args=model_args)
            model = get_peft_model(model, peft_config)
        return model
    
    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        if self.Vision_module == 'PMC-CLIP':
            batch_size = xis.shape[0]
            res_fea = self.vision_model(xis) #batch_size,feature_size,patch_num,patch_num
            out_emb = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
            out_emb = out_emb.mean(dim = 1)
            #h = rearrange(res_fea,'b n d -> (b n) d')
            #batch_size,num,feature_size
            # h = h.squeeze()
            #out_emb = res_fea
        if self.Vision_module == 'CLIP':
            out_emb = self.vision_model(pixel_values = xis)['last_hidden_state'][:,0,:] # dismiss the cls token dim=b n d
        if self.Vision_module == 'Scratch':
            out_emb = self.vision_model(pixel_values = xis)['last_hidden_state'][:,0,:] # dismiss the cls token dim=b n d
        out_emb = out_emb.unsqueeze(dim = 1)
        return out_emb
    
    def forward(self,input_ids,images,labels = None):
        
        B = images.shape[0]
        ### images encoding ###
        x = self.image_encoder(images)
        features = x #patch_num b dim
        
        ### Q-Former ###
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        # features,ws = self.decoder(query_embed, features, 
        #     memory_key_padding_mask=None, pos=None, query_pos=None)
        #features = features.transpose(0,1)
        ### fc  ### 
        features = rearrange(features,'b n d  -> (b n) d')
        features = self.fc_l1(features)
        features = F.relu(features)
        features = self.fc_l2(features)
        features = rearrange(features,'(b n) d -> b n d',b=B)
        ### LLM ###
        input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([features,input_embedding], dim=1)
        labels = labels[:,31:]
        output = self.llamacasual(inputs_embeds = input_embedding, labels = labels)
        
        return output

    def generate(self,input_ids,images):
        with torch.no_grad():
            B = images.shape[0]
            ### images encoding ###
            x = self.image_encoder(images)
            features = x.transpose(0,1) #patch_num b dim
            
            ### Q-Former ###
            # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
            # features,ws = self.decoder(query_embed, features, 
            #     memory_key_padding_mask=None, pos=None, query_pos=None)
            features = features.transpose(0,1)
            features = rearrange(features,'b n d  -> (b n) d')
            features = self.fc_l1(features)
            features = F.relu(features)
            features = self.fc_l2(features)
            features = rearrange(features,'(b n) d -> b n d',b=B)
            ### LLM ###
            input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([features,input_embedding], dim=1)
            
            generation = self.llamacasual(inputs_embeds = input_embedding)['logits']
            #generation = self.llamacasual.generate(inputs_embeds = input_embedding,max_length=100, do_sample=True, top_k=50)
            return generation
    
    def generate_long_sentence(self,input_ids,images):
        with torch.no_grad():
            B = images.shape[0]
            ### images encoding ###
            x = self.image_encoder(images)
            features = x.transpose(0,1) #patch_num b dim
            
            ### Q-Former ###
            # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
            # features,ws = self.decoder(query_embed, features, 
            #     memory_key_padding_mask=None, pos=None, query_pos=None)
            features = features.transpose(0,1)
            features = rearrange(features,'b n d  -> (b n) d')
            features = self.fc_l1(features)
            features = F.relu(features)
            features = self.fc_l2(features)
            features = rearrange(features,'(b n) d -> b n d',b=B)
            ### LLM ###
            input_embedding = self.llamacasual.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([features,input_embedding], dim=1)
            
            #generation = self.llamacasual(inputs_embeds = input_embedding)['logits']
            generation = self.llamacasual.generate(inputs_embeds = input_embedding, max_new_tokens =50)
            return generation