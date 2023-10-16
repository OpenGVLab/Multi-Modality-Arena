import os
import torch
import time
import json

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from .lavin.eval_model import ModelArgs, Transformer
from .lavin.tokenizer import Tokenizer
from .lavin.generator import LaVIN_Generator
from .lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter

import warnings
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from .lavin.util.apply_delta import apply_model_delta_online

warnings.filterwarnings('ignore')


def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def load(
    ckpt_dir: str,
    llm_model:str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim:int,
    adapter_scale:float,
    hidden_proj:int,
    visual_adapter_type: str,
    temperature: float,
use_vicuna: bool
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")


    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)
    set_Clip_Adapter(model.backbone.visual, visual_adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


world_size, local_rank = 1, 0
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1203'
torch.distributed.init_process_group("nccl", init_method='env://', world_size=world_size, rank=local_rank)
initialize_model_parallel(world_size)
torch.cuda.set_device(local_rank)

from . import get_image, DATA_DIR
ckpt_dir = f'{DATA_DIR}/llama_checkpoints/'
adapter_path = f'{DATA_DIR}/LaVIN/sqa-llama-7b.pth'


class TestLaVIN:
    def __init__(self, device=None) -> None:
        self.generator = load(ckpt_dir, '7B', f"{ckpt_dir}/tokenizer.model", adapter_path, local_rank, world_size, 512, 16, 'attn', 8, 1, 128, 'router', 5, False)
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        image = [self.image_transforms(image)]
        image = torch.stack(image, dim=0)
        prompt = [f"Instruction: {question}\nResponse: "]
        results = self.generator.generate(
            prompt, images=image, indicators=[1], max_gen_len=max_new_tokens, temperature=0.1, top_p=0.75, n_feats=6
        )
        result = results[0]
        result = result.lower().strip().split('response:')[1]
        return result
    
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        prompts = [f"Instruction: {question}\nResponse: " for question in question_list]
        prompts = [prompt.replace("  ", " ").strip() for prompt in prompts]
        images = [Image.open(x).convert('RGB') for x in image_list]
        images = [self.image_transforms(x) for x in images]
        # images = [image.unsqueeze(0) for image in images]
        images = torch.stack(images, dim=0)

        with torch.cuda.amp.autocast():
            results = self.generator.generate(
                prompts, images=images, indicators=[1] * len(prompts), max_gen_len=max_new_tokens, temperature=0.1, top_p=0.75, n_feats=6
            )
        results = [result.lower().strip().split('response:')[1] for result in results]
        return results
