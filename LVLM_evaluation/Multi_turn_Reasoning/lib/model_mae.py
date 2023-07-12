import os
import json
import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, List

import clip
from sentencepiece import SentencePieceProcessor
from timm.models.vision_transformer import PatchEmbed, Block

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear

MAX_SEQ_LEN=256
MAX_BATCH_SIZE=64

tokenizer_path = './LLaMA-Adapter-v2/LLaMA-7B/tokenizer.model'
llama_7b_dir = './LLaMA-Adapter-v2/LLaMA-7B/'


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
      
        self.bias = True

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=self.bias
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=self.bias
        )
        if self.bias:
            nn.init.constant_(self.wq.bias.data, 0)
#            nn.init.constant_(self.wk.bias.data, 0)
#            nn.init.constant_(self.wv.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)
#        self.gate = torch.nn.Parameter(torch.randn(1, self.n_local_heads, 1, 1))
        self.lora = True
        self.lora_rank = 16
        if self.lora:
           self.lora_wq_l1 = Linear(args.dim, self.lora_rank, bias=False)
           self.lora_wq_l2 = Linear(self.lora_rank, args.dim, bias=False)

           self.lora_wk_l1 = Linear(args.dim, self.lora_rank, bias=False)
           self.lora_wk_l2 = Linear(self.lora_rank, args.dim, bias=False)

           self.lora_wv_l1 = Linear(args.dim, self.lora_rank, bias=False)
           self.lora_wv_l2 = Linear(self.lora_rank, args.dim, bias=False)

           self.lora_wo_l1 = Linear(args.dim, self.lora_rank, bias=False)
           self.lora_wo_l2 = Linear(self.lora_rank, args.dim, bias=False)
           nn.init.constant_(self.lora_wq_l2.weight.data, 0)
           nn.init.constant_(self.lora_wk_l2.weight.data, 0)
           nn.init.constant_(self.lora_wv_l2.weight.data, 0)
           nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ) # .cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ) # .cuda()
        
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.new_gate = torch.nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        if prompt is not None:
            prompt_len = prompt.shape[1]
            prompt_k = self.wk(prompt).view(bsz, prompt_len, self.n_local_heads, self.head_dim)
            prompt_v = self.wv(prompt).view(bsz, prompt_len, self.n_local_heads, self.head_dim)

            prompt_k = prompt_k.transpose(1, 2)
            prompt_v = prompt_v.transpose(1, 2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        
        
        if prompt is not None:
            prompt_scores = torch.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            prompt_scores = self.gate.tanh().half() * self.new_gate * F.softmax(prompt_scores.float(), dim=-1).type_as(xq)
            output = output + torch.matmul(prompt_scores, prompt_v)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        if self.lora:
           return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
           return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.bias = True
        self.w1 = Linear(
            dim, hidden_dim, bias=self.bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=self.bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=self.bias
        )
        if self.bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

        self.lora = True
        self.lora_rank = 16
        if self.lora:
           self.lora_w1_l1 = Linear(dim, self.lora_rank, bias=False)
           self.lora_w1_l2 = Linear(self.lora_rank, hidden_dim, bias=False)
           self.lora_w2_l1 = Linear(hidden_dim, self.lora_rank, bias=False)
           self.lora_w2_l2 = Linear(self.lora_rank, dim, bias=False)
           self.lora_w3_l1 = Linear(dim, self.lora_rank, bias=False)
           self.lora_w3_l2 = Linear(self.lora_rank, hidden_dim, bias=False)
           nn.init.constant_(self.lora_w1_l2.weight.data, 0)
           nn.init.constant_(self.lora_w2_l2.weight.data, 0)
           nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        if self.lora:
           out = F.silu(self.w1(x) + self.lora_w1_l2(self.lora_w1_l1(x))) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
           return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
        else:
           return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()






class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)




class Config(object):
    def __init__(self):
        self.hidden_dim = 512
        self.pad_token_id = 0
        self.max_position_embeddings = 63
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 1
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim).to('cpu')
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)).to('cpu')
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False).to('cpu')  # fixed sin-cos embedding

        print('Get encoder specifics')

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True).to('cpu')

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)).to('cpu')

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False).to('cpu')  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(768, 16, 4, qkv_bias=True, norm_layer=norm_layer).to('cpu')
            for i in range(8)])
        
        print('Get decoder specifics')

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # self.initialize_weights()
        checkpoint = torch.load(os.path.join(llama_7b_dir, 'consolidated.00.pth'), map_location="cpu")
###        adapter_weight = torch.load('llama_adapter_len10_layer30_release.pth', map_location="cpu")
        with open(os.path.join(llama_7b_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=MAX_SEQ_LEN, max_batch_size=MAX_BATCH_SIZE, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)

        print('Load checkpoint and prepare tokenizer')

        model = Transformer(model_args)

        print('Get the transformer model')

        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
#        model.load_state_dict(adapter_weight, strict=False)
        self.llma = model
        for name, para in self.llma.named_parameters():
                para.requires_grad = False
        self.prompt_layer = 31

        for name, para in self.llma.named_parameters():
            if 'norm' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'bias' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'lora' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'new_gate' in  name:
                para.data = para.data.float()
                para.requires_grad = True
            if '.0.' in name:
                para.data = para.data.half()
                para.requires_grad = False
        
        print('Freeze model weight')

        self.visual_query = nn.Embedding(10, 768)
        self.prefix_query = nn.Embedding(10 * self.prompt_layer, model_args.dim)
        self.prefix_projector_norm = nn.LayerNorm(model_args.dim)
        self.gpt_embedding_size = model_args.dim
        self.prefix_projector = nn.Linear(768, model_args.dim)
        self.clip, _ = clip.load("ViT-L/14", device='cpu')

        print('Get clip model')

        self.clip_proj = nn.Linear(768, 768)
        self.clip_proj_norm = nn.LayerNorm(768)
        for name, para in self.clip.named_parameters():
            para.requires_grad = False
        self.clip_proj.weight.requires_grad = False
        self.clip_proj.bias.requires_grad = False
        self.clip_proj_norm.weight.requires_grad = False
        self.clip_proj_norm.bias.requires_grad = False
        self.prefix_projector_norm.weight.requires_grad = False
        self.prefix_projector_norm.bias.requires_grad = False
        self.prefix_query.weight.requires_grad = False
        self.prefix_projector.weight.requires_grad = False
        self.prefix_projector.bias.requires_grad = False
        for name, para in self.blocks.named_parameters():
            para.requires_grad = False
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #        print(f"Trainable param: {name}, {param.shape}, {param.dtype}") 
    
        
    def encode_image(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x


    @torch.no_grad()
    def generate(
        self,
        imgs,
        prompt_prefix,
        max_gen_len: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.75,
        ):

        bsz = len(imgs)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        ### Processing visual features
        # if self.clip.dtype != torch.float32:
        #     self.clip.float()
        with torch.cuda.amp.autocast():
            visual_feats = self.encode_image(imgs)
        visual_feats = self.clip_proj_norm(self.clip_proj(visual_feats))
        query = self.visual_query.weight.unsqueeze(0).repeat(bsz, 1, 1)
        query = torch.cat([query, visual_feats], dim=1)
        for block in self.blocks:
            query = block(query)
        query = query[:, :10, :]
        query = self.prefix_projector(query)
        query = self.prefix_projector_norm(query)
        visual_tokens = query
        len_visual = visual_tokens.shape[1]

        if isinstance(prompt_prefix[0], str):
            prompt_prefix = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompt_prefix]

        min_prompt_size = min([len(t) for t in prompt_prefix])
        max_prompt_size = max([len(t) for t in prompt_prefix])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long().to(visual_feats.device)
        for k, t in enumerate(prompt_prefix):
            tokens[k, : len(t)] = torch.tensor(t).long().to(visual_feats.device)

        get_result = [False for _ in range(bsz)]

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.forward_inference(visual_tokens, tokens[:, prev_pos:cur_pos], prev_pos)
            # logits = self.forward_inference(visual_tokens, tokens[:, :cur_pos], 0)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            for idx in range(len(next_token)):
                if next_token[idx] == self.tokenizer.eos_id:
                    get_result[idx] = True
            if all(get_result):
                break
            prev_pos = cur_pos
 
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_prefix[i]) : len(prompt_prefix[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            if -1 in t:
                t.remove(-1)
            result = self.tokenizer.decode(t)
            decoded.append(result)

        return decoded


    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


    def forward_inference(self, visual_tokens, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        
        h = self.llma.tok_embeddings(tokens)
        freqs_cis = self.llma.freqs_cis.to(h.device)
        # freqs_cis = freqs_cis[:seqlen]
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        # start_pos = 0
        for layer in self.llma.layers[:-1 * self.prompt_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        prompt_index = 0 
        prompt = self.prefix_query.weight.reshape(-1, 10, 4096).unsqueeze(1)

        for layer in self.llma.layers[-1 * self.prompt_layer:]:
            dynamic_prompt = prompt[prompt_index].repeat(_bsz, 1, 1)
            dynamic_prompt = dynamic_prompt + visual_tokens
            h = layer(h, start_pos, freqs_cis, mask, dynamic_prompt)
            prompt_index = prompt_index + 1


        h = self.llma.norm(h)
        output = self.llma.output(h[:, -1, :])

        return output.float()


    def forward(self, examples, labels, example_mask, visual_feats, mask_ratio=0.75):
        visual_feats, _ = self.clip.encode_image(visual_feats)
        _bsz, seqlen = examples.shape
        if not self.training:
            visual_feats = self.clip_proj_norm(self.clip_proj(visual_feats.float()))
            query = self.visual_query.weight.unsqueeze(0).repeat(_bsz, 1, 1)
            query = torch.cat([query, visual_feats], dim=1)
            for block in self.blocks:
                query = block(query)
            query = query[:, :10, :]
            query = self.prefix_projector(query)
            query = self.prefix_projector_norm(query)
            visual_proj = query
#            visual_proj = self.visual_f2(self.visual_f1(visual_feats.float().mean(1)))
        else:
            visual_feats = self.clip_proj_norm(self.clip_proj(visual_feats))
            query = self.visual_query.weight.unsqueeze(0).repeat(_bsz, 1, 1)
            query = torch.cat([query, visual_feats], dim=1)
            for block in self.blocks:
                query = block(query)
            query = query[:, :10, :]
            query = self.prefix_projector(query)
            query = self.prefix_projector_norm(query)
            visual_proj = query
#            visual_proj = self.visual_f2(self.visual_f1(visual_feats.mean(1)))
        with torch.no_grad():
             h = self.llma.tok_embeddings(examples)
             freqs_cis = self.llma.freqs_cis.to(h.device)
             freqs_cis = freqs_cis[:seqlen]
             mask = None
             mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
             mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
             start_pos = 0
             for layer in self.llma.layers[:-1 * self.prompt_layer]:
                 h = layer(h, start_pos, freqs_cis, mask)
        prompt_index = 0 
        prompt = self.prefix_query.weight.reshape(-1, 10, 4096).unsqueeze(1)
        for layer in self.llma.layers[-1 * self.prompt_layer:]:
            #            if prompt_index < 25:
#                with torch.no_grad():
#                    dynamic_prompt = prompt[prompt_index].repeat(_bsz, 1, 1)
 #                   dynamic_prompt = dynamic_prompt + visual_proj
  #                  h = layer(h, start_pos, freqs_cis, mask, dynamic_prompt.half())
#            else:
                dynamic_prompt = prompt[prompt_index].repeat(_bsz, 1, 1)
                dynamic_prompt = dynamic_prompt + visual_proj
                h = layer(h, start_pos, freqs_cis, mask, dynamic_prompt.half())
                prompt_index = prompt_index + 1
        h = self.llma.norm(h)
        output = self.llma.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]
#        new_labels = (labels * 0).to(labels)
#        output = output[:, 5:-1, :]
#        captions = captions[:, 1:]
#        labels = new_labels
        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
#        if torch.isnan(c_loss):
#            c_loss = 0 * c_loss

        pred = 0
        mask = 0
        return c_loss, c_loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
