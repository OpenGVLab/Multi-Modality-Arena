from transformers.utils.bitsandbytes import *
from transformers import BitsAndBytesConfig
import torch
from torch import  nn
import bitsandbytes as bnb

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
def _replace_with_bnb_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, has_been_replaced=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, nn.Linear) or isinstance(module, ColumnParallelLinear)  or isinstance(module, RowParallelLinear)  ) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                # with init_empty_weights():
                if quantization_config.quantization_method() == "llm_int8":
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                        threshold=quantization_config.llm_int8_threshold,
                    )
                    has_been_replaced = True
                else:
                    if (
                        quantization_config.llm_int8_skip_modules is not None
                        and name in quantization_config.llm_int8_skip_modules
                    ):
                        pass
                    else:
                        model._modules[name] = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                        )
                        has_been_replaced = True
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def quant_model_bnb(model, quant_bit='4bit', keep_in_fp32_modules=[],
                    quantization_config=None):
    if quantization_config is None:
        # set default quantization config
        # compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_bit == '4bit',
            load_in_8bit=quant_bit == '8bit',
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    model,_ = _replace_with_bnb_linear(
        model, modules_to_not_convert=keep_in_fp32_modules, quantization_config=quantization_config
    )

    return model

