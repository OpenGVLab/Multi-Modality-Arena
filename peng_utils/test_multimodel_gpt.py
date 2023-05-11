import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor
from .flamingo.modeling_flamingo import FlamingoForConditionalGeneration

open_flamingo_path = '/nvme/data1/VLP_web_data/openflamingo-9b-hf'
finetune_path = "/nvme/data1/VLP_web_data/Multimodel-GPT/mmgpt-lora-v0-release.pt"


def get_prompt(message, have_image):
    sep = "\n\n### "
    format_dict = {"user_prefix": "Instruction", "ai_prefix": "Response"}
    prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt_template = prompt_template.format(**format_dict)
    ret = prompt_template

    context = []
    if have_image:
        context.append(sep + "Image:\n<image>" + sep + 'Instruction' + ":\n" + message)
    else:
        context.append(sep + 'Instruction' + ":\n" + message)
    context.append(sep + 'Response' + ":\n")
    ret += "".join(context[::-1])
    
    return ret


def prepare_model_for_tuning(model: torch.nn.Module, config):
    if config.lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",  # won't use bias currently
            modules_to_save=[],  # TODO: might be helpful if save partial model
            task_type="VL",
        )
        model.lang_encoder = get_peft_model(model.lang_encoder, peft_config=lora_config)

    # manually unfreeze modules, we use a `substring` fashion mathcing
    for name, param in model.named_parameters():
        if any(substr in name for substr in config.unfrozen):
            param.requires_grad = True

    return model


class TestMultimodelGPT:
    def __init__(self, finetune_path=finetune_path, open_flamingo_path=open_flamingo_path):
        model = FlamingoForConditionalGeneration.from_pretrained(open_flamingo_path)
        image_processor = CLIPImageProcessor()
        text_tokenizer = model.text_tokenizer
        text_tokenizer.padding_side = "left"
        # text_tokenizer.add_eos_token = False
        # text_tokenizer.bos_token_id = 1
        # text_tokenizer.eos_token_id = 2

        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)
            model = prepare_model_for_tuning(model, tuning_config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.model = model
        self.tokenizer = text_tokenizer
        self.image_processor = image_processor


    def generate(self, text, image=None, device=None, keep_in_device=False):
        # try:
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.model = self.model.to(device, dtype=dtype)
            self.model.vision_encoder = self.model.vision_encoder.to('cpu', dtype=torch.float32)
        else:
            dtype = torch.float32
            self.model = self.model.to('cpu').float()
    
        prompt = get_prompt(text, image is not None)
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))

        output_ids = self.model.generate(
            # vision_x=vision_x.to(self.model.device),
            vision_x=vision_x.to('cpu'),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=dtype),
            max_new_tokens=256,
            num_beams=1,
            no_repeat_ngram_size=3,
        )
        result = self.tokenizer.decode(output_ids[0]) #, skip_special_tokens=True)
        
        if not keep_in_device:
            self.model = self.model.to('cpu')

        return result
        # except Exception as e:
        #     return getattr(e, 'message', str(e))
