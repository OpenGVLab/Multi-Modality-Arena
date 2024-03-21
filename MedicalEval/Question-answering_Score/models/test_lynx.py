import torch
import ruamel.yaml as yaml
from .lynx.models.lynx import LynxBase
from .lynx.dataset import get_image_transform
from . import get_image


class TestLynx:
    def __init__(self, model_name, device=None) -> None:
        device = torch.device('cuda' if device is None else device)
        config = yaml.load(open('models/lynx/configs/LYNX.yaml', 'r'), Loader=yaml.Loader)
        model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)
        model = model.to(device, dtype=torch.float16)
        model.eval()
        self.model = model
        self.config = config
        self.device = device

        self.use_left_pad = config['use_left_pad']
        self.tokenizer = model.tokenizer
        _, self.img_transform = get_image_transform(config)
        self.lower_text = config['lower_text']

    def pad_batch_data(self, input_ids):
        max_length = max([len(ids) for ids in input_ids])

        # pad + get atts
        input_ids_pad = []
        input_atts_pad = []

        for ids in input_ids:
            n_tokens = len(ids)
            n_pads = max_length - n_tokens

            if self.use_left_pad:  # recommend
                input_ids_pad.append([self.tokenizer.pad_token_id] * n_pads + ids)
                input_atts_pad.append([0] * n_pads + [1] * n_tokens)

            else:  # right pad
                input_ids_pad.append(ids + [self.tokenizer.pad_token_id] * n_pads)
                input_atts_pad.append([1] * n_tokens + [0] * n_pads)

        input_ids_pad = torch.LongTensor(input_ids_pad)
        input_atts_pad = torch.LongTensor(input_atts_pad)

        return input_ids_pad, input_atts_pad

    @torch.no_grad()
    def batch_data_process(self, image_list, question_list):
        images = [get_image(image) for image in image_list]
        images = [self.img_transform(image) for image in images]
        vision_input = torch.stack(images, dim=0)

        prompts = [f"User: {question}\nBot:" for question in question_list]
        prompts = [prompt.lower() if self.lower_text else prompt for prompt in prompts]

        input_ids_list = [[self.tokenizer.bos_token] + self.tokenizer.tokenize(prompt) for prompt in prompts]
        input_ids_list = [self.tokenizer.convert_tokens_to_ids(input_ids) for input_ids in input_ids_list]
        input_ids_pad, input_atts_pad = self.pad_batch_data(input_ids_list)

        return vision_input, input_ids_pad, input_atts_pad

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=64):
        vision_input, input_ids, input_atts = self.batch_data_process(image_list, question_list)
        vision_input = vision_input.to(self.device, non_blocking=True)
        input_ids = input_ids.to(self.device)
        input_atts = input_atts.to(self.device)

        with torch.cuda.amp.autocast():
            text_outputs = self.model.generate(
                vision_input=vision_input,
                input_ids=input_ids, input_atts=input_atts,
                use_nucleus_sampling=self.config.get('use_nucleus_sampling', False),
                apply_lemmatizer=self.config['apply_lemmatizer'],
                num_beams=self.config['num_beams'],
                min_length=self.config['min_length'],
                length_penalty=self.config.get('length_penalty', 1.0),
                no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', -1),
                top_p=self.config.get('top_p', 0.9),
                top_k=self.config.get('top_k', 3),
                max_new_tokens=max_new_tokens # self.config.get('max_new_tokens', 64)
            )

        outputs = [output.strip() for output in text_outputs]
        return outputs