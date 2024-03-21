import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig

# THUDM/visualglm-6b
# internlm/internlm-xcomposer-7b
# Qwen/Qwen-VL-Chat


class TestAutoModel:
    def __init__(self, model_name, device=None) -> None:
        device = 'cuda' if device is None else device
        if 'Qwen' in model_name:
            self.response_type = 'Qwen'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, bf16=True).eval()
            self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        else:
            self.response_type = 'intern' if 'intern' in model_name else 'glm'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(device)
            if self.response_type == 'intern':
                self.model.tokenizer = self.tokenizer

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=64):
        assert type(image) is str
        with torch.cuda.amp.autocast():
            if self.response_type == 'glm':
                response, history = self.model.chat(self.tokenizer, image, question, history=[], max_length=max_new_tokens)
            elif self.response_type == 'intern':
                response, history = self.model.chat([question], image, max_new_tokens=max_new_tokens)
            elif self.response_type == 'Qwen':
                query = self.tokenizer.from_list_format([{'image': image}, {'text': question}])
                response, history = self.model.chat(self.tokenizer, query=query, history=None) 
            else:
                raise NotImplementedError(f"Invalid response type: {self.response_type}")
        return response

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1282):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output