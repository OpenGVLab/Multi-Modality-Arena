import os
import gc
import torch
import argparse
from models import get_model


class Chat:
    def __init__(self, human_name='Human', model_name='Assistant', seperator='\n', system_begin='', system_end='', sep_style='default'):
        self.history_chat = []
        self.human = human_name
        self.model = model_name
        self.sep = seperator
        self.sys_begin = system_begin
        self.sys_end = system_end
        self.sep_style = sep_style

    def get_prompt(self):
        ret = self.sys_begin
        if self.sep_style == 'default':
            for role, message in self.history_chat:
                if message:
                    ret += role + ": " + message.strip() + " " + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == 'mpt':
            for i, (role, message) in enumerate(self.history_chat):
                if message:
                    if role == self.human and i == len(self.history_chat) - 2:
                        ret += role + message.strip() + '\n<image>' + self.sep
                    else:
                        ret += role + message.strip() + self.sep
                else:
                    ret += role
        elif self.sep_style == 'gpt':
            for i, (role, message) in enumerate(self.history_chat):
                if message:
                    if role == self.human and i == len(self.history_chat) - 2:
                        ret += role + ": " + '<Img><ImageHere></Img> ' + message.strip() + self.sep
                    else:
                        ret += role + ": " + message.strip() + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == 'otter':
            for i, (role, message) in enumerate(self.history_chat):
                if message:
                    if role == self.human:
                        ret += role + ": " + message.strip() + self.sep
                    else:
                        ret += role + ":<answer> " + message.strip() + "<|endofchunk|>"
                else:
                    if role == self.model:
                        ret += role + ":<answer>"
                    else:
                        ret += role + ": "
        ret += self.sys_end
        return ret
    
    def ask(self, input_text):
        self.history_chat.append([self.human, input_text])
        self.history_chat.append([self.model, None])
        return self.get_prompt()

    def answer(self, model_output):
        self.history_chat[-1][1] = model_output

    def clear(self):
        self.history_chat = []
        gc.collect()


def get_chat(model_name):
    if model_name == 'BLIP2':
        return Chat()
    elif model_name == 'InstructBLIP':
        return Chat()
    elif model_name == 'LLaMA-Adapter-v2':
        return Chat(
            human_name='Instruction',
            model_name='Response',
            seperator='\n\n###',
            system_begin="Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### "
        )
    elif model_name == 'LLaVA':
        return Chat(
            human_name="<|im_start|>user\n",
            model_name="<|im_start|>assistant\n",
            seperator="<|im_end|>",
            sep_style='mpt',
            system_begin="""<|im_start|>system
- You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.<|im_end|>"""
        )
    elif model_name == 'MiniGPT-4':
        return Chat(
            seperator='###',
            sep_style='gpt',
            system_begin="Give the following image: <Img>ImageContent</Img>. "
                         "You will be able to see the image once I provide it to you. Please answer my questions.###",
        )
    elif model_name == 'VPGTrans':
        return Chat(
            seperator='###',
            sep_style='gpt',
            system_begin="###"
        )
    elif model_name == 'mPLUG-Owl':
        return Chat(
            human_name="Human",
            model_name="AI",
            seperator='\n',
            system_begin="The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\n"
        )
    elif model_name in ['Otter', 'Otter-Image']:
        return Chat(
            human_name='<image>User',
            model_name='GPT',
            seperator=' ',
            sep_style='otter'
        )
    else:
        raise NotImplementedError(f"Invalid model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", type=str, default="BLIP2")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name, device=torch.device('cuda'))
    image = 'multi_turn_example.jpg'

    chat = get_chat(args.model_name)
    while True:
        user_input = input('User: ')
        input_prompt = chat.ask(user_input)
        model_answer = model.pure_generate(image, input_prompt)
        chat.answer(model_answer)
        print(f"Input prompt: {input_prompt}")
        print(f"Model output: {model_answer}")
