import os
import json
import base64
import random
import datetime

import torch
import numpy as np
import gradio as gr

from peng_utils import TestBlip2, TestMiniGPT4, TestMplugOwl, TestMultimodelGPT, TestOtter, TestFlamingo

LOGDIR = '/data1/VLP_web_data/vote_data'
os.makedirs(LOGDIR, exist_ok=True)
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)
device = torch.device('cuda:4')
vlp_models = {
    'blip2': TestBlip2(),
    'minigpt4': TestMiniGPT4(),
    'owl': TestMplugOwl(),
    # 'mmgpt': TestMultimodelGPT(),
    'otter': TestOtter(),
    'flamingo': TestFlamingo()
}


def save_vote_data(state):
    t = datetime.datetime.now()
    log_name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    with open(log_name, 'a') as fout:
        state['image'] = np.array(state['image'], dtype='uint8').tolist()
        fout.write(json.dumps(state) + "\n")


def vote_up_model(state, chatbot):
    state['user_vote'] = 'up'
    save_vote_data(state)
    chatbot.append((
        'Your Vote: Up!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def vote_down_model(state, chatbot):
    state['user_vote'] = 'down'
    save_vote_data(state)
    chatbot.append((
        'Your Vote: Down!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def vote_model_tie(state, chatbot):
    state['user_vote'] = 'tie'
    save_vote_data(state)
    chatbot.append((
        'Your Vote: Tie!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def clear_chat(state):
    if state is not None:
        state = {}
    return state, None, gr.update(value=None, interactive=True), gr.update(placeholder="Enter text and press ENTER"), disable_btn, disable_btn, disable_btn, enable_btn


def user_ask(state, chatbot, text_box):
    state['text'] = text_box
    if text_box == '':
        return state, chatbot, '', enable_btn
    chatbot = chatbot + [[text_box, None], [text_box, None]] 
    return state, chatbot, '', disable_btn


def run_VLP_models(state, chatbot, gr_img):
    if state['text'] == '' or gr_img is None:
        return state, chatbot, enable_btn, disable_btn, disable_btn, disable_btn, enable_btn

    selected_VLP_models = random.sample(list(vlp_models.keys()), 2)
    vlp_outputs = [vlp_models[x].generate(state['text'], gr_img, device) for x in selected_VLP_models]
    state['image'] = gr_img
    state['VLP_names'] = selected_VLP_models
    state['VLP_outputs'] = vlp_outputs
    chatbot[-2][1] = vlp_outputs[0]
    chatbot[-1][1] = vlp_outputs[1]
    return state, chatbot, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn


with gr.Blocks() as demo:
    state = gr.State({})

    with gr.Row():
        with gr.Column(scale=0.5):
            imagebox = gr.Image(type="pil")
            with gr.Row() as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
        with gr.Column():
            chatbot = gr.Chatbot(label='ChatBox')
            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(placeholder="Enter text and press ENTER")
                with gr.Column(scale=1, min_width=60):
                    submit_btn = gr.Button(value="Submit")
    
    btn_list = [upvote_btn, downvote_btn, tie_btn, clear_btn]
    textbox.submit(user_ask, [state, chatbot, textbox], [state, chatbot, textbox, submit_btn]).then(run_VLP_models, [state, chatbot, imagebox], [state, chatbot, submit_btn] + btn_list)
    submit_btn.click(user_ask, [state, chatbot, textbox], [state, chatbot, textbox, submit_btn]).then(run_VLP_models, [state, chatbot, imagebox], [state, chatbot, submit_btn] + btn_list)
    clear_btn.click(clear_chat, [state], [state, chatbot, imagebox, textbox] + btn_list)
    upvote_btn.click(vote_up_model, [state, chatbot], [chatbot] + btn_list)
    downvote_btn.click(vote_down_model, [state, chatbot], [chatbot] + btn_list)
    tie_btn.click(vote_model_tie, [state, chatbot], [chatbot] + btn_list)


server_name = '103.108.182.56' # '127.0.0.1'
demo.launch(share=True, enable_queue=True, server_name='0.0.0.0', server_port=7860)