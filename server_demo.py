"""
The gradio demo server for chatting with a single model.
"""
import os
import time
import json
import uuid
import random
import datetime
import argparse
import requests
import multiprocessing

import gradio as gr
import numpy as np
from PIL import Image

from server_utils.constants import (
    LOGDIR, WORKER_API_TIMEOUT, CONVERSATION_SAVE_DIR,
    rules_markdown, notice_markdown, license_markdown
)
from server_utils.utils import build_logger, server_error_msg

os.makedirs(CONVERSATION_SAVE_DIR, exist_ok=True)
os.makedirs(f"{CONVERSATION_SAVE_DIR}/images", exist_ok=True)
logger = build_logger("web_server", f"{LOGDIR}/web_server.log")
headers = {"User-Agent": "fastchat Client"}
model_list = []
controller_url = None
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

model_info = {
    'blip2': 'BLIP2-flan-t5-xl',
    'flamingo': 'OpenFlamingo-9B',
    'minigpt4': 'MiniGPT-4-7B',
    'owl': 'mPLUG-Owl-Pretrained',
    'otter': 'Otter-9B'
}


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    return list(set(ret.json()["models"]))


def get_model_worker_addr(model_name):
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    return ret.json()["address"]


def save_vote_data(state, request: gr.Request):
    t = datetime.datetime.now()
    # save image
    img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    while os.path.exists(img_name):
        img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    image = np.array(state['image'], dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image.save(img_name)
    # save conversation
    log_name = os.path.join(CONVERSATION_SAVE_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conversation.json")
    with open(log_name, 'a') as fout:
        log_data = state.copy()
        log_data['image'] = img_name
        log_data['ip'] = request.client.host
        fout.write(json.dumps(state) + "\n")


def vote_left_model(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'left'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_right_model(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'right'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_model_tie(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'tie'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_model_bad(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'bad'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def clear_chat(state):
    if state is not None:
        state = {}
    return state, None, None, gr.update(value=None), gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn


def share_click(state):
    return state


def user_ask(state, chatbot_A, chatbot_B, textbox, imagebox):
    if (textbox == '' is None) or (imagebox is None and 'image' not in state):
        state['get_input'] = False
        return state, chatbot_A, chatbot_B, "", "", textbox, imagebox, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn
    if imagebox is not None:
        state['image'] = np.array(imagebox, dtype='uint8').tolist()
        imagebox.save('.tmp_img.png')
    state['text'] = textbox
    state['get_input'] = True
    selected_VLP_models = random.sample(model_list, 2)
    state['VLP_names'] = selected_VLP_models
    chatbot_A = chatbot_A + [(('.tmp_img.png',), None), (textbox, None), (None, None)]
    chatbot_B = chatbot_B + [(('.tmp_img.png',), None), (textbox, None), (None, None)]
    return state, chatbot_A, chatbot_B, gr.update(value=None), gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn


def model_worker_stream_iter(worker_addr, state):
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json={"text": state['text'], "image": state['image']},
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def get_model_worker_output(worker_input):
    (worker_addr, state) = worker_input
    stream_iter = model_worker_stream_iter(worker_addr, state)
    try:
        for data in stream_iter:
            if data["error_code"] == 0:
                output = data["text"].strip()
                return output
            elif data["error_code"] == 1:
                output = data["text"] + f" (error_code: {data['error_code']})"
                return output
            time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        output = server_error_msg + f" (error_code: 4)"
        return output
    except Exception as e:
        output = server_error_msg + f" (error_code: 5, {e})"
        return output
    

def run_VLP_models(state, chatbot_A, chatbot_B):
    if 'get_input' not in state or not state['get_input']:
        return state, chatbot_A, chatbot_B, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn
    model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
    pool = multiprocessing.Pool()
    vlp_outputs = pool.map(get_model_worker_output, [(worker_addr, state) for worker_addr in model_worker_addrs])
    state['VLP_outputs'] = vlp_outputs
    chatbot_A[-1][1] = vlp_outputs[0]
    chatbot_B[-1][1] = vlp_outputs[1]
    return state, chatbot_A, chatbot_B, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn


def build_demo():
    with gr.Blocks(theme='snehilsanyal/scikit-learn', title='Multimodality Chatbot Arena') as demo:
        state = gr.State({})

        with gr.Row():
            gr.HTML(open("CVLAB/header.html", "r").read())
        gr.Markdown(notice_markdown)

        with gr.Box():
            with gr.Row():
                with gr.Column(scale=1):
                    imagebox = gr.Image(type="pil")
                    gr.Markdown(rules_markdown)
                with gr.Column():
                    model_name_A = gr.Markdown("")
                    chatbot_A = gr.Chatbot(label='Model A').style(height=550)
                with gr.Column():
                    model_name_B = gr.Markdown("")
                    chatbot_B = gr.Chatbot(label='Model B').style(height=550)
            with gr.Box():
                with gr.Row():
                    leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                    rightvote_btn = gr.Button(value="👉  B is better", interactive=False)
                    tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                    bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=20):
                textbox = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press ENTER"
                ).style(container=False)
            with gr.Column(scale=1, min_width=50):
                send_btn = gr.Button(value="Send")
        
        with gr.Row():
            regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
            clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
            share_btn = gr.Button(value="📷  Share")
        
        gr.Examples(examples=[
            [f"examples/merlion.png", "Which city is this?"],
            [f"examples/kun_basketball.jpg", "Is the man good at playing basketball?"],
            [f"examples/tiananmen.jpg", "Which country this image describe?"]
        ], inputs=[imagebox, textbox])
        gr.Markdown(license_markdown)
        
        chat_list = [chatbot_A, chatbot_B]
        model_name_list = [model_name_A, model_name_B]
        state_list = [state] + chat_list
        vote_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn]
        btn_list = vote_list + [regenerate_btn, clear_btn, send_btn]
        leftvote_btn.click(vote_left_model, [state] + model_name_list, model_name_list + vote_list)
        rightvote_btn.click(vote_right_model, [state] + model_name_list, model_name_list + vote_list)
        tie_btn.click(vote_model_tie, [state] + model_name_list, model_name_list + vote_list)
        bothbad_btn.click(vote_model_bad, [state] + model_name_list, model_name_list + vote_list)
        clear_btn.click(clear_chat, [state], state_list + model_name_list + [textbox, imagebox] + btn_list)
        share_btn.click(share_click, [state], [state])
        regenerate_btn.click(run_VLP_models, state_list, state_list + btn_list)
        textbox.submit(user_ask, state_list + [textbox, imagebox], state_list + model_name_list + [textbox, imagebox] + btn_list).then(run_VLP_models, state_list, state_list + btn_list)
        send_btn.click(user_ask, state_list + [textbox, imagebox], state_list + model_name_list + [textbox, imagebox] + btn_list).then(run_VLP_models, state_list, state_list + btn_list)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--controller-url", type=str, default="http://localhost:12001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller_url = args.controller_url
    model_list = get_model_list(controller_url)
    print(f"Available model: {', '.join(model_list)}")
    assert len(model_list) >= 2, "Available model number should not smaller than 2."
    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )