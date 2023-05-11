"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests

import torch
import uvicorn

from server_utils.constants import WORKER_HEART_BEAT_INTERVAL, LOGDIR
from server_utils.utils import build_logger, server_error_msg, pretty_print_semaphore
from peng_utils import get_model, get_device_name, generate_stream

global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_pair,
        device=None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name, self.model = model_pair
        if device is None:
            self.device, self.device_name = None, None
        elif type(device) is str:
            self.device_name = device
            self.device = torch.device(device)
        elif type(device) is torch.device:
            self.device = device
            self.device_name = get_device_name(device)
        else:
            raise ValueError("Invalid device")
        
        if args.keep_in_device:
            self.model.move_to_device(self.device)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            device = params['device'] if 'device' in params else self.device
            for output in generate_stream(
                self.model,
                params['text'],
                params['image'],
                device,
                args.keep_in_device
            ):
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=12002)
    parser.add_argument("--controller-address", type=str, default="http://localhost:12001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model-name", type=str, help="Optional display name")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--keep-in-device", action="store_true")
    args = parser.parse_args()

    logger = build_logger("model_worker", f"{LOGDIR}/model_worker_{args.model_name}.log")
    logger.info(f"args: {args}")

    device = 'cpu' if args.device == -1 else f'cuda:{args.device}'
    worker = ModelWorker(
        args.controller_address,
        f"http://{args.host}:{args.port}",
        str(uuid.uuid4())[:6],
        args.no_register,
        [args.model_name, get_model(args.model_name)],
        device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
