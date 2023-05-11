from .test_blip2 import TestBlip2
from .test_llava import TestLLaVA
from .test_minigpt4 import TestMiniGPT4
from .test_mplug_owl import TestMplugOwl
from .test_multimodel_gpt import TestMultimodelGPT
from .test_otter import TestOtter
from .test_flamingo import TestFlamingo


import gc
import torch
import numpy as np
from PIL import Image

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_model(model_name):
    if model_name == 'blip2':
        return TestBlip2()
    elif model_name == 'minigpt4':
        return TestMiniGPT4()
    elif model_name == 'owl':
        return TestMplugOwl()
    elif model_name == 'otter':
        return TestOtter()
    elif model_name == 'flamingo':
        return TestFlamingo()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")


def get_device_name(device: torch.device):
    return f"{device.type}{'' if device.index is None else ':' + str(device.index)}"


@torch.inference_mode()
def generate_stream(model, text, image, device=None, keep_in_device=False):
    image = np.array(image, dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    output = model.generate(text, image, device, keep_in_device)
    print(f"{'#' * 20} Model out: {output}")
    gc.collect()
    torch.cuda.empty_cache()
    yield output