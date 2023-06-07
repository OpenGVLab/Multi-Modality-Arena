import gc
import torch
import numpy as np
from PIL import Image

DATA_DIR = '/root/VLP_web_data'

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(model_name, device=None):
    if model_name == 'blip2':
        from .test_blip2 import TestBlip2
        return TestBlip2(device)
    elif model_name == 'minigpt4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'owl':
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'otter':
        from .test_otter import TestOtter
        return TestOtter(device)
    elif model_name == 'instruct_blip':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device)
    elif model_name == 'vpgtrans':
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'llava':
        from .test_llava import TestLLaVA
        return TestLLaVA(device)
    elif model_name == 'llama_adapter_v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2, TestLLamaAdapterV2_web
        return TestLLamaAdapterV2(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")


def get_device_name(device: torch.device):
    return f"{device.type}{'' if device.index is None else ':' + str(device.index)}"


@torch.inference_mode()
def generate_stream(model, text, image, device=None, keep_in_device=False):
    image = np.array(image, dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    if device != model.device:
        model.move_to_device(device)
    output = model.generate(image, text, device, keep_in_device)
    if not keep_in_device:
        model.move_to_device(None)
    print(f"{'#' * 20} Model out: {output}")
    gc.collect()
    torch.cuda.empty_cache()
    yield output