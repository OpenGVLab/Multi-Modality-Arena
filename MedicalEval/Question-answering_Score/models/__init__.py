import torch
import numpy as np
from PIL import Image
import sys, pathlib, os

DATA_DIR = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), 'VLP_web_data')

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
    if model_name == 'BLIP2':
        from .test_blip2 import TestBlip2
        return TestBlip2(device)
    elif model_name == 'MiniGPT-4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'mPLUG-Owl':
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'Otter':
        from .test_otter import TestOtter
        return TestOtter(device)
    elif model_name == 'Otter-Image':
        from .test_otter_image import TestOtterImage
        return TestOtterImage(device)
    elif model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device)
    elif model_name == 'VPGTrans':
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'LLaVA':
        from .test_llava import TestLLaVA
        return TestLLaVA(device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(device)
    elif 'PandaGPT' in model_name:
        from .test_pandagpt import TestPandaGPT
        return TestPandaGPT(device)
    elif 'OFv2' in model_name:
        _, version = model_name.split('_')
        from .test_OFv2 import OFv2
        return OFv2(version, device)
    elif 'LaVIN' in model_name:
        from .test_lavin import TestLaVIN
        return TestLaVIN(device)
    elif model_name == 'Lynx':
        from .test_lynx import TestLynx
        return TestLynx(device)
    elif model_name == 'Cheetah':
        from .test_cheetah import TestCheetah
        return TestCheetah(device)
    elif model_name == 'BLIVA':
        from .test_bliva import TestBLIVA
        return TestBLIVA(device)
    elif model_name == 'MIC':
        from .test_mic import TestMIC
        return TestMIC(device)
    else:
        from .test_automodel import TestAutoModel
        return TestAutoModel(model_name, device)
