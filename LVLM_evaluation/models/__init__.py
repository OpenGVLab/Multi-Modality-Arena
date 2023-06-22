import torch
import numpy as np
from PIL import Image

DATA_DIR = '/nvme/share/VLP_web_data'

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
        from .test_llama_adapter_v2 import TestLLamaAdapterV2, TestLLamaAdapterV2_web
        return TestLLamaAdapterV2(device)
    elif model_name == 'Multimodal-GPT':
        from .test_multimodel_gpt import TestMultiModelGPT # Web version
        return TestMultiModelGPT(device)
    elif model_name == 'ImageBind':
        from .test_imagebind import TestImageBind
        return TestImageBind()
    elif 'LLaMA-Adapter-v3' in model_name:
        from .test_llama_adapter_v3 import TestLLamaAdapterV3
        return TestLLamaAdapterV3(model_name, device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
