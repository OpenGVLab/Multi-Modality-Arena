from .llama import load, format_prompt
from functools import partial
from torchvision import transforms


llama_dir = '/nvme/share/xupeng/llama_checkpoints'
model_path = '/nvme/share/xupeng/llama_checkpoints/7B-epoch0.pth'
load_model = partial(load, llama_dir=llama_dir)
image_transform = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
