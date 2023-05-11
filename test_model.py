import torch
import argparse
from PIL import Image
from peng_utils import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='owl')
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--image-path", type=str, default='examples/kun_basketball.jpg')
    args = parser.parse_args()

    device = torch.device('cpu' if args.device == -1 else f'cuda:{args.device}')
    tester = get_model(args.model_name)

    image = Image.open('examples/kun_basketball.jpg').convert('RGB')
    question = "Is the man good at playing basketball?"
    output = tester.generate(question, image, device)
    print(output)