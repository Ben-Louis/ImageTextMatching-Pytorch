import os
import re
import argparse

import torch
from PIL import Image
from torchvision import transforms
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint

from model import Model

def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/baseline.py")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image", type=str, default="example/Barack Obama by Gage Skidmore.png")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--caption", type=str, default="example/captions.txt")
    args = parser.parse_args()

    assert os.path.exists(args.config), f"{args.config} does not exists!"
    assert os.path.exists(args.image), f"{args.image} does not exists!"
    if args.filename is None:
        args.filename = args.image.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]
    if args.caption is None:
        raise ValueError('`--caption` should be either caption or path.')
    elif os.path.exists(args.caption):
        with open(args.caption, 'r') as f:
            args.caption = f.readlines()
    else:
        args.caption = args.caption.split(';')

    return args

def expand_to_three_channel(ts, size=224):
    if ts.size(0) == 2:
        ts = ts[:1]
    return ts[:3].float().expand(3, size, size)

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(expand_to_three_channel),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

def load_image(image: str):
    img_pil = Image.open(image)
    img = transform(img_pil)
    if hasattr(img_pil, "close"):
        img_pil.close()
    return img

def process_caption(caption: str):
    caption = re.sub(r" \[SEP\]", r".", caption)
    caption = re.sub(r" +", r" ", caption.strip())
    return caption

def compute_similarity(model, 
                       image: torch.FloatTensor, 
                       filename: str, 
                       caption: list):
    if image.ndim == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():
        img_feat = model.forward(image, [filename]).cpu()
        cap_feats = model.forward(caption).cpu()
        similarity = torch.nn.functional.cosine_similarity(img_feat, cap_feats, dim=1)
    
    return similarity

def inference(args):
    cfg = Config.fromfile(args.config)
    cfg.device = args.device

    # init model
    model = Model(device=cfg.device, **cfg.model).to(cfg.device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.eval()

    # load data
    image = load_image(args.image)
    filename = args.filename
    captions = list(map(process_caption, args.caption))

    # compute similarity
    similarity = compute_similarity(model, image, filename, captions)

    # display
    for i, caption in enumerate(args.caption):
        score = similarity[i]
        print(f"similarity: {score:.4f} || {caption}")

if __name__ == "__main__":
    inference(get_parameter())
