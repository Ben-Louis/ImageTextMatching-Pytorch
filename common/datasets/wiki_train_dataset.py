import os
import pickle
import torch
import urllib
import re
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random

def expand_to_three_channel(ts, size=224):
    if ts.size(0) == 2:
        ts = ts[:1]
    return ts[:3].float().expand(3, size, size)

class WikiTrainSet(Dataset):
    def __init__(self, data_root, use_image=True, use_name=True):
        self.data_root = data_root
        self.use_image = use_image
        self.use_name = use_name
        with open(os.path.join(data_root, "samples.pkl"), 'rb') as f:
            self.samples = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_three_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        image_url, caption, path = self.samples[item]
        output = {}

        if self.use_image:
            img_pil = Image.open(os.path.join(self.data_root, path))
            img = self.transform(img_pil)
            if hasattr(img_pil, "close"):
                img_pil.close()
            output["image"] = img

        if self.use_name:
            image_name = re.sub(r"[_\-\%\.]", r" ", urllib.parse.unquote(image_url).split('/')[-1][:-4])
            image_name = re.sub(r" +", r" ", image_name.strip())
            if len(image_name) == 0:
                image_name = "random"
            output["image_name"] = image_name

        caption = re.sub(r" \[SEP\]", r".", random.choice(caption))
        caption = re.sub(r" +", r" ", caption.strip())
        if len(caption) == 0:
            caption = "random"
        output["caption"] = caption

        return output

    def get_loader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)