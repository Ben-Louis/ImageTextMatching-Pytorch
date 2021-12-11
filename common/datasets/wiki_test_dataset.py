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

class WikiTestSet(Dataset):
    def __init__(self, data_root, use_image=True, use_name=True):
        self.data_root = data_root
        assert use_image or use_name
        self.use_image = use_image
        self.use_name = use_name
        with open(os.path.join(data_root, "samples_image.pkl"), 'rb') as f:
            self.samples_image = pickle.load(f)
        with open(os.path.join(data_root, "samples_text.pkl"), 'rb') as f:
            self.samples_text = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_three_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.set_domain("image")

    def __len__(self):
        return len(self.samples_image)

    def set_domain(self, domain="image"):
        assert domain in ("image", "text")
        self.domain = domain

    def __getitem__(self, item):
        if self.domain == "image":
            idx, image_url, path = self.samples_image[item]
            output = {"id": idx}

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

            return output

        elif self.domain == "text":
            idx, caption = self.samples_text[item]
            output = {"id": idx}
            caption = re.sub(r" \[SEP\]", r".", caption)
            caption = re.sub(r" +", r" ", caption.strip())
            if len(caption) == 0:
                caption = "random"
            output["caption"] = caption
            return output

        else:
            raise ValueError

    def get_loader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)
    
    def idx_to_caption(self, index):
        return self.samples_text[index][1]