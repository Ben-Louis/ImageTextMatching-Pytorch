import os, sys
import torch
from torch.backends import cudnn
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from tqdm import tqdm
import json
import pandas as pd

from model import Model
from common.dataset import dataset


def compute_rank_result(image_idxs, image_feats, text_idxs, text_feats, dataset, topk=5):
    result = []
    text_feats = text_feats.cuda()
    image_feats = image_feats.cuda()
    with torch.no_grad():
        for i in tqdm(range(image_idxs.size(0)), desc="compute ranking"):
            similarity = torch.nn.functional.cosine_similarity(image_feats.narrow(0, i, 1), text_feats, dim=1)
            top_index = text_idxs[similarity.topk(topk)[1].cpu()]
            for j in range(top_index.size(0)):
                result.append([image_idxs[i].item(), dataset.idx_to_caption(top_index[j])])
    return result

def compute_test_tensor(cfg, dataset):
    model = Model(device=cfg.device, **cfg.model).to(cfg.device)
    load_checkpoint(model, os.path.join(cfg.save_path, f"{cfg.pretrained_model}.ckpt"))
    model.eval()

    text_feats, text_idxs = [], []
    dataset.set_domain("text")
    for batch_data in tqdm(dataset.get_loader(**cfg.test.dataloader), desc="compute text feature"):
        text_idxs.append(batch_data["id"])
        with torch.no_grad():
            text_feats.append(model.forward(batch_data["caption"]).cpu())
    text_idxs = torch.cat(text_idxs, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    image_feats, image_idxs = [], []
    dataset.set_domain("image")
    for batch_data in tqdm(dataset.get_loader(**cfg.test.dataloader), desc="compute image feature"):
        image_idxs.append(batch_data["id"])
        with torch.no_grad():
            image_feats.append(model.forward(batch_data["image"], batch_data["image_name"]).cpu())
    image_idxs = torch.cat(image_idxs, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    return {"text_idxs": text_idxs, "text_feats": text_feats, "image_feats": image_feats, "image_idxs":image_idxs}



if __name__ == "__main__":
    cfg = Config.fromfile(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        cfg.pretrained_model = eval(sys.argv[2])
    cudnn.benchmark = True
    cfg.device = "cuda:0"

    cfg.save_path = os.path.join("logs", cfg.name)
    if not os.path.exists(cfg.save_path):
        raise FileNotFoundError
    dataset = dataset["test"](**cfg.dataset)
    test_tensor_path = os.path.join(cfg.save_path, f"test_tensors_{cfg.pretrained_model}.ts")
    if not os.path.exists(test_tensor_path):
        test_tensor = compute_test_tensor(cfg, dataset)
        torch.save(test_tensor, test_tensor_path)
        torch.cuda.empty_cache()
    else:
        test_tensor = torch.load(test_tensor_path)


    result = compute_rank_result(dataset=dataset, topk=cfg.test.topk, **test_tensor)
    sub = pd.DataFrame(result, columns=['id', 'caption_title_and_reference_description'])
    sub.to_csv(f'{cfg.save_path}/submission_{cfg.pretrained_model}.csv', index=False)