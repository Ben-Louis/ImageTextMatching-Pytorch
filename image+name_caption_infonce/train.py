import os, sys
import torch
from torch.backends import cudnn
from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint
from tqdm import tqdm
import json

from model import Model
from common.dataset import dataset
from common.models import losses

if __name__ == "__main__":
    cfg = Config.fromfile(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        cfg.pretrained_model = eval(sys.argv[2])
    cudnn.benchmark = True
    cfg.device = "cuda:0"

    save_path = os.path.join("logs", cfg.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.system(f"cp *.py {save_path}/")
    cfg.dump(os.path.join(save_path, "cfg.py"))

    dataset = dataset["train"](**cfg.dataset)
    dataloader = dataset.get_loader(**cfg.train.dataloader)
    model = Model(device=cfg.device, **cfg.model).to(cfg.device)
    loss_func = losses[cfg.loss.pop("type")](**cfg.loss)
    opt = getattr(torch.optim, cfg.train.opt.pop("type"))(model.parameters(), **cfg.train.opt)
    scheduler = getattr(torch.optim.lr_scheduler, cfg.train.scheduler.pop("type"))(opt, **cfg.train.scheduler)
    if cfg.pretrained_model > 0:
        load_checkpoint(model, os.path.join(save_path, f"{cfg.pretrained_model}.ckpt"))
        for _ in range(cfg.pretrained_model): scheduler.step()

    iter_index = tqdm(range(cfg.train.num_steps), initial=cfg.pretrained_model+1, desc="Start training...")
    dataiter = iter(dataloader)
    opt.zero_grad()
    loss_moving_avg, acc_moving_avg = 0, 0

    for step in iter_index:
        # torch.cuda.empty_cache()
        try:
            batch_data = next(dataiter)
        except:
            dataiter = iter(dataloader)
            batch_data = next(dataiter)

        x1, x2 = (batch_data["image"], batch_data["image_name"]), (batch_data["caption"],)
        if (step + 1) % 2 == 0: x1, x2 = x2, x1
        with torch.no_grad():
            feat2 = model(*x2)
        label = torch.arange(feat2.size(0), dtype=torch.long, device=feat2.device)

        batch_size = feat2.size(0) // cfg.train.grad_accum_step
        for j in range(cfg.train.grad_accum_step):
            try:
                feat1 = model.forward(*[x[j * batch_size:(j + 1) * batch_size] for x in x1])
                loss, acc = loss_func(feat1, feat2, label[j * batch_size:(j + 1) * batch_size])
            except RuntimeError as e:
                print(e)
                continue
            acc_moving_avg = acc_moving_avg * 0.9 + acc.item() * 0.1
            loss_ = loss / cfg.train.grad_accum_step
            loss_moving_avg = loss_moving_avg * 0.9 + loss.item() * 0.1
            loss_.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

        if (step + 1) % cfg.train.log_step == 0:
            iter_index.set_description(f"{cfg.name} | loss: {loss_moving_avg:.4f}, acc: {acc_moving_avg * 100:.2f}%")
        if (step + 1) % cfg.train.model_save_step == 0:
            save_checkpoint(model, os.path.join(save_path, f"{step + 1}.ckpt"), opt)


