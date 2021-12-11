import os
import torch
import csv
import random
import numpy
import json
from math import pi
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Image Processing
def shuffle(ts, dim=0, inv=False):
    if inv:
        idx = torch.arange(ts.size(dim) - 1, -1, step=-1, device=ts.device)
    else:
        idx = torch.randperm(ts.size(dim)).to(ts.device)
    return ts.index_select(index=idx.long(), dim=dim)


def flip(ts, dim=-1):
    return shuffle(ts, dim=dim, inv=True)


def one_hot(y, dim=7):
    y = y.view(-1, 1)
    label = torch.zeros(y.size(0), dim, device=y.device)
    return label.scatter(1, y, 1)


def ts2pil(ts):
    if ts.dim() == 4:
        assert ts.size(0) == 1
        ts = ts[0]
    if ts.min() < 0:
        ts = ts * 0.5 + 0.5
    return transforms.ToPILImage()(ts)


def pil2ts(img):
    return transforms.ToTensor()(img)  # (0, 1)


# Log
def save_log(log, config, print_items=[], summary_writer=None):
    log_path = os.path.join(config.log_path, "log.csv")
    write_header = not os.path.exists(log_path)

    with open(log_path, "a+") as f:
        f_csv = csv.DictWriter(f, sorted(log.keys()))
        if write_header:
            f_csv.writeheader()
        f_csv.writerows([log])

    if summary_writer is not None:
        for key in log:
            if not ("step" in key):
                if key == 'vector/lmk_weight':
                    fig = plt.figure()
                    x = list(range(len(log[key][0])))
                    plt.xticks(x, log[key][0], rotation=270, fontsize=3)
                    plt.plot(x, log[key][1])
                    plt.grid()
                    #plt.tight_layout()
                    #plt.savefig('lmk_weight.pdf')
                    #assert 0
                    summary_writer.add_figure(key, fig, log["step"])
                    #lmk_dict = dict(zip(log[key][0], log[key][1])) 
                    #summary_writer.add_scalars(key, lmk_dict, log["step"])
                else:
                    summary_writer.add_scalar(key, log[key], log["step"])
        
    if config.print_log:
        logg = ""
        logg += "[{}/{}] time:{:.3f}  ".format(
            log["step"], log["nsteps"],  log["time_elapse"]
        )
        if print_items:
            for items in print_items:
                for item in items:
                    logg += "{}:{:.3f}  ".format(item, log[item])
        print("\r%s" % logg, end="\n")


def save_config(config):
    path = os.path.join(config.log_path, "config.js")
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(vars(config), f)


# Model
def save_checkpoint(models, optimizors, epoch, save_path, device=torch.device("cuda:0")):
    for key, opt in optimizors.items():
        data = opt.state_dict()
        data_save_path = os.path.join(save_path, "opt-%s-%s.cpkt" % (key, epoch))
        torch.save(data, data_save_path)
        latest_link = os.path.join(save_path, "opt-%s-latest.cpkt" % key)
        if os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(data_save_path, latest_link)

    for key, model in models.items():
        if key in ("image", "text"):
            continue
        model = model.module if hasattr(model, "module") else model
        if not hasattr(model, "state_dict"):
            continue
        data = model.state_dict()
        data_save_path = os.path.join(save_path, "model-%s-%s.cpkt" % (key, epoch))
        torch.save(data, data_save_path)
        latest_link = os.path.join(save_path, "model-%s-latest.cpkt" % key)
        if os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(data_save_path, latest_link)
        # model.to(device)

    print("Save checkpoint, epoch: %s" % epoch)


def load_checkpoint(models, save_path, optimizors={}, epoch=0):
    model_marker = epoch if epoch > 0 else "latest"
    for key, model in models.items():
        if key in ("image", "text"):
            continue
        data_save_path = os.path.join(save_path, f"model-{key}-{model_marker}.cpkt")
        model = model.module if hasattr(model, "module") else model
        if not hasattr(model, "state_dict"):
            continue 
        pretrained_dict = torch.load(data_save_path)
        model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'gridstn' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    for key, opt in optimizors.items():
        data_save_path = os.path.join(save_path, "opt-%s-%s.cpkt" % (key, model_marker)) 
        opt.load_state_dict(torch.load(data_save_path))

    if model_marker == "latest":
        path = os.readlink(data_save_path)
        epoch = eval(path[:-5].rsplit("-", 1)[1])

    print("Load checkpoint, epoch: %s" % epoch)
    return epoch


# Help functions
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data(data_iter, data_loader):
    try:
        phos = next(data_iter)
    except:
        data_iter = iter(data_loader)
        phos = next(data_iter)

    return phos, data_iter


def merge_list(lst):
    res = []
    for l in lst:
        res.extend(l)
    return res


def get_GPU_info():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_gpu = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return memory_gpu


def pose2label(poses):
    poses_ = poses.clone()
    poses_[poses < -pi / 3] = 0
    poses_[poses >= -pi / 3] = 1
    poses_[poses >= -pi / 6] = 2
    poses_[poses >= pi / 6] = 3
    poses_[poses >= pi / 3] = 4
    return poses_.long()
