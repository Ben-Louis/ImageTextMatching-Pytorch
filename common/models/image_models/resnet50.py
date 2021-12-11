import os
import torch
from mmcls.apis import init_model

CURR_DIR = os.path.abspath(__file__).rsplit(os.sep, 1)[0]

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        resnet = init_model(
            os.path.join(CURR_DIR, "configs/resnet/seresnet50.py"),
            "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth",
            "cpu"
        )
        self.backbone = resnet.backbone

    def forward(self, x):
        return self.backbone(x)[0]
