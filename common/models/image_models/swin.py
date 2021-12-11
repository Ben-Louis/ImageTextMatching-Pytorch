import os
import torch
from mmcls.apis import init_model

CURR_DIR = os.path.abspath(__file__).rsplit(os.sep, 1)[0]

class Swin(torch.nn.Module):
    def __init__(self, frozen_stage=-1):
        super(Swin, self).__init__()

        swin_model = init_model(
            os.path.join(CURR_DIR, "configs/swin/swin.py"),
            "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
            "cpu"
        )
        swin_model.eval()
        self.backbone = swin_model.backbone
        self.fronzen_stage = frozen_stage

        if self.fronzen_stage >= 0:
            self.backbone.patch_embed.eval()
            for i in range(self.fronzen_stage):
                self.backbone.stages[i].eval()
                norm_layer = getattr(self.backbone, f'norm{i}', None)
                if norm_layer: norm_layer.eval()

        
    def forward(self, x):
        # stage 0
        x = self.backbone.patch_embed(x)
        if self.backbone.use_abs_pos_embed:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.drop_after_pos(x)
        if self.fronzen_stage >= 0: x = x.detach()

        # stage > 0
        with torch.no_grad():
            for i in range(self.fronzen_stage):
                x = self.backbone.stages[i](x)
        for i in range(self.fronzen_stage, 4):
            x = self.backbone.stages[i](x)

        norm_layer = getattr(self.backbone, 'norm3')
        out = norm_layer(x)
        out = out.view(-1, *self.backbone.stages[3].out_resolution,
                       self.backbone.stages[3].out_channels).permute(0, 3, 1, 2).contiguous()

        if self.fronzen_stage == 4:
            out = out.detach()

        return out
