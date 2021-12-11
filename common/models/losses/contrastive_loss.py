import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img_feature, cap_feature):
        img_feature = F.normalize(img_feature)
        cap_feature = F.normalize(cap_feature)
        dist_matrix = (img_feature.unsqueeze(1) - cap_feature.unsqueeze(0)).pow(2).sum(dim=2)
        eye = torch.eye(dist_matrix.size(0), device=dist_matrix.device).bool()
        loss_intra = dist_matrix[eye]
        dist_matrix = dist_matrix[torch.logical_not(eye)].view(img_feature.size(0), -1)
        dist_matrix = (self.margin - dist_matrix).clamp(min=0)
        loss_inter, _ = dist_matrix.max(dim=1)
        return {
            "intra": loss_intra,
            "inter": loss_inter
        }
