import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcInfoNCE(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super(ArcInfoNCE, self).__init__()
        self.margin = margin
        self.scale = scale
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def create_mask(self, target, num_classes):
        batch_size = target.size(0)
        mask = torch.zeros(batch_size, num_classes, device=target.device)
        mask.scatter_(1, target.view(batch_size, 1), 1)
        return mask.bool()

    def forward(self, feat1, feat2, y=None):
        logits = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
        if y is None and feat1.size(0) == feat2.size(0):
            y = torch.arange(0, feat1.size(0), device=feat1.device, dtype=torch.long)
        acc = (logits.argmax(dim=1) == y).float().mean()
        mask = self.create_mask(y, num_classes=feat2.size(0))
        logits[mask] -= self.margin
        return self.loss(logits * self.scale, y), acc