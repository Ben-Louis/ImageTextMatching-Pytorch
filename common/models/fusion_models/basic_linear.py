import torch
import torch.nn as nn

class BasicLinear(nn.Module):
    def __init__(self, **kwargs):
        super(BasicLinear, self).__init__()
        dim_feature = kwargs.get("dim_feature", 512)
        self.fc1 = nn.Sequential(
            nn.Linear(1024 + 768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, dim_feature),
            # nn.BatchNorm1d(dim_feature, affine=False)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, dim_feature),
            # nn.BatchNorm1d(dim_feature, affine=False)
        )
        
    def forward(self, img_feature, caption_feature, name_feature):
        img_feature = nn.functional.adaptive_avg_pool2d(img_feature, (1, 1)).squeeze(-1).squeeze(-1)
        img_feature = self.fc1(torch.cat((img_feature, name_feature), dim=1))
        cap_feature = self.fc2(caption_feature)
        return img_feature, cap_feature
