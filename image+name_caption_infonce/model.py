import torch
import torch.nn as nn
import torch.nn.functional as F
from common.models import text_models, image_models

class Attention(nn.Module):
    def __init__(self, feat_dim, attention=None, ffn=None, last_norm=True):
        super(Attention, self).__init__()

        self.att = attention if attention else nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = ffn if ffn else nn.Sequential(nn.Linear(feat_dim, 1024), nn.ReLU(inplace=True), nn.Linear(1024, feat_dim))
        self.last_norm = nn.LayerNorm(feat_dim) if last_norm else (lambda x: x)

    def forward(self, q, k, v, attention_mask=None):
        feat = self.att(q, k, v, attention_mask)[0]
        feat = self.norm(feat + q)
        feat = self.last_norm(feat + self.ffn(feat))
        return feat

class Model(nn.Module):
    def __init__(self, device="cpu", swin_frozen_stage=3):
        super(Model, self).__init__()

        self.tokenizer, self.bert = text_models["bert"](device)
        self.image_extractor = image_models["swin"](frozen_stage=swin_frozen_stage)
        self.position_embedding = nn.Parameter(torch.randn(1, 2, 7, 7), requires_grad=True)
        self.att1 = Attention(768, nn.MultiheadAttention(embed_dim=768, num_heads=8, kdim=1026, vdim=1026))
        self.att2 = Attention(768)
        self.att3 = Attention(768, last_norm=False)
        self.norm = nn.LayerNorm(768)
        self.ffn = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 768))

        self.device = device

    def forward_text(self, text, post_ffn=False):
        token = self.tokenizer(text)
        output = self.bert(**token)
        seq_embedding = output['last_hidden_state']
        text_embedding = output['pooler_output']
        text_embedding = self.norm(text_embedding)
        if post_ffn:
            return text_embedding + self.ffn(text_embedding)
        else:
            return seq_embedding, text_embedding, (1 - token["attention_mask"]).bool()

    def forward_image(self, image, name):
        batch_size = image.size(0)
        image_embedding = self.image_extractor(image.to(self.device))
        seq_embedding, name_embedding, attention_mask = self.forward_text(name)

        image_embedding = torch.cat((image_embedding, self.position_embedding.expand(batch_size, -1, -1, -1)), dim=1)
        image_embedding = image_embedding.view(batch_size, 1026, -1).permute(2, 0, 1).contiguous()
        seq_embedding = seq_embedding.permute(1, 0, 2).contiguous()
        seq_embedding = self.att1(seq_embedding, image_embedding, image_embedding)
        seq_embedding = self.att2(seq_embedding, seq_embedding, seq_embedding, attention_mask)

        name_embedding = name_embedding.unsqueeze(0)
        feat = self.att3(name_embedding, seq_embedding, seq_embedding, attention_mask)
        return feat.squeeze(0)

    def forward(self, *args):
        if len(args) == 1:
            return self.forward_text(args[0], True)
        else:
            return self.forward_image(*args)
