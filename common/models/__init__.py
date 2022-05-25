from .text_models import bert_inference_gen
from .image_models import Swin, ResNet50
from .losses import ContrastiveLoss, CircleLoss, ArcInfoNCE

text_models = {'bert': bert_inference_gen}
image_models = {'swin': Swin, 'resnet': ResNet50} 
losses = {'contrastive': ContrastiveLoss, 
		  'circle': CircleLoss,
		  'arc_infonce': ArcInfoNCE}