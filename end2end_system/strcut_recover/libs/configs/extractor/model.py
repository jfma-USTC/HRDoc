from torch import nn
from .backbone import build_backbone
from .fpn import build_fpn
from . import config as cfg
from .utils import load_checkpoint

class Model(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.backbone = build_backbone(cfg.arch, cfg.pretrained_backbone, norm_layer=norm_layer)
        self.fpn = build_fpn(cfg.backbone_out_channels, cfg.fpn_out_channels)
        self.reduce_dim = nn.Conv2d(cfg.fpn_out_channels, cfg.reduce_channels, 3, 1, 1)

    def init_weights(self, pretrained=None):
        load_checkpoint(pretrained, self)

    def forward(self, images):
        feats = self.backbone(images) # stride = 2, 4, 8, 16
        feats = self.fpn(feats)[0] # stride=2
        feats = self.reduce_dim(feats)
        return feats
