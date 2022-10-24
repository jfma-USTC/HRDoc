import torch
from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == 4
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        for in_channels_pl in in_channels:
            self.lat_layers.append(
                nn.Conv2d(in_channels_pl, out_channels, kernel_size=1, stride=1, padding=0)
            )
            self.out_layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
            )
    
    def forward(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.lat_layers[3](c5)
        p4 = F.interpolate(p5, size=c4.shape[2:], align_corners=False, mode='bilinear') + self.lat_layers[2](c4)
        p3 = F.interpolate(p4, size=c3.shape[2:], align_corners=False, mode='bilinear') + self.lat_layers[1](c3)
        p2 = F.interpolate(p3, size=c2.shape[2:], align_corners=False, mode='bilinear') + self.lat_layers[0](c2)

        p2 = self.out_layers[0](p2)
        p3 = self.out_layers[1](p3)
        p4 = self.out_layers[2](p4)
        p5 = self.out_layers[3](p5)
        return p2, p3, p4, p5


def build_fpn(in_channels, out_channels):
    return FPN(in_channels, out_channels)
