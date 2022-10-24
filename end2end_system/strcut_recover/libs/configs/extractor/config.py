# model params
# backbone
arch = 'res34'
pretrained_backbone = True
backbone_out_channels = (64, 128, 256, 512)

# fpn
fpn_out_channels = 256

# reduce dim from fpn_out_channels
reduce_channels = 64