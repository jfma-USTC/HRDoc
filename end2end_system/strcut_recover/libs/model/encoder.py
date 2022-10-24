import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .extractor import RoiFeatExtractor, PosFeat, VSFD
from .utils import tokenize, align_feats


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = batch_norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x, mask = x
        identity = x

        if self.stride == 2:
            mask = F.max_pool2d(mask, (3, 3), (2, 2), (1, 1))

        out = self.conv1(x)
        out = out * mask
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = out * mask
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = out * mask

        return (out, mask)


class ResNet(nn.Module):
    def __init__(self, in_dim, block, encoder_dim, layers, batch_norm=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.inplanes = in_dim
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = batch_norm(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_dim, layers[0], stride=1, dilation=2, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, encoder_dim, layers[1], stride=1, dilation=4, batch_norm=batch_norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, batch_norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, batch_norm=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                batch_norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, batch_norm=batch_norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input, input_mask):
        x = self.conv1(input)
        x = x * input_mask
        x = self.bn1(x)
        x = self.relu(x)

        c2, input_mask = self.layer1((x, input_mask))
        c3, input_mask = self.layer2((c2, input_mask))

        return c3, input_mask


class LstmFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.birnn_lr_0 = PadRNN(in_dim, hidden_dim)
        self.birnn_rl_0 = PadRNN(in_dim, hidden_dim)
    
    def forward(self, feats, feats_mask):
        
        lr_0 = self.birnn_lr_0(feats, feats_mask)
        rl_0 = self.birnn_rl_0(feats.flip(1), feats_mask.flip(1))
        
        feat = torch.cat([lr_0, rl_0.flip(1)], dim = 2)

        return feat


class PadRNN(nn.Module):
    def __init__(self, input_size, hidden_size, reverse=False):
        super(PadRNN, self).__init__()
        self.rnn = nn.GRUCell(input_size = input_size, hidden_size = hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs, masks, hidden=None):
       
        B, L, _  = inputs.shape
        if hidden == None:
            hidden = torch.zeros(B, self.hidden_size).to(inputs)

        outputs = torch.zeros((B, L, self.hidden_size)).to(inputs)
        for idx in range(L):
            hidden_new = self.rnn(inputs[:, idx, :], hidden)
            hidden = masks[:, idx][:, None] * hidden_new + (1 - masks[:, idx][:,None]) * hidden
            outputs[:, idx] = hidden

        return outputs


class SemanticsFeatExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, tokenizer, bert, transcrtips):
        # Bert Embedding
        with torch.no_grad():
            semantics_emb = []
            for transcrtips_pb in transcrtips:
                input_ids, attention_mask = tokenize(tokenizer, transcrtips_pb)
                semantics_emb.append(bert(input_ids=input_ids.to(bert.device), attention_mask=attention_mask.to(bert.device))[0][:, 0])

        semantics_emb = [self.transform(semantics_emb_pb) for semantics_emb_pb in semantics_emb]
        return semantics_emb


class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encode_layers, scale, pool_size, word_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.conv_encoder = ResNet(input_dim, BasicBlock, encoder_dim, encode_layers)
        self.lstm_encoder = LstmFusion(encoder_dim, encoder_dim//2)

        self.box_feat_extractor = RoiFeatExtractor(
            scale,
            pool_size,
            encoder_dim,
            encoder_dim
        )

        self.pos_feat = PosFeat(encoder_dim)

        self.semantics_feat_extractor = SemanticsFeatExtractor(input_dim=word_dim, output_dim=encoder_dim)
        self.fuse_gate = VSFD(encoder_dim)

    def forward(self, encoder_input, encoder_input_mask, image_size, transcrtips, encoder_input_bboxes, extractor, tokenizer, bert):
        
        with torch.no_grad():
            encoder_input = [extractor(encoder_input_pb) for encoder_input_pb in encoder_input]
            encoder_input_mask = [F.interpolate(encoder_input_mask_pb, encoder_input_pb.shape[-2:], mode='bilinear', align_corners=False) \
                for encoder_input_mask_pb, encoder_input_pb in zip(encoder_input_mask, encoder_input)]
            
        encoder_input = torch.cat(encoder_input, dim=0)
        encoder_input_mask = torch.cat(encoder_input_mask, dim=0)

        feats = torch.zeros((encoder_input.shape[0], self.encoder_dim, encoder_input.shape[2], encoder_input.shape[3])).to(encoder_input.device)

        for i in range(int(np.ceil(encoder_input.shape[0] / 5))):
            feats_conv, _ = self.conv_encoder(encoder_input[5*i: 5*(i+1)], encoder_input_mask[5*i: 5*(i+1)]) #(bs*page, C, H, W)
            feats[5*i: 5*(i+1)] = feats_conv
        
        # context feat
        context_feat = self.box_feat_extractor(feats, [box_page for box_pb in encoder_input_bboxes for box_page in box_pb])
        context_feat = list(torch.split(context_feat, [sum([len(box_page) for box_page in box_pb]) for box_pb in encoder_input_bboxes])) # [(N, 128)]

        # pos embedding
        postion_feat = self.pos_feat([torch.cat([box_pp for box_pp in box_pb], dim=0) for box_pb in encoder_input_bboxes], \
             [[list(image_size[batch_idx][page_idx]) for page_idx, box_pp in enumerate(box_pb) for _ in range(box_pp.shape[0])] for batch_idx, box_pb in enumerate(encoder_input_bboxes)])

        # semantics feat
        transcrtips = [[transcrtips_pi for transcrtips_pp in transcrtips_pb for transcrtips_pi in transcrtips_pp] for transcrtips_pb in transcrtips]
        semantics_feat = self.semantics_feat_extractor(tokenizer, bert, transcrtips)

        # fuse both context and semantics feat
        feats = self.fuse_gate(context_feat, semantics_feat, postion_feat)

        feats, feats_mask = align_feats(feats)

        lstm_feat = self.lstm_encoder(feats, feats_mask)
        
        feats = feats + lstm_feat # (B, N, C)
        
        return feats, feats_mask


def build_encoder(cfg):
    encoder = Encoder(
        input_dim=cfg.in_dim, 
        encoder_dim=cfg.encoder_dim, 
        encode_layers=cfg.encoder_layers, 
        scale=cfg.scale, 
        pool_size=cfg.pool_size, 
        word_dim=cfg.word_dim
    )
    return encoder