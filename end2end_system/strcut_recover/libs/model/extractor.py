import torch
from torch import nn
from torchvision.ops import roi_align
from torch.nn import functional as F


def convert_to_roi_format(lines_box):
    concat_boxes = torch.cat(lines_box, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full((lines_box_pi.shape[0], 1), i, dtype=dtype, device=device)
            for i, lines_box_pi in enumerate(lines_box)
        ],
        dim=0
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class RoiFeatExtractor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim

        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, feats, lines_box):
        rois = convert_to_roi_format(lines_box)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )
        
        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.fc(lines_feat)
        return lines_feat


class PosFeatAppender(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.bbox_ln = nn.LayerNorm(output_dim)
        self.bbox_tranform = nn.Linear(6, output_dim)
        self.add_ln = nn.LayerNorm(output_dim)

    def forward(self, feats, lines_box, img_sizes):
        for idx, (line_box, img_size) in enumerate(zip(lines_box, img_sizes)):
            img_size = torch.tensor(img_size).to(line_box)
            x1 = line_box[:, 0] / img_size[:, 1] # x1
            y1 = line_box[:, 1] / img_size[:, 0] # y1
            x2 = line_box[:, 2] / img_size[:, 1] # x2
            y2 = line_box[:, 3] / img_size[:, 0] # y2
            w = (line_box[:, 2] - line_box[:, 0]) / img_size[:, 1] # w
            h = (line_box[:, 3] - line_box[:, 1]) / img_size[:, 0] # h
            input_feats = torch.stack((x1, y1, x2, y2, w, h), dim=-1)
            feats[idx] = self.add_ln(feats[idx] + self.bbox_ln(self.bbox_tranform(input_feats)))
        return list(feats)

class PosFeat(nn.Module):
    def __init__(self, output_dim):
        super().__init__()


    def forward(self, lines_box, img_sizes):
        pos_features_lst = []
        for idx, (line_box, img_size) in enumerate(zip(lines_box, img_sizes)):
            img_size = torch.tensor(img_size).to(line_box)
            x1 = line_box[:, 0] / img_size[:, 1] # x1
            y1 = line_box[:, 1] / img_size[:, 0] # y1
            x2 = line_box[:, 2] / img_size[:, 1] # x2
            y2 = line_box[:, 3] / img_size[:, 0] # y2
            # w = (line_box[:, 2] - line_box[:, 0]) / img_size[:, 1] # w
            # h = (line_box[:, 3] - line_box[:, 1]) / img_size[:, 0] # h
            w_avg = torch.mean(line_box[:, 2] - line_box[:, 0], dim=-1)
            h_avg = torch.mean(line_box[:, 3] - line_box[:, 1], dim=-1)
            w_relative = (line_box[:, 2] - line_box[:, 0]) / w_avg # w_relative
            h_relative = (line_box[:, 3] - line_box[:, 1]) / h_avg # h_relative
            last_line_box = F.pad(line_box, (0, 0, 1, 0), 'constant', 0)[:-1]
            delata_y = (line_box[:, 1] - last_line_box[:, 1]) / img_size[:, 0]
            new_col = delata_y < -0.2
            delata_y[new_col] = y1[new_col]
            next_line_box = F.pad(line_box, (0, 0, 0, 1), 'constant', 0)[1:]
            delata_y_next = (next_line_box[:, 1] - line_box[:, 1]) / img_size[:, 0]
            new_col_next = delata_y_next < -0.2
            delata_y_next[new_col_next] = 1 - y1[new_col_next]
            pos_features = torch.stack((x1, y1, x2, y2, w_relative, h_relative, 
                                        delata_y * img_size[:, 0] / h_avg, delata_y_next * img_size[:, 0] / h_avg), dim=-1)
            pos_features_lst.append(pos_features)
        return pos_features_lst

class VSFD(nn.Module):
    def __init__(self, outputp_dim=128):
        super(VSFD, self).__init__()
        self.fc0 = nn.Linear(in_features=outputp_dim * 2 + 8, out_features=outputp_dim)
        self.fc1 = nn.Linear(in_features=outputp_dim, out_features=outputp_dim)
        self.fc2 = nn.Linear(in_features=8, out_features=outputp_dim)

    def forward(self, v_feature, s_feature, p_feature):
        out_lst = []
        for batch_i, (v_f, s_f, p_f)  in enumerate(zip(v_feature, s_feature, p_feature)):
            combine_feature = torch.cat([v_f, s_f, p_f], dim=-1)
            combine_ratio = self.fc0(combine_feature)
            combine_ratio = F.sigmoid(combine_ratio)
            output = combine_ratio * v_f + (1.0 - combine_ratio) * s_f + self.fc2(p_f)
            output = self.fc1(output)
            out_lst.append(output)
        return out_lst        