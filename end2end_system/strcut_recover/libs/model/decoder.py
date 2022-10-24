import torch
from torch import nn


class ImageAttention(nn.Module):
    def __init__(self, key_dim, query_dim, cover_kernel):
        super().__init__()
        self.query_transform = nn.Linear(query_dim, key_dim)
        self.weight_transform = nn.Conv1d(1, key_dim, cover_kernel, 1, padding=cover_kernel//2)
        self.cum_weight_transform = nn.Conv1d(1, key_dim, cover_kernel, 1, padding=cover_kernel//2)
        self.logit_transform = nn.Conv1d(key_dim, 1, 1, 1, 0)

    def forward(self, key, key_mask, query, spatial_att_weight, cum_spatial_att_weight):
        query = self.query_transform(query)
        weight_query = self.weight_transform(spatial_att_weight)
        cum_weight_query = self.cum_weight_transform(cum_spatial_att_weight)
        fusion = key + query[:, :, None] + weight_query + cum_weight_query

        new_spatial_att_logit = self.logit_transform(torch.tanh(fusion))
        new_spatial_att_weight = new_spatial_att_logit - (1 - key_mask[:, None]) * 1e8
        bs, _, n = new_spatial_att_weight.shape
        new_spatial_att_weight = new_spatial_att_weight.reshape(bs, n)
        new_spatial_att_weight = torch.softmax(new_spatial_att_weight, dim=1).reshape(bs, 1, n)

        new_cum_spatial_att_weight = cum_spatial_att_weight + new_spatial_att_weight

        return new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight


class Decoder(nn.Module):

    def __init__(self, re_vocab, embed_dim, feat_dim, lm_state_dim, proj_dim, cover_kernel):
        super().__init__()
        self.re_vocab = re_vocab
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.lm_state_dim = lm_state_dim
        self.proj_dim = proj_dim
        self.cover_kernel = cover_kernel
        self.feat_projection = nn.Conv1d(self.feat_dim, self.proj_dim, 1, 1, 0)
        self.state_init_projection = nn.Conv1d(self.feat_dim, self.lm_state_dim, 1, 1, 0)

        self.lm_rnn1 = nn.GRUCell(input_size=self.feat_dim, hidden_size=self.lm_state_dim)
        self.lm_rnn2 = nn.GRUCell(input_size=self.feat_dim, hidden_size=self.lm_state_dim)

        self.image_attention = ImageAttention(self.proj_dim, self.feat_dim + self.lm_state_dim, cover_kernel)

        self.re_cls = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim + self.lm_state_dim, self.lm_state_dim),
            nn.Tanh(),
            nn.Linear(self.lm_state_dim, len(self.re_vocab)),
        )

    def init_state(self, feats, feats_mask):
        bs, c, n = feats.shape
        project_feats = self.feat_projection(feats) * feats_mask[:, None]

        init_state = torch.sum(self.state_init_projection(feats) * feats_mask[:, None], dim=-1) / torch.sum(feats_mask, dim=-1)[:, None]

        init_spatial_att_weight = torch.zeros([bs, 1, n], dtype=torch.float, device=feats.device)
        init_cum_spatial_att_weight = torch.zeros([bs, 1, n], dtype=torch.float, device=feats.device)

        return project_feats, init_state, init_spatial_att_weight, init_cum_spatial_att_weight

    def step(self, feats, project_feats, feats_mask, time_t, state, context, spatial_att_weight, cum_spatial_att_weight, ma_att_label=None):
        new_state = self.lm_rnn1(context, state)
        new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight = self.image_attention(
            project_feats,
            feats_mask,
            torch.cat([context, new_state], dim=1),
            spatial_att_weight,
            cum_spatial_att_weight,
        )

        context_parent = list()
        for batch_idx, ma_att_pred in enumerate(new_spatial_att_logit):
            line_idx = ma_att_pred[0, :time_t+1].argmax(-1)
            context_parent.append(feats[batch_idx, :,  line_idx])
        context_parent = torch.stack(context_parent, dim=0)

        new_state = self.lm_rnn2(context_parent, new_state)
        cls_feat = torch.cat([context, context_parent, new_state], dim=1)
        re_cls_logits_pt = self.re_cls(cls_feat)

        return re_cls_logits_pt, new_state, new_spatial_att_logit, new_spatial_att_weight, new_cum_spatial_att_weight
    
    def forward(self, feats, feats_mask, re_cls_labels=None, re_labels_mask=None, ma_att_labels=None, ma_att_masks=None):
        return self.inference(feats, feats_mask)

    def inference(self, feats, feats_mask):
        feats = torch.cat((torch.zeros((feats.shape[0], feats.shape[1], 1)).\
            to(feats), feats), dim=-1)
        feats_mask = torch.cat((torch.ones((feats_mask.shape[0], 1)).\
            to(feats_mask), feats_mask), dim=-1)
        
        max_length = feats.shape[-1] - 1
        
        project_feats, init_state, spatial_att_weight, cum_spatial_att_weight = self.init_state(feats, feats_mask)
        state = init_state

        re_cls_preds = list()
        ma_att_preds = list()

        step_mask = torch.triu(torch.ones(feats_mask.shape[1], feats_mask.shape[1]), diagonal=0).transpose(1,0).to(feats_mask)

        for time_t in range(max_length):
            re_cls_logits_pt, state, spatial_att_logit, \
                spatial_att_weight, cum_spatial_att_weight = self.step(
                        feats, project_feats, feats_mask * step_mask[None, time_t], time_t, \
                            state, feats[:, :, time_t+1], \
                                spatial_att_weight, cum_spatial_att_weight
                    )
            re_cls_preds.append(torch.argmax(re_cls_logits_pt, dim=1).detach())
            ma_att_preds.append(spatial_att_logit[:, :, :-1])

        return re_cls_preds, ma_att_preds

def build_decoder(cfg):
    decoder = Decoder(
        re_vocab=cfg.re_vocab,
        embed_dim=cfg.embed_dim,
        feat_dim=cfg.feat_dim,
        lm_state_dim=cfg.lm_state_dim,
        proj_dim=cfg.proj_dim,
        cover_kernel=cfg.cover_kernel,
    )
    return decoder