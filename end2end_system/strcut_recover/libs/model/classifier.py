import torch
from torch import nn

class Classifier(nn.Module):

    def __init__(self, ly_vocab, feat_dim):
        super().__init__()
        self.ly_vocab = ly_vocab
        self.feat_dim = feat_dim

        self.ly_cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim//2),
            nn.Tanh(),
            nn.Linear(self.feat_dim//2, len(self.ly_vocab))
        )
    
    def forward(self, feats, feats_mask, ly_cls_labels=None, ly_labels_mask=None):
        ly_cls_logits = self.ly_cls(feats)
        ly_cls_preds = torch.argmax(ly_cls_logits, dim=-1).detach()

        return ly_cls_preds


def build_classifier(cfg):
    classifier = Classifier(
        ly_vocab=cfg.ly_vocab,
        feat_dim=cfg.feat_dim
    )
    return classifier