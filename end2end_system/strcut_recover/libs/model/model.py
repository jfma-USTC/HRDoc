from torch import nn
from .utils import align_feats
from .encoder import build_encoder
from .classifier import build_classifier
from .decoder import build_decoder
from .encoder import LstmFusion


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.classifier = build_classifier(cfg)
        self.title_encoder = LstmFusion(cfg.embed_dim, cfg.embed_dim // 2)
        self.decoder = build_decoder(cfg)
        
    def forward(self, encoder_input, encoder_input_mask, image_size, transcripts, encoder_input_bboxes, extractor, tokenizer, bert, ly=None, ly_mask=None, re=None, re_mask=None, pa=None, pa_mask=None):

        feats, feats_mask = self.encoder(encoder_input, encoder_input_mask, image_size, transcripts, encoder_input_bboxes, extractor, tokenizer, bert)

        result_info = dict()

        ly_cls_preds = self.classifier(feats, feats_mask, ly, ly_mask)

        def cal_struct_mask(ly, vocab):
            assert len(vocab.struct_word_ids)
            mask = ly == vocab.struct_word_ids[0]
            for i in range(1, len(vocab.struct_word_ids)):
               mask |=  ly == vocab.struct_word_ids[i]
            return mask
        if ly is not None:
            struct_mask = cal_struct_mask(ly, self.cfg.ly_vocab)
            struct_feats = [feats[batch_idx, struct_mask_pb, :] for batch_idx, struct_mask_pb in enumerate(struct_mask)]
        else:
            struct_mask = cal_struct_mask(ly_cls_preds, self.cfg.ly_vocab)
            struct_feats = [feats[batch_idx, struct_mask_pb, :] for batch_idx, struct_mask_pb in enumerate(struct_mask)]

        struct_feats, struct_feats_mask = align_feats(struct_feats)

        struct_feats = self.title_encoder(struct_feats, struct_feats_mask)

        struct_feats = struct_feats.permute(0,2,1).contiguous()
        
        de_results = self.decoder(struct_feats, struct_feats_mask)
        return ly_cls_preds, de_results

