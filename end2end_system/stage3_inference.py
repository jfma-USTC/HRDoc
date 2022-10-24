#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import json
import tqdm
import torch
from strcut_recover.libs.configs import cfg, setup_config
from strcut_recover.libs.model import build_model
from strcut_recover.libs.data import create_valid_dataloader
from strcut_recover.libs.utils import logger


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='infer')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    logger.setup_logger('Document Decoder Model', cfg.work_dir, 'train.log')
    logger.info('Use config: %s' % args.cfg)


def recover_json(cfg, ly_pred, re_pred, pa_pred, texts, bboxes):
    # extract struct feats
    def cal_struct_mask(ly, vocab):
        assert len(vocab.struct_word_ids)
        mask = ly == vocab.struct_word_ids[0]
        for i in range(1, len(vocab.struct_word_ids)):
            mask |=  ly == vocab.struct_word_ids[i]
        return mask
    def recover_parent(struct_mask, struct_id):
        s_id = 0
        s_id2tol_id = {}
        for b_id, b in enumerate(struct_mask.cpu().numpy().tolist()):
            if b:
                s_id += 1
                s_id2tol_id[s_id] = b_id
        converted = []
        for i in struct_id.cpu().numpy().tolist():
            converted.append(s_id2tol_id[i] if i!= 0 else 0)
        return torch.tensor(converted).to(struct_id)
    ly_pred_mask = cal_struct_mask(ly_pred, cfg.ly_vocab)
    pa_pred = pa_pred*torch.tril(torch.ones(pa_pred.shape[0], pa_pred.shape[1], device=pa_pred.device), diagonal=0)
    pa_pred = pa_pred-torch.triu(torch.ones(pa_pred.shape[0], pa_pred.shape[1], device=pa_pred.device)*1e8, diagonal=1)
    pa_pred_softmax = torch.softmax(pa_pred, dim=-1)
    pa_pred_argmax = torch.argmax(pa_pred_softmax, dim=-1)
    pa_pred_argmax = recover_parent(ly_pred_mask, pa_pred_argmax)
    pa_pred_extend = torch.zeros_like(ly_pred)
    pa_pred_extend=pa_pred_extend.scatter(0, torch.squeeze(torch.nonzero(ly_pred_mask), dim=-1), pa_pred_argmax)
    re_pred_extend = torch.zeros_like(ly_pred)
    re_pred_extend=re_pred_extend.scatter(0, torch.squeeze(torch.nonzero(ly_pred_mask), dim=-1), re_pred)

    ly_pred_mask_list = ly_pred_mask.to('cpu').tolist()
    pa_pred_list = pa_pred_extend.to('cpu').tolist()
    re_pred_list = re_pred_extend.to('cpu').tolist()
    ly_pred_list = ly_pred.to('cpu').tolist()
    bboxes_list = torch.cat(bboxes).to('cpu').tolist()
    texts_list = []
    for page in texts: texts_list.extend(page)
    assert len(pa_pred_list) == len(re_pred_list) == len(ly_pred_list) == len(bboxes_list) == len(texts_list)

    id2page = {}
    cur_id = 0
    for page_id, page in enumerate(texts):
        for cl in page:
            id2page[cur_id] = page_id
            cur_id += 1
    valid_json = []
    for cl_id in range(len(texts_list)):
        cls_ = cfg.ly_vocab._ids_words_map[ly_pred_list[cl_id]]
        valid_json.append({
            "text": texts_list[cl_id],
            "box": bboxes_list[cl_id],
            "class": cls_,
            "page": id2page[cl_id],
            "is_meta": True if cls_ not in cfg.ly_vocab.struct_words else False,
            "line_id": cl_id,
            "parent_id": -1 if pa_pred_list[cl_id] == 0 else pa_pred_list[cl_id],
            "relation": "meta" if not ly_pred_mask_list[cl_id] else cfg.re_vocab._ids_words_map[re_pred_list[cl_id]]
        })

    return valid_json

def valid(cfg, dataloader, model):
    model.eval()
    tokenizer = cfg.tokenizer
    extractor = cfg.extractor.to(cfg.device)
    bert = cfg.bert.to(cfg.device)
    all_predicted_result = list()
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        try:
            encoder_input = [data.to(cfg.device) for data in data_batch['encoder_input']]
            encoder_input_mask = [data.to(cfg.device) for data in data_batch['encoder_input_mask']]
            encoder_input_bboxes = [[torch.tensor(page).to(cfg.device).float() for page in data] for data in data_batch['bboxes']]
            transcripts = data_batch['transcripts']
            image_size = data_batch['image_size']
            pdf_paths = data_batch['pdfs']
            batch_lines = data_batch['lines']

            pred_result = model(encoder_input, encoder_input_mask, image_size, transcripts, \
                encoder_input_bboxes, extractor, tokenizer, bert
            )

            # post-procrssing
            ly_cls_preds, (re_cls_preds, pa_att_preds) = pred_result
            # ly_cls_preds = pred_result
            if ly_cls_preds != None and re_cls_preds != []:
                re_cls_preds = torch.stack(re_cls_preds, dim=1)
                pa_att_preds = torch.cat(pa_att_preds, dim=1)
                for batch_idx in range(len(image_size)):
                    ly_cls_preds_pb = ly_cls_preds[batch_idx]
                    transcripts_batch = transcripts[batch_idx]
                    bboxes_batch = encoder_input_bboxes[batch_idx]
                    re_cls_preds_pb = re_cls_preds[batch_idx]
                    pa_att_preds_pb = pa_att_preds[batch_idx]
                    json_data = recover_json(cfg, ly_cls_preds_pb, re_cls_preds_pb, pa_att_preds_pb, transcripts_batch, bboxes_batch)
                    all_predicted_result.append({
                        'pdf': pdf_paths[batch_idx],
                        'decoded_json': json_data
                    })
            else:
                logger.info('iter = ' + str(it) + ' no title predicted in ' + str(pdf_paths))
                pass
        except RuntimeError as E:
            if 'out of memory' in str(E):
                logger.info('iter = ' + str(it) + ' CUDA Out Of Memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info('iter = ' + str(it) + ' ' + str(E))

    with open(os.path.join(cfg.work_dir, 'pred_result.json'), 'w') as f:
        json_data = json.dumps(all_predicted_result)
        f.write(json_data)


def load_checkpoint(checkpoint, model, optimizer=None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model_param'])

def main():
    init()

    valid_dataloader = create_valid_dataloader(cfg.ly_vocab, cfg.re_vocab, cfg.valid_data_path, cfg.valid_batch_size, cfg.valid_num_workers)
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.cuda()
    
    load_checkpoint('models/best_re_acc_model.pth', model)

    with torch.no_grad():
        try:
            valid(cfg, valid_dataloader, model)
        except RuntimeError as E:
            if 'out of memory' in str(E):
                logger.info(' CUDA Out Of Memory')
            else:
                logger.info(str(E))


if __name__ == '__main__':
    def setup_seed(seed):
        import torch, numpy as np, random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(2021)
    main()
