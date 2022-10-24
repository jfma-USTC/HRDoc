import os
import copy
import torch
import pickle
import numpy as np
import json
import random
import cv2
import sys
import tqdm
from ..utils.vocab import TypeVocab, RelationVocab
from torchvision.transforms import functional as F

class PickleLoader():
    
    def __init__(self, json_path, ly_vocab=TypeVocab(), re_vocab=RelationVocab(), mode='test'):
        self.json_path = json_path
        self.mode = mode
        self.info = []
        self.init()
        self.ly_vocab = ly_vocab
        self.re_vocab = re_vocab
        
    def init(self):
        jd = json.load(open(self.json_path, "r"))
        for pdf_path in jd.keys():
            anno_path = jd[pdf_path]['annotation']
            imgs_path = jd[pdf_path]['images']
            anno_jd = json.load(open(anno_path, "r"))
            lines = list()
            pages = set()
            for cl in anno_jd:
                pages.add(cl['page'])
            for page_id in sorted(list(pages)):
                lines.append([x for x in anno_jd if x['page']==page_id])
            temp_data = {
                'lines': lines, 
                'page_num': len(lines), 
                'imgs_path': imgs_path,
                'pdf_path': pdf_path
            }
            self.info.append(temp_data)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        data = self.info[idx]
        img_lst = []
        for img_path in data['imgs_path']:
            img = cv2.imread(img_path)
            img_lst.append(img)
        data['imgs'] = img_lst
        encoder_input, texts, bboxes = self.cal_items(data)
        if texts == []:
            print('texts==[] when idx =', idx, data['pdf_path'])
            return self[random.randint(0, len(self) - 1)]
        return dict(
            idx=idx,
            bboxes=bboxes,
            transcripts=texts,
            encoder_input=encoder_input,
            lines=data['lines'],
            pdf_path=data['pdf_path']
        )            

    def cal_items(self, data):
        texts, bboxes = [], []

        for page_id, lines_pg in enumerate(data['lines']):
            texts_pg = []
            bboxes_pg = []
            for line_idx, line in enumerate(lines_pg):
                bboxes_pg.append(line['box'])
                texts_pg.append(line['text'])
            texts.append(texts_pg)
            bboxes.append(bboxes_pg)

        imgs = data['imgs']

        encoder_input = list()
        for image in imgs:
            image = F.to_tensor(image)
            image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
            encoder_input.append(image)

        return encoder_input, texts, bboxes

def valid_collate_func(batch_data):
    batch_size = len(batch_data)
    assert batch_size == 1
    input_channels = batch_data[0]['encoder_input'][0].shape[0]

    max_H = max([max([page.shape[1] for page in data['encoder_input']]) for data in batch_data])
    max_W = max([max([page.shape[2] for page in data['encoder_input']]) for data in batch_data])
    max_page = max([len(data['encoder_input']) for data in batch_data])

    batch_encoder_input = []
    batch_encoder_input_mask = []
    batch_image_size = []

    batch_transcripts = []
    batch_bboxes = []
    batch_lines = []
    pdf_paths = []

    for batch_idx, data in enumerate(batch_data):
        pdf_paths.append(data['pdf_path'])
        encoder_input = torch.zeros(len(data['encoder_input']), input_channels, max_H, max_W).to(torch.float32)
        encoder_input_mask = torch.zeros(len(data['encoder_input']), 1, max_H, max_W).to(torch.float32)
        image_size = []
        
        for page_id, encoder_input_page in enumerate(data['encoder_input']):
            encoder_input[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = encoder_input_page
            encoder_input_mask[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = 1.
            image_H = encoder_input_page.shape[1]
            image_W = encoder_input_page.shape[2]
            image_size.append((image_H, image_W))

        batch_encoder_input.append(encoder_input)
        batch_encoder_input_mask.append(encoder_input_mask)
        batch_image_size.append(image_size)

        batch_transcripts.append(data['transcripts'])
        batch_bboxes.append(data['bboxes'])
        batch_lines.append(data['lines'])

    return dict(
        encoder_input=batch_encoder_input,
        encoder_input_mask=batch_encoder_input_mask,
        bboxes=batch_bboxes,
        transcripts=batch_transcripts,
        image_size=batch_image_size,
        lines=batch_lines,
        pdfs=pdf_paths
    )
