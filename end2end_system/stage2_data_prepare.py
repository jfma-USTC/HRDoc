#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tqdm
import pickle
import logging
import json
import os.path as osp
from PIL import Image, ImageDraw, ImageFont
from mmdet.apis import init_detector, inference_detector

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_detect_bbox(model, img_list):
    result_info = []
    all_img_pages = [int(osp.basename(x).split(".")[0]) for x in img_list]
    assert set(all_img_pages) == set(list(range(max(all_img_pages)+1))), \
        "Image list should be continuous and start from 0, :{}".format(str(all_img_pages))
    pbar = tqdm.tqdm(img_list)
    for image in pbar:
        pbar.set_description("Process image {}".format(osp.basename(image)))
        try:
            result = inference_detector(model, image)
            result_info_image = dict()
            for cls_id, cls in enumerate(model.CLASSES):
                result_info_image[cls] = [[round(x) for x in result[cls_id].tolist()[i][:4]]+result[cls_id].tolist()[i][4:] \
                    for i in range(len(result[cls_id].tolist()))]
            result_info.append(result_info_image)
        except BaseException as e:
            logger.error("Meet error when dealing image: {}".format(image))
            logger.error("Error message: {}".format(e))
            continue
    return result_info

def min_overlap_ratio(box1, box2):
    ''' Note that this is not the standard iou '''
    left_max = max(box1[0], box2[0])
    right_min = min(box1[2], box2[2])
    up_max = max(box1[1], box2[1])
    down_min = min(box1[3], box2[3])
    if left_max >= right_min or down_min <= up_max:
        return 0
    else:
        S1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        S2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        S_cross = (down_min-up_max)*(right_min-left_max)
        return S_cross/max(min(S1, S2), 1) # min overlap ratio
        # return S_cross/(S1+S2-S_cross) if S1+S2-S_cross != 0 else 1 # standard IOU

def insert_first_match(cur_page_cls, box, specific_text):
    ''' Find the first place the box should be inserted '''
    assert specific_text != None
    def overlap_len(min1, len1, min2, len2):
        min_ = min1
        max_ = min1 + len1
        if min1 > min2:
            min_ = min2
        if (min1 + len1) < (min2 + len2):
            max_ = min2 + len2
        return max(0, len1+len2-(max_-min_))
    need_insert_pos = -1
    for cl_id in range(len(cur_page_cls)):
        cl_box = cur_page_cls[cl_id]['box']
        overlap_len_ = overlap_len(box[0], box[2]-box[0], cl_box[0], cl_box[2]-cl_box[0])
        if box[3] < cl_box[3] and overlap_len_/max(min(box[2]-box[0], cl_box[2]-cl_box[0]), 1) > 0.7:
            need_insert_pos = cl_id
            break
    if need_insert_pos == -1:
        cur_page_cls.append({'box': box, 'page': cur_page_cls[0]['page'], 'text': specific_text})
        return cur_page_cls
    else:
        new_cls = cur_page_cls[:need_insert_pos]
        new_cls.append({'box': box, 'page': cur_page_cls[0]['page'], 'text': specific_text})
        new_cls.extend(cur_page_cls[need_insert_pos:])
        return new_cls

def renew_with_detect(cur_page_cls, bboxes, specific_text=None, must_has_match_text=False):
    if len(cur_page_cls) == 0: return cur_page_cls
    for box in bboxes:
        if box[-1] < 0.8: continue # Skip those detected box whose confidence lower than 0.8
        box = box[:4]
        meet_iou_require_id = []
        for cl_id in range(len(cur_page_cls)):
            if min_overlap_ratio(box, cur_page_cls[cl_id]['box']) > 0.6: # Find all min_overlap_ratio > 0.6
                meet_iou_require_id.append(cl_id)
        if len(meet_iou_require_id) == 0: 
            if must_has_match_text: continue # If must_has_match_text but find none, skip this detect box
            new_cls = insert_first_match(cur_page_cls, box, specific_text)
        else:
            min_id = min(meet_iou_require_id)
            all_texts = [cur_page_cls[x]["text"] for x in meet_iou_require_id]
            text = specific_text if specific_text is not None else " ".join(all_texts)
            new_cls = cur_page_cls[:min_id]
            new_cls.append({'box': box, 'page': cur_page_cls[0]['page'], 'text': text})
            for i in range(min_id, len(cur_page_cls)): # Put those box that not has iou > 0.6 overlap back
                if i in meet_iou_require_id: continue
                else: new_cls.append(cur_page_cls[i])
        cur_page_cls = new_cls
    return cur_page_cls

def gen_renewed_json(raw_json, detect_box):
    renewed_cls = []
    for page_id in range(len(detect_box)):
        cur_page_cls = [x for x in raw_json if x['page'] == page_id]
        cur_page_cls = renew_with_detect(cur_page_cls, detect_box[page_id]["equation"], must_has_match_text=True)
        cur_page_cls = renew_with_detect(cur_page_cls, detect_box[page_id]["figure"], specific_text="Figure")
        cur_page_cls = renew_with_detect(cur_page_cls, detect_box[page_id]["table"], specific_text="Table")
        renewed_cls.extend(cur_page_cls)
    return renewed_cls

def visual_cl(content_lines, img_savedir):
    def put_label(img, box, color_rgb, label, left_border=20, bg_color="#000000"):
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=color_rgb, width=1)
        lable_box = [box[0]-left_border, box[1], box[0]-3, box[1]+10]
        draw.rectangle(lable_box, fill=bg_color, width=-1)
        font = ImageFont.truetype("msyh.ttf", size=8)
        draw.text((lable_box[0],lable_box[1]), label, font=font, fill="#FFFFFF")
    all_page = set()
    for cl in content_lines: all_page.add(cl['page'])
    for page_id in all_page:
        img = Image.open(osp.join(img_savedir, "{}.png".format(page_id))) # RGB format
        cur_page_cls = [x for x in content_lines if x['page'] == page_id]
        for cl_id, cl in enumerate(cur_page_cls):
            put_label(img, cl['box'], (0,0,0), str(cl_id))
        img.save(osp.join(img_savedir, "{}_renewed.png".format(page_id)), "png")

def init_args():
    import argparse
    parser = argparse.ArgumentParser(description = "Detect tab/img/equ on images, \
        generate new annotation using the detection results and the raw content line results")
    parser.add_argument("--save_json_path_stg1", default = "pdf_parser/acl_stage1.json", type = str, \
        help = "The path to save informations about image paths/json annotations of pdfs")
    parser.add_argument("--save_json_path_stg2", default = "pdf_parser/acl_stage2.json", type = str, \
        help = "The path to save merged informations about image paths/json annotations of pdfs")
    parser.add_argument("--config_file", default = "detection/cascade_rcnn_r101_fpn_1x_paperanno.py", type = str, \
        help = "The path to cascade-rcnn config file.")
    parser.add_argument("--check_pt", default = "detection/epoch_12.pth", type = str, \
        help = "The path to cascade-rcnn checkpoint.")
    parser.add_argument("--device", default = "cuda:0", type = str, \
        help = "The gpu id to infer on.")
    args = parser.parse_args()
    return args

def main():
    args = init_args()
    model = init_detector(config=args.config_file, checkpoint=args.check_pt, device=args.device)
    json_stg1 = json.load(open(args.save_json_path_stg1, "r"))
    pbar = tqdm.tqdm(json_stg1.keys())
    for pdf_path in pbar:
        pbar.set_description("Process pdf {}".format(osp.basename(pdf_path)))
        img_list = json_stg1[pdf_path]["images"]
        if len(img_list) == 0: continue
        detect_box = get_detect_bbox(model=model, img_list=img_list)
        raw_json = json.load(open(json_stg1[pdf_path]["annotation"], "r"))
        renewed_json = gen_renewed_json(raw_json, detect_box)
        json_path_stg2 = json_stg1[pdf_path]["annotation"].replace(".raw.json", ".merged.json")
        json.dump(renewed_json, open(json_path_stg2, "w"), indent=4)
        json_stg1[pdf_path]['annotation'] = json_path_stg2
        # visual_cl(renewed_json, os.path.dirname(json_stg1[pdf_path]["images"][0]))
    json.dump(json_stg1, open(args.save_json_path_stg2, "w"), indent=4)

if __name__ == "__main__":
    main()