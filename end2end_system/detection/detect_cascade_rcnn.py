#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tqdm
import pickle
import logging
from mmdet.apis import init_detector, inference_detector

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def init_args():
    import argparse
    parser = argparse.ArgumentParser(description = "Detect tab/img/equ on each image listed by image_list.txt, \
        the detection results are saved to detected_result.pkl")
    parser.add_argument("--img_list", default = "image_list.txt", type = str, \
        help = "The path to each test images, each line refers to one image")
    parser.add_argument("--config_file", default = "cascade_rcnn_r101_fpn_1x_paperanno.py", type = str, \
        help = "The path to config file.")
    parser.add_argument("--check_pt", default = "epoch_12.pth", type = str, \
        help = "The path to cascade-rcnn checkpoint.")
    parser.add_argument("--device", default = "cuda:0", type = str, \
        help = "The gpu id to infer on.")
    parser.add_argument("--save_path", default = "detected_result.pkl", type = str, \
        help = "The gpu id to infer on.")
    args = parser.parse_args()
    return args

def test_infer(args):
    model = init_detector(config=args.config_file, checkpoint=args.check_pt, device=args.device)
    image_list = [x.strip() for x in open(args.img_list, "r").readlines()]
    result_info = dict()
    pbar = tqdm.tqdm(image_list)
    for image in pbar:
        pbar.set_description("Process image {}".format(os.path.basename(image)))
        try:
            result = inference_detector(model, image)
            result_info[image] = dict()
            for cls_id, cls in enumerate(model.CLASSES):
                result_info[image][cls] = [[round(x) for x in result[cls_id].tolist()[i][:4]]+result[cls_id].tolist()[i][4:] \
                    for i in range(len(result[cls_id].tolist()))] 
        except BaseException as e:
            logger.error("Meet error when dealing image: {}".format(image))
            logger.error("Error message: {}".format(e))
            continue
    pickle.dump(result_info, open(args.save_path, "wb"))
    logger.info("Detecting finished, saving results to {}".format(args.save_path))

def main():
    args = init_args()
    test_infer(args)

if __name__ == "__main__":
    main()