#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path as osp
import os
import json
import argparse
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)

xmllabel = {
    "title":    { "long": "Title",        "level": 0, "color": (0,100,0)},
    "author":   { "long": "Author",       "level": 0, "color": (46,139,87)},
    "affili":   { "long": "Affili",       "level": 0, "color": (218,165,32)},
    "mail":     { "long": "Mail",         "level": 0, "color": (205,92,92)},
    "fnote":    { "long": "Footnote",     "level": 0, "color": (178,34,34)},
    "header":   { "long": "Header",       "level": 0, "color": (139,71,38)},
    "foot":     { "long": "Foot",         "level": 0, "color": (139,71,38)},
    # * above items not get envolved in decoder content lines
    "fig":      { "long": "Image",        "level": 0, "color": (255,20,147)},
    "tab":      { "long": "Table",        "level": 0, "color": (255,140,0)},
    "sec1":     { "long": "Section_1",    "level": 1, "color": (205,85,85)},
    "sec2":     { "long": "Section_2",    "level": 2, "color": (205,85,85)},
    "sec3":     { "long": "Section_3",    "level": 3, "color": (205,85,85)},
    "fstline":  { "long": "FirstLine",    "level": 5, "color": (255,215,0)},
    "para":     { "long": "ContentLine",  "level": 6, "color": (0,0,0)},
    "equ":      { "long": "Equation",     "level": 6, "color": (0,255,0)},
    "figcap":   { "long": "Image_Caption","level": 7, "color": (0,139,69)},
    "tabcap":   { "long": "Table_Caption","level": 7, "color": (0,191,255)},
    "opara":    { "long": "Section_Lines","level": 8, "color": (0,0,0)},
}

relation2rgb = {
    "start": '#FF0000',
    "contain": '#00FF00',
    "connect": '#000000',
    "equality": '#0000FF',
}

Meta_Classes = ["title", "author", "affili", "mail", "fnote", "foot", "header"]

def xml2json(xml_file):
    info = []
    page_id = int(xml_file.split('_')[-1].split('.')[0])
    try:
        tree = ET.parse(xml_file)
    except BaseException as e:
        logging.error("XML file {} has error when parsing!".format(xml_file))
        logging.error("Error message: {}!".format(e))
    root = tree.getroot()
    arranged_child = []
    for child in root:
        arranged_child.append([float(child.attrib['index']), child])
    arranged_child = sorted(arranged_child, key=lambda x: x[0])
    all_index = [x[0] for x in arranged_child]
    if len(set(all_index)) != len(all_index):
        logging.warning("xml_file:{} has duplicated index".format(xml_file))
    for child_id in range(arranged_child.__len__()):
        child = arranged_child[child_id][1]
        cls_ = child.attrib['labelType'].split(':')[-1]
        if 'points' in child.attrib.keys():
            x = []
            y = []
            for pt in child.attrib['points'].split(';'):
                xy = pt.replace("(", "").replace(")", "").split(",")
                if len(xy) == 2:
                    x.append(float(xy[0]))
                    y.append(float(xy[1]))
            box = [
                int(min(x)),
                int(min(y)),
                int(max(x)),
                int(max(y))
            ]
        else:
            box = [
                int(float(child.attrib['x'])),
                int(float(child.attrib['y'])),
                int(float(child.attrib['x']))+int(float(child.attrib['w'])),
                int(float(child.attrib['y']))+int(float(child.attrib['h']))
            ]
        cl = {
            "text": child.attrib['text'],
            "box": box,
            "class": cls_,
            "page": page_id,
        }
        info.append(cl)
    return info

def folder2json(xml_folder):
    def prev_low_level(pdf_info, cl_id):
        cl = pdf_info[cl_id]
        cur_cls = cl["class"]
        assert cur_cls not in ['figcap', 'tabcap', 'fig', 'tab']
        cur_level = xmllabel[cur_cls]["level"]
        # * Handling opara finding parent
        if cur_cls == "opara":
            find_ = False
            for i in range(cl_id-1, -1, -1):
                pre_cl = pdf_info[i]
                pre_cls = pre_cl["class"]
                pre_level = xmllabel[pre_cls]["level"]
                if cur_level >= pre_level: 
                    find_ = True
                    break # Until meet one equal/high_level cl, stop
            if not find_: # Regard as meta
                logging.warning("xml_folder:{} has one #{}# node that can't find parent node, return [-1, \"meta\"]".format(xml_folder, cur_cls))
                cl['is_meta'] = True
                return -1, 'meta'
            else:
                if pre_cls in ['fstline', 'para', 'equ', 'foot', 'fig', 'tab']:
                    logging.warning("xml_folder:{} page:{} has one #{}# node that find #{}# parent node, text:#{}#".format(xml_folder, cl['page'], cur_cls, pre_cls, cl['text']))
                    pdf_info[cl_id]["class"] = 'fstline'
                    cur_cls = 'fstline'
                    cur_level = xmllabel[cur_cls]["level"]
                    # If opara find parent in xxx, assert it was fstline and go on search
                else:
                    if pre_cl['is_meta']:
                        cl['is_meta'] = True
                    return i, "connect"
        # * Handling No opara finding parent
        find_ = False
        for i in range(cl_id-1, -1, -1):
            pre_cl = pdf_info[i]
            pre_cls = pre_cl["class"]
            pre_level = xmllabel[pre_cls]["level"]
            if pre_cl['is_meta']: continue # cannot find parent in Meta_Classes
            if pre_cls in ['tab', 'fig']: continue
            if cur_level >= pre_level: 
                find_ = True
                break # Until meet one equal/high_level cl, stop
        if not find_:
            if cur_cls != "sec1":
                logging.warning("xml_folder:{} has one #{}# node that can't find parent node, return [-1, \"start\"]".format(xml_folder, cur_cls))
            return -1, "contain"
        elif cur_level == pre_level: # the same level, relationship should be in ["equality", "connect"]
            if cur_cls in ["sec1", "sec2", "sec3", "secx", "fstline"]: # equality
                return i, "equality"
            if cur_cls in ["para", "equ", "opara"]: # connect
                return i, "connect"
        elif cur_level > pre_level: # different level, relationship should be in ["contain", "connect"]
            if cur_cls in ["sec1", "sec2", "sec3", "secx", "fstline"]:
                return i, "contain"
            if cur_cls in ["para", "opara", "equ"]:
                return i, "connect"
            
    def find_nearest_label_id(pdf_info, tgt_label, cur_id):
        cur_page_past = []
        i = cur_id - 1
        while i > 0:
            if pdf_info[i]['page'] == pdf_info[cur_id]['page']:
                cur_page_past.append(i)
            i -= 1
        for i in cur_page_past:
            if pdf_info[i]['class'] == tgt_label and pdf_info[i]['parent_id'] == -1:
                return i
        return -1

    def find_parent(pdf_info, cl_id):
        cl = pdf_info[cl_id]
        cls_ = cl["class"]
        if cls_ in Meta_Classes:
            cl["is_meta"] = True
            return -1, "meta"
        elif cls_ == "figcap":
            pa_id = find_nearest_label_id(pdf_info, "fig", cl_id)
            return pa_id, "contain"
        elif cls_ == "tabcap":
            pa_id = find_nearest_label_id(pdf_info, "tab", cl_id)
            return pa_id, "contain"
        elif cls_ == "fig":
            pa_id = find_nearest_label_id(pdf_info, "figcap", cl_id)
            return pa_id, "contain"
        elif cls_ == "tab":
            pa_id = find_nearest_label_id(pdf_info, "tabcap", cl_id)
            return pa_id, "contain"
        else:
            pre_cl_id, relation = prev_low_level(pdf_info, cl_id)
            return pre_cl_id, relation

    # * Merge xmls of single PDF to one json
    pdf_info = []
    all_xmls = glob.glob(os.path.join(xml_folder, "*.xml"))
    page2xml = {}
    for xml in all_xmls:
        page = int(xml.split("_")[-1].replace(".xml", ""))
        page2xml[page] = xml
    for page_id in range(len(all_xmls)):
        cur_xml = page2xml[page_id]
        json_page = xml2json(cur_xml)
        pdf_info.extend(json_page)

    # * Find parent id
    for cl_id, cl in enumerate(pdf_info):
        cl["is_meta"] = False
        parent_id, relation = find_parent(pdf_info, cl_id)
        cl["parent_id"] = parent_id
        cl["relation"]  = relation
    return pdf_info

if __name__ == "__main__":
    
    # * xml folder to json
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_folder', type=str, 
                        help='Where to store xmls')
    parser.add_argument('--exp_json_path', type=str, 
                        help='Where to save experiment json file')
    args = parser.parse_args()

    os.makedirs(args.exp_json_path, exist_ok=True)
    valid_xml_folders = [osp.join(args.xml_folder, x) for x in os.listdir(args.xml_folder) \
        if glob.glob(osp.join(args.xml_folder, x, "*.xml")).__len__() != 0]
    pbar = tqdm(valid_xml_folders)
    
    has_error = []
    for xml_folder in pbar:
        try:
            pdf_name = osp.basename(xml_folder)
            pbar.set_description("Processing {}".format(pdf_name))
            pdf_info = folder2json(xml_folder)
            json.dump(pdf_info, open(osp.join(args.exp_json_path, pdf_name+".json"), "w"), indent=4)
        except BaseException as e:
            has_error.append(xml_folder)
            logging.error("Skipping xml folder: {} ...".format(xml_folder))
            logging.error("Error Message: {}".format(e))
