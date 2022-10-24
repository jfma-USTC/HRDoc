#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import tqdm
import glob
import json
import argparse
import pdfplumber
import logging
import fitz
import pickle
import math
import string
from PIL import Image, ImageDraw, ImageFont
from utils import extract_pdf_line, cal_wer
from collections import Counter, defaultdict


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename='acl_papers_dealing.log',
                    filemode='a',
                    level=logging.INFO)
logger = logging.getLogger()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)

def get_pdf_paths(folder, recursive=False):
    if not os.path.exists(folder):
        return "Error: No such file or directory: {}".format(folder)
    if recursive:
        pdf_list = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
    else:
        pdf_list = glob.glob(os.path.join(folder, "*.pdf"), recursive=False)
    return pdf_list

class DocParser:
    def __init__(self, pdf_path) -> None:
        self.ALL_CLASS = { #All color from https://tool.oschina.net/commons?type=3
            "Title":{"color_rgb": (0,100,0), "short":'Title'}, #DarkGreen
            "Title_Lines":{"color_rgb": (0,0,0), "short":'TitleLine'}, #black
            "Author":{"color_rgb": (46,139,87), "short":'Author'}, #SeaGreen
            "Affili":{"color_rgb": (218,165,32), "short":'Affili'}, #goldenrod
            "Mail":{"color_rgb": (205,92,92), "short":'Mail'}, #IndianRed
            "Footnote":{"color_rgb": (178,34,34), "short":'Fnote'}, #Firebrick
            "Footnote_Lines":{"color_rgb": (0,0,0), "short":'FnoteLine'}, #black
            "Page":{"color_rgb": (139,101,8), "short":'Page'}, #DarkGoldenrod4
            "Conference":{"color_rgb": (139,71,38), "short":'Conf'}, #Sienna4
            "Image":{"color_rgb": (255,20,147), "short":'Fig'}, #DeepPink
            "Table":{"color_rgb": (255,140,0), "short":'Tab'}, #DarkOrange
            "Algorithm":{"color_rgb": (205,16,118), "short":'Alg'}, #DeepPink3
            "Image_Caption":{"color_rgb": (0,139,69), "short":'FigCap'}, #SpringGreen4
            "Image_Caption_Lines":{"color_rgb": (0,0,0), "short":'FigCapL'}, #black
            "Table_Caption":{"color_rgb": (0,191,255), "short":'TabCap'}, #DeepSkyBlue
            "Table_Caption_Lines":{"color_rgb": (0,0,0), "short":'TabCapL'}, #black
            "Equation":{"color_rgb": (0,255,0), "short":'Equ'}, #Green1
            "Equation_Label":{"color_rgb": (205,16,118), "short":'EquLabel'}, #DeepPink3
            "FirstLine":{"color_rgb": (255,215,0), "short":'FstLine'}, #Gold1
            "ContentLine":{"color_rgb": (0,0,0), "short":'Para'}, #black
            "Section":{"color_rgb": (205,85,85), "short":'Sec'}, #IndianRed3
            "Section_1":{"color_rgb": (205,85,85), "short":'Sec1'}, #IndianRed3
            "Section_2":{"color_rgb": (205,85,85), "short":'Sec2'}, #IndianRed3
            "Section_3":{"color_rgb": (205,85,85), "short":'Sec3'}, #IndianRed3
            "Section_X":{"color_rgb": (205,85,85), "short":'SecX'}, #IndianRed3
            "Section_Lines":{"color_rgb": (0,0,0), "short":'SecLine'}, #black
            "Header":{"color_rgb": (250, 66, 23), "short":'Header'},
            "Footer":{"color_rgb": (250, 66, 23), "short":'Footer'},
            "Refer":{"color_rgb": (48, 144, 0), "short":'Ref'},
            "Abstract":{"color_rgb": (250, 182, 193), "short":'Abs'},
        }
        self.color2label = self.get_color2label()
        self.pdf_path = pdf_path # Current dealing pdf file name
        self.raw_cl_path = pdf_path[:-4]+".pkl" # corresponding pkl name
        self.img_savedir = pdf_path[:-4]+'_vis' # Directory to store visualization results
        self.convert_pdf2img()
        self.pdf = pdfplumber.open(pdf_path) # An instance for pdfplumber
        self.page2img_size = self.get_page2img_size() # List of [width, height] for each page
        if os.path.exists(self.raw_cl_path):
            self.content_lines = pickle.load(open(self.raw_cl_path, "rb"))
        else:
            self.content_lines = extract_pdf_line(pdf_path) # Get text line information for each page
            self.remove_invalid_cl() # Remove those out of page border content line
            pickle.dump(self.content_lines, open(self.raw_cl_path, "wb"))

    def get_color2label(self):
        color2label = defaultdict(lambda: 'ContentLine')
        for k in self.ALL_CLASS:
            color_repr = repr(list(self.ALL_CLASS[k]["color_rgb"]))
            color2label[color_repr] = k
        color2label['[0, 0, 0]'] = 'ContentLine'
        return color2label
    
    def convert_pdf2img(self, zoom_x=1.0, zoom_y=1.0, rotate=0):
        if not os.path.exists(self.img_savedir):
            os.makedirs(self.img_savedir)
            pdfDoc = fitz.open(self.pdf_path)
            for page_index in range(pdfDoc.pageCount): # Image name start from zero
                page = pdfDoc[page_index]
                mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
                pix = page.getPixmap(matrix=mat, alpha=False)
                pix.writePNG(os.path.join(self.img_savedir, '%s.png' % (page_index)))

    def get_label_each_line(self):
        for page in self.content_lines:
            for line in page:
                color_array = [repr(x[-2]) for x in line[2]]
                most_commom_color = Counter(color_array).most_common(1)[0][0]
                label = self.color2label.get(most_commom_color)
                line.append(label)

    def max_bbox(self, boxes):
        x = [b[0] for b in boxes if b[0]>0]+[b[2] for b in boxes if b[2]>0]
        y = [b[1] for b in boxes if b[1]>0]+[b[3] for b in boxes if b[3]>0]
        if len(x) == 0 or len(y) == 0: return [-1,-1,-1,-1]
        return [min(x), min(y), max(x), max(y)]

    def rect_distance(self, bx1, bx2):
        """
        calculate distance between bx1 and bx2
        input: two box, each contain 4 elements, [x1,y1,x2,y2] for left-upper and right-below points
        return: minimum distance
        """
        x1, y1, x1b, y1b = bx1[0], bx1[1], bx1[2], bx1[3]
        x2, y2, x2b, y2b = bx2[0], bx2[1], bx2[2], bx2[3]
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return self.dist(x1, y1b, x2b, y2)
        elif left and bottom:
            return self.dist(x1, y1, x2b, y2b)
        elif bottom and right:
            return self.dist(x1b, y1, x2, y2b)
        elif right and top:
            return self.dist(x1b, y1b, x2, y2)
        elif left:
            return x1 - x2b
        elif right:
            return x2 - x1b
        elif bottom:
            return y1 - y2b
        elif top:
            return y2 - y1b
        else:  # rectangles intersect
            return 0
    
    def merge_cls(self, page_id, cl_stack, label = None, text = None):
        if len(cl_stack) == 0: return
        if label == None:
            all_label = [self.content_lines[page_id][x][-1] for x in cl_stack]
            label = Counter(all_label).most_common(1)[0][0]
        min_idx = min(cl_stack)
        max_idx = max(cl_stack)
        texts = [self.content_lines[page_id][x][0] for x in cl_stack]
        boxes = [self.content_lines[page_id][x][1] for x in cl_stack]
        chars = []
        for x in cl_stack:
            chars.extend(self.content_lines[page_id][x][2])
        new_cl = [" ".join(texts), self.max_bbox(boxes), chars, label]
        if text != None:
            new_cl[0] = text
        self.content_lines[page_id] = self.content_lines[page_id][:min_idx] + [new_cl] + self.content_lines[page_id][max_idx+1:]

    def merge_connect(self, label, specific_label = None, text = None):
        for page_id, page_cl in enumerate(self.content_lines):
            cur_page_merge_stacks = []
            cl_stack = []
            for cl_id, cl in enumerate(page_cl):
                if cl[-1] == label:
                    cl_stack.append(cl_id)
                else:
                    if len(cl_stack) != 0:
                        cur_page_merge_stacks.append(cl_stack)
                        cl_stack = []
                    
            if cl_stack.__len__() != 0:
                cur_page_merge_stacks.append(cl_stack)
            cur_page_merge_stacks.reverse()
            for cl_stack in cur_page_merge_stacks:
                self.merge_cls(page_id, cl_stack, label = label if specific_label == None else specific_label, text = text)

    def visual_labeled_cl(self):
        def put_label(img, box, color_rgb, label, left_border=20, bg_color="#000000"):
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline=color_rgb, width=1)
            lable_box = [box[0]-left_border, box[1], box[0]-3, box[1]+10]
            # lable_box = [box[0]-left_border, box[1]-10, box[0]-3, box[1]]
            draw.rectangle(lable_box, fill=bg_color, width=-1)
            font = ImageFont.truetype("msyh.ttf", size=8)
            draw.text((lable_box[0],lable_box[1]), label, font=font, fill="#FFFFFF")
        for page_id in range(self.content_lines.__len__()):
            img = Image.open(os.path.join(self.img_savedir, "{}.png".format(page_id))) # RGB format
            start_id = 1
            for cl in self.content_lines[page_id]:
                cl_class = cl[-1]
                if not isinstance(cl_class, str): # ! Note this is temp operation
                    cl_class = 'ContentLine'
                if cl_class not in self.ALL_CLASS.keys():
                    logger.warning("PDF:{} Page:{} has error in visual, cl_class:{} not in self.ALL_CLASS!".format(self.pdf_path, page_id, cl_class))
                text = self.ALL_CLASS[cl_class]["short"]
                color_rgb = self.ALL_CLASS[cl_class]["color_rgb"]
                if cl_class in ["Footnote", "Image_Caption", "Table_Caption", "FirstLine", "Section_1", "Section_2", "Section_3", "Section_X"]:
                    text = str(start_id)+" "+text
                    start_id += 1
                    put_label(img, cl[1], color_rgb, text, left_border = 30, bg_color = "#FF0000")
                else:
                    text = str(start_id)+" "+text
                    start_id += 1
                    put_label(img, cl[1], color_rgb, text)
            img.save(os.path.join(self.img_savedir, "{}_visual_labeled_cl.png".format(page_id)), "png")
        
    def get_page2img_size(self):
        page2img_size = [] # List of [width, height] for each page
        for page in self.pdf.pages:
            page2img_size.append([int(page.width), int(page.height)])
        return page2img_size

    def valid_box(self, box):
        return box[0] <= box[2] and box[1] <= box[3]

    def box_contained_in(self, small_box, big_box): # Judge whether small_box contained in big_box
        return True if small_box[0]>=big_box[0] and small_box[1]>=big_box[1] and small_box[2]<=big_box[2] and small_box[3]<=big_box[3] else False
        
    def remove_invalid_cl(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()-1, -1, -1):
                cl = self.content_lines[page_id][cl_id]
                box = self.content_lines[page_id][cl_id][1] # [x0,y0,x1,y1]
                w, h = self.page2img_size[page_id][0], self.page2img_size[page_id][1]
                if (not self.valid_box(box)) or (not self.box_contained_in(box, [0,0,w,h])) or len(cl[2]) == 0:
                    self.content_lines[page_id].pop(cl_id)

    def remove_tiny_cl(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()-1, -1, -1):
                if cl_id < 5 or self.content_lines[page_id].__len__() - cl_id < 5: continue
                cl = self.content_lines[page_id][cl_id]
                if len(cl[0]) <= 2:
                    self.content_lines[page_id].pop(cl_id)

    def same_line(self, box1, box2):
        # if 
        x0, y0, x1, y1 = box1
        x2, y2, x3, y3 = box2
        cond1 = abs((y1-y0)-(y3-y2))/max(y1-y0, y3-y2, 1) < 0.3 # similar line height
        cond2 = abs(abs(y3-y1)+abs(y2-y0))/max(y1-y0, y3-y2, 1) < 0.5 # in the same line
        if cond1 and cond2:
            return True
        return False

    def connect_same_line(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()-1, 0, -1):
                cur_cl = self.content_lines[page_id][cl_id]
                pre_cl = self.content_lines[page_id][cl_id-1]
                if self.same_line(cur_cl[1], pre_cl[1]) and cur_cl[-1] == pre_cl[-1] and self.rect_distance(cur_cl[1], pre_cl[1]) < 10:
                    self.merge_cls(page_id, [cl_id, cl_id-1], cur_cl[-1])

    def get_fst_line(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()):
                cur_cl = self.content_lines[page_id][cl_id]
                if cur_cl[-1] != "ContentLine": continue # skip those are not common line
                if len(cur_cl[0]) == 0: continue # skip those not start with uppercase
                if cur_cl[0][0] not in string.ascii_uppercase: continue # skip those not start with uppercase
                if cl_id == 0 or cl_id == self.content_lines[page_id].__len__()-1:
                    cur_cl[-1] = "FirstLine"
                else:
                    pre_cl = self.content_lines[page_id][cl_id-1]
                    next_cl = self.content_lines[page_id][cl_id+1]
                    if pre_cl[-1] not in ["ContentLine", "FirstLine"]:
                        cur_cl[-1] = "FirstLine"
                    if abs(cur_cl[1][0] - pre_cl[1][0]) > 5 and abs(cur_cl[1][0] - next_cl[1][0]) > 5:
                        cur_cl[-1] = "FirstLine"
                    if abs(cur_cl[1][1] - pre_cl[1][3]) > 5 and abs(cur_cl[1][1] - pre_cl[1][3]) > 2*abs(next_cl[1][1] - cur_cl[1][3]):
                        cur_cl[-1] = "FirstLine"
                    if abs(cur_cl[1][1] - pre_cl[1][3]) > 5 and abs(next_cl[1][1] - cur_cl[1][3]) > 5:
                        cur_cl[-1] = "FirstLine"

    def get_head_foot(self):
        for page_id in range(self.content_lines.__len__()):
            _, h = self.page2img_size[page_id][0], self.page2img_size[page_id][1]
            headbox = []
            footbox = []
            for cl_id in range(min(self.content_lines[page_id].__len__(), 5)): # determine head page
                cur_cl = self.content_lines[page_id][cl_id]
                if cur_cl[0].isdigit() and cur_cl[1][3] < h/2:
                    cur_cl[-1] = "Header"
                    headbox.append(cur_cl[1])
                if cur_cl[0].isdigit() and cur_cl[1][3] > h/2:
                    cur_cl[-1] = "Footer"
                    footbox.append(cur_cl[1])
            for cl_id in range(max(self.content_lines[page_id].__len__()-5, 0), self.content_lines[page_id].__len__()): # determine head page
                cur_cl = self.content_lines[page_id][cl_id]
                if cur_cl[0].isdigit() and cur_cl[1][3] < h/2:
                    cur_cl[-1] = "Header"
                    headbox.append(cur_cl[1])
                if cur_cl[0].isdigit() and cur_cl[1][3] > h/2:
                    cur_cl[-1] = "Footer"
                    footbox.append(cur_cl[1])
            headbox = self.max_bbox(headbox)
            footbox = self.max_bbox(footbox)
            for cl_id in range(min(self.content_lines[page_id].__len__(), 5)): # determine head page
                cur_cl = self.content_lines[page_id][cl_id]
                if self.same_line(cur_cl[1], headbox):
                    cur_cl[-1] = "Header"
                if self.same_line(cur_cl[1], footbox):
                    cur_cl[-1] = "Footer"
            for cl_id in range(max(self.content_lines[page_id].__len__()-5, 0), self.content_lines[page_id].__len__()): # determine head page
                cur_cl = self.content_lines[page_id][cl_id]
                if self.same_line(cur_cl[1], headbox):
                    cur_cl[-1] = "Header"
                if self.same_line(cur_cl[1], footbox):
                    cur_cl[-1] = "Footer"

    def insert_img(self):
        for image in self.pdf.images:
            img_box = [int(image['x0']), int(image['top']), int(image['x1']), int(image['bottom'])]
            img_page = image['page_number']-1
            inserted_pos = -1
            for cl_id in range(self.content_lines[img_page].__len__()):
                if inserted_pos >= 0: break
                cur_cl = self.content_lines[img_page][cl_id]
                cur_box = cur_cl[1]
                if cl_id == 0:
                    if cur_box[1] - img_box[3] > 0:
                        inserted_pos = 0
                elif cl_id == self.content_lines[img_page].__len__() - 1:
                    if img_box[1] - cur_box[3] > 0:
                        inserted_pos = self.content_lines[img_page].__len__() - 1
                else:
                    pre_cl = self.content_lines[img_page][cl_id-1]
                    pre_box = pre_cl[1]
                    if img_box[1] - pre_box[3] > 0 and cur_box[1] - img_box[3] > 0:
                        inserted_pos = cl_id
            if inserted_pos == -1:
                inserted_pos = 0
            image_node = ["IMAGE", img_box, [], 'Image']
            self.content_lines[img_page] = self.content_lines[img_page][:inserted_pos] \
                + [image_node] + self.content_lines[img_page][inserted_pos:]

    def renew_caption(self):
        for page_id in range(self.content_lines.__len__()):
            cl_id = 0
            while cl_id < self.content_lines[page_id].__len__():
                cur_cl = self.content_lines[page_id][cl_id]
                if cur_cl[-1] == "Table_Caption":
                    if "FIG" in cur_cl[0][:10].upper():
                        cur_cl[-1] = "Image_Caption"
                        cl_id += 1
                        if cl_id >= self.content_lines[page_id].__len__():
                            break
                        cur_cl = self.content_lines[page_id][cl_id]
                        while cur_cl[-1] == "Table_Caption":
                            cur_cl[-1] = "Image_Caption_Lines"
                            cl_id += 1
                            if cl_id >= self.content_lines[page_id].__len__():
                                break
                            cur_cl = self.content_lines[page_id][cl_id]
                    else:
                        cl_id += 1
                        if cl_id >= self.content_lines[page_id].__len__():
                            break
                        cur_cl = self.content_lines[page_id][cl_id]
                        while cur_cl[-1] == "Table_Caption":
                            cur_cl[-1] = "Table_Caption_Lines"
                            cl_id += 1
                            if cl_id >= self.content_lines[page_id].__len__():
                                break
                            cur_cl = self.content_lines[page_id][cl_id]
                cl_id += 1

    def renew_abs(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()):
                if self.content_lines[page_id][cl_id][-1] == "Abs":
                    self.content_lines[page_id][cl_id][-1] = "Section_1"

    def renew_ref(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()):
                cur_cl = self.content_lines[page_id][cl_id]
                if cur_cl[-1] != "Refer": continue
                if cal_wer(cur_cl[0].upper(), "References".upper()) < 0.4:
                    cur_cl[-1] = "Section_1"
                elif cl_id == 0 and cur_cl[0] and cur_cl[0][0] in string.ascii_uppercase or cur_cl[0][0] == '[':
                    cur_cl[-1] = "FirstLine"
                elif cl_id == self.content_lines[page_id].__len__()-1 and cur_cl[0][0] in string.ascii_uppercase or cur_cl[0][0] == '[':
                    cur_cl[-1] = "FirstLine"
                elif cl_id > 0 and cl_id < self.content_lines[page_id].__len__()-1:
                    pre_cl = self.content_lines[page_id][cl_id-1]
                    next_cl = self.content_lines[page_id][cl_id+1]
                    if abs(cur_cl[1][1] - pre_cl[1][3]) > 5:
                        cur_cl[-1] = "FirstLine"
                    elif pre_cl[1][0]-cur_cl[1][0] + next_cl[1][0]-cur_cl[1][0] > 5:
                        cur_cl[-1] = "FirstLine"
                if cur_cl[-1] == "Refer":
                    cur_cl[-1] = "ContentLine"

    def extract_all(self):
        self.get_label_each_line()
        self.merge_connect("Equation")
        self.merge_connect("Table", text = "TABLE")
        self.merge_connect("Algorithm", specific_label = "Table", text = "TABLE")
        self.remove_tiny_cl()
        self.connect_same_line()
        self.get_fst_line()
        self.get_head_foot()
        self.insert_img()
        self.renew_caption()
        self.renew_abs()
        self.renew_ref()
        # self.visual_labeled_cl()

    def save_info(self):
        cur_cls = []
        for page_id in range(len(self.content_lines)):
            cur_page_cls = []
            for cl_id in range(len(self.content_lines[page_id])):
                cl = self.content_lines[page_id][cl_id]
                if not isinstance(cl[-1], str):
                    cur_page_cls.append({"text": cl[0], "box": [int(x) for x in cl[1]], "class": 'ContentLine'})
                else:
                    add_new_cl = {"text": cl[0], "box": [int(x) for x in cl[1]], "class": cl[-1]}
                    cur_page_cls.append(add_new_cl)
            cur_cls.append(cur_page_cls)
        all_info = {
            "pdf": self.pdf_path,
            "conetent_lines": cur_cls,
        }
        pdf_base_path, pdf_name = os.path.split(self.pdf_path)
        json_path = os.path.join(pdf_base_path, pdf_name.replace('.pdf', '.json'))
        json.dump(all_info, open(json_path, 'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pdf meta infomations of Multi-Layout papers')
    parser.add_argument('--pdf_folder', type=str, 
                        default="Path To Your PDF folder",
                        help='The path storing all pdf files')
    args = parser.parse_args()

    pdf_list = get_pdf_paths(args.pdf_folder, recursive=True)
    pdf_list.sort()
    for pdf_path in tqdm.tqdm(pdf_list):
        try:
            doc_parser = DocParser(pdf_path)
            doc_parser.extract_all()
            doc_parser.save_info()
        except Exception as e:
            print(repr(e))
            continue
    print('All Done!') 