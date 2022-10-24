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
import math
import os.path as osp
from PIL import Image, ImageDraw, ImageFont
from utils import extract_pdf_line

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
    if not osp.exists(folder):
        return "Error: No such file or directory: {}".format(folder)
    if recursive:
        pdf_list = glob.glob(osp.join(folder, "**/*.pdf"), recursive=True)
    else:
        pdf_list = glob.glob(osp.join(folder, "*.pdf"), recursive=False)
    return pdf_list

def convert_pdf2img(pdf_path, img_savedir, zoom_x=1.0, zoom_y=1.0, rotate=0):
    raw_image_paths = []
    if not osp.exists(img_savedir):
        os.makedirs(img_savedir)
    pdfDoc = fitz.open(pdf_path)
    for page_index in range(pdfDoc.page_count): # Image name start from zero
        page = pdfDoc[page_index]
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        pix.writePNG(osp.join(img_savedir, '%s.png' % (page_index)))
        raw_image_paths.append(osp.join(img_savedir, '%s.png' % (page_index)))
    return raw_image_paths
        
def visual_cl(content_lines, img_savedir):
    def put_label(img, box, color_rgb, label, left_border=20, bg_color="#000000"):
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=color_rgb, width=1)
        lable_box = [box[0]-left_border, box[1], box[0]-3, box[1]+10]
        draw.rectangle(lable_box, fill=bg_color, width=-1)
        font = ImageFont.truetype("msyh.ttf", size=8)
        draw.text((lable_box[0],lable_box[1]), label, font=font, fill="#FFFFFF")
    for page_id in range(content_lines.__len__()):
        img = Image.open(osp.join(img_savedir, "{}.png".format(page_id))) # RGB format
        for cl_id, cl in enumerate(content_lines[page_id]):
            put_label(img, cl[1], (0,0,0), str(cl_id))
        img.save(osp.join(img_savedir, "{}_cl.png".format(page_id)), "png")

def get_page2img_size(pdf):
    page2img_size = [] # List of [width, height] for each page
    for page in pdf.pages:
        page2img_size.append([int(page.width), int(page.height)])
    return page2img_size

def find_two_column_spliter(content_lines, page2img_size):
    spliter = [] # List of [left_space_left_border, left_space_right_border, right_space_left_border, right_space_right_border]
    for page_id in range(len(content_lines)):
        boxes = [x[1] for x in content_lines[page_id]] # [x0,y0,x1,y1]
        page_width = page2img_size[page_id][0]
        mid_x = page_width/2
        l_box_ids, r_box_ids = [], []
        for i in range(len(boxes)):
            x0, x1 = boxes[i][0], boxes[i][2]
            if (x1-x0) < page_width/3: # Filter out those short text lines
                continue
            if x1<page_width/2:
                l_box_ids.append(i)
            if x0>page_width/2:
                r_box_ids.append(i)
        l_l = min([boxes[i][0] for i in l_box_ids]) if len(l_box_ids) != 0 else -1
        l_r = max([boxes[i][2] for i in l_box_ids]) if len(l_box_ids) != 0 else -1
        r_l = min([boxes[i][0] for i in r_box_ids]) if len(r_box_ids) != 0 else -1
        r_r = max([boxes[i][2] for i in r_box_ids]) if len(r_box_ids) != 0 else -1
        spliter.append([l_l, l_r, r_l, r_r])
    return spliter

def dist(x1, y1, x2, y2):
    distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    return distance

def rect_distance(bx1, bx2):
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
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
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
    
def max_bbox(boxes):
    x = [b[0] for b in boxes if b[0]>0]+[b[2] for b in boxes if b[2]>0]
    y = [b[1] for b in boxes if b[1]>0]+[b[3] for b in boxes if b[3]>0]
    if len(x) == 0 or len(y) == 0: return [-1,-1,-1,-1]
    return [min(x), min(y), max(x), max(y)]
        
def merge_cl_lines(content_lines, space_spliters):
    def overlap_len(min1, len1, min2, len2):
        min_ = min1
        max_ = min1 + len1
        if min1 > min2:
            min_ = min2
        if (min1 + len1) < (min2 + len2):
            max_ = min2 + len2
        return max(0, len1+len2-(max_-min_))
    def needs_merge(cl1, cl2, page_id, th = 0.2, ave_char_width = 5):
        overlap_l = overlap_len(cl1[1][1], cl1[1][3]-cl1[1][1], cl2[1][1], cl2[1][3]-cl2[1][1])
        if rect_distance(cl1[1], cl2[1]) < ave_char_width \
            and min(cl1[1][2]-cl1[1][0], cl2[1][2]-cl2[1][0]) < 2*ave_char_width \
                and overlap_l/max(min(cl1[1][3]-cl1[1][1], cl2[1][3]-cl2[1][1]), 1) > th: # overlap big enough
            return True
        width = page2img_size[page_id][0]
        if abs(cl1[1][0]-space_spliters[page_id][0])+abs(cl1[1][2]-space_spliters[page_id][1]) < 4*ave_char_width or abs(cl1[1][0]-space_spliters[page_id][2])+abs(cl1[1][2]-space_spliters[page_id][3]) < 4*ave_char_width:
            return False
        if abs(cl2[1][0]-space_spliters[page_id][0])+abs(cl2[1][2]-space_spliters[page_id][1]) < 4*ave_char_width or abs(cl2[1][0]-space_spliters[page_id][2])+abs(cl2[1][2]-space_spliters[page_id][3]) < 4*ave_char_width:
            return False
        if (cl1[1][2]-cl1[1][0]) > width/6 and (cl2[1][2]-cl2[1][0]) > width/6:
            return False
        if rect_distance(cl1[1][:4], cl2[1][:4]) > 5*ave_char_width:
            return False
        overlap_l = overlap_len(cl1[1][1], cl1[1][3]-cl1[1][1], cl2[1][1], cl2[1][3]-cl2[1][1])
        if overlap_l/max(min(cl1[1][3]-cl1[1][1], cl2[1][3]-cl2[1][1]), 1) > th: # overlap big enough
            return True
        return False # In same column, has overlap
    def need_merge_ids_in_groups(cur_id, id_groups):
        ids = set()
        for x in id_groups[cur_id]:
            for i in range(len(id_groups)):
                if x in id_groups[i]:
                    ids.add(i)
        return ids
    def make_unique(id_groups):
        """Make sure every groups in id_groups are unique the sort

        Args:
            id_groups ([list of [list of id]])
        Exsample:
            make_unique([[1,2,3,4], [2,3,8], [9999,10000],[3,8,20], [0,-1]]) -> [[1,2,3,4,8,20], [9999,10000], [-1,0]]
        """
        cur_id = 0
        while cur_id < len(id_groups):
            need_merge_ids = need_merge_ids_in_groups(cur_id, id_groups)
            if len(need_merge_ids) == 1: # Means id_groups[cur_id] are unique in id_groups
                cur_id += 1
            else:
                need_merge_ids = list(need_merge_ids)
                need_merge_ids.sort()
                assert need_merge_ids[0] == cur_id
                for i in range(len(need_merge_ids)-1,0,-1):
                    id_groups[cur_id].extend(id_groups[need_merge_ids[i]])
                    id_groups.pop(need_merge_ids[i])
                id_groups[cur_id] = list(set(id_groups[cur_id]))
                id_groups[cur_id].sort()
        for g in id_groups:
            g.sort()
    def which_group(x, groups):
        for gid, g in enumerate(groups):
            if x in g:
                return gid
        return None
    def cl_join(cl_s, page_id):
        if len(cl_s) == 0:
            return cl_s
        if len(cl_s) == 1:
            return cl_s[0]
        cl_x_id = [[cl_s[cl_id][1][0], cl_id] for cl_id in range(len(cl_s))]
        cl_x_id.sort(key=lambda x:x[0])
        sorted_cl_ids = [x[1] for x in cl_x_id] # in x decent range
        strings = [cl_s[cl_i][0] for cl_i in sorted_cl_ids]
        max_box = max_bbox([cl[1] for cl in cl_s])
        chars = []
        for cl_i in sorted_cl_ids:
            chars.extend(cl_s[cl_i][2])
        return [' '.join(strings), max_box, chars]

    content_lines_merged = []
    for _ in range(len(content_lines)):
        content_lines_merged.append([])
    for page_id in range(content_lines.__len__()):
        need_merge_line_id_groups = []
        RANGE = 5
        for cl_id in range(content_lines[page_id].__len__()):
            cur_group = []
            for cur_cl_id in range(max(0, cl_id-RANGE), min(len(content_lines[page_id]), cl_id+RANGE)):
                if needs_merge(content_lines[page_id][cl_id], content_lines[page_id][cur_cl_id], page_id):
                    cur_group.append(cur_cl_id)
            if len(cur_group) > 1:
                need_merge_line_id_groups.append(cur_group)

        if len(cur_group) > 1:
            need_merge_line_id_groups.append(cur_group)
        make_unique(need_merge_line_id_groups)
        all_need_to_merge_ids = []
        for group in need_merge_line_id_groups:
            all_need_to_merge_ids.extend(group)
        new_cl_this_page = []
        merged_to_new = [False]*len(need_merge_line_id_groups)
        for cl_id in range(len(content_lines[page_id])):
            if cl_id not in all_need_to_merge_ids:
                new_cl_this_page.append(content_lines[page_id][cl_id])
            else:
                group_id = which_group(cl_id, need_merge_line_id_groups)
                if not merged_to_new[group_id]:
                    cl_lists = [content_lines[page_id][i] for i in need_merge_line_id_groups[group_id]]
                    new_cl_this_page.append(cl_join(cl_lists, page_id))
                    merged_to_new[group_id] = True
        content_lines_merged[page_id] = new_cl_this_page
    return content_lines_merged

def in_which_region(page2img_size, bbox, page_id):
    page_width = page2img_size[page_id][0]
    assert bbox[0]<=bbox[2] and bbox[1]<=bbox[3]
    if bbox[2] <= page_width/2:
        return 'Left'
    if bbox[0] >= page_width/2:
        return 'Right'
    return 'Mid'

def is_in_same_paragraph(page2img_size, pre_line, next_line, pre_line_page_id, next_line_page_id, ave_char_width=5):
    """ Verify whether next_line is in the same para with pre_line """
    pre_line_box = pre_line[1]
    next_line_box = next_line[1]
    pre_line_h = pre_line_box[3]-pre_line_box[1]
    next_line_h = next_line_box[3]-next_line_box[1]
    cond1 = abs(next_line_h-pre_line_h)/max(max(pre_line_h, next_line_h), 1) < 0.5 # * Should has similar line height
    if not cond1: 
        return False
    if len(pre_line[2])==0 or len(next_line[2])==0:
        return False
    pre_in_ls  = in_which_region(page2img_size, pre_line_box, pre_line_page_id) == 'Left' # preline in left half space
    pre_in_rs  = in_which_region(page2img_size, pre_line_box, pre_line_page_id) == 'Right'
    next_in_ls = in_which_region(page2img_size, next_line_box, next_line_page_id) == 'Left'
    next_in_rs = in_which_region(page2img_size, next_line_box, next_line_page_id) == 'Right'
    cond2 = (pre_in_ls and next_in_rs and pre_line_page_id == next_line_page_id) \
            or (pre_in_rs and next_in_ls  and pre_line_page_id == next_line_page_id-1) # * Belong two different column, either in the same or different page
    cond3 = ((pre_in_ls and next_in_ls) or (pre_in_rs and next_in_rs)) and pre_line_page_id == next_line_page_id # * Same column and same page
    cond4 = next_line_box[1] - pre_line_box[3] < 0.25*(pre_line_h + next_line_h) # * Should be close in vertical direction
    cond5 = (pre_line_box[0] - next_line_box[0]) > -ave_char_width \
        and (pre_line_box[0] - next_line_box[0]) < 4*ave_char_width \
        and (pre_line_box[2] - next_line_box[2]) > -ave_char_width # * Should follow special pattern in horizontal direction
    if next_in_rs: # * Special demand for next content line, should not be first line of new para, x0 of next line and x1 of pre line should be close to space_spliters
        cond6 = abs(next_line_box[0]-space_spliters[next_line_page_id][2]) < ave_char_width and abs(pre_line_box[2]-space_spliters[pre_line_page_id][1]) < ave_char_width
    elif next_in_ls:
        cond6 = abs(next_line_box[0]-space_spliters[next_line_page_id][0]) < ave_char_width and abs(pre_line_box[2]-space_spliters[pre_line_page_id][3]) < ave_char_width
    cond7 = (pre_line_box[2]-pre_line_box[0]) > 0.6*page2img_size[pre_line_page_id][0] \
        and abs((pre_line_box[2]+pre_line_box[0])/2 - page2img_size[pre_line_page_id][0]/2) < 0.1*page2img_size[pre_line_page_id][0] # * Cross two column condition
    if cond1:
        if cond2: # * If the two belong to different column or different page
            if cond6:
                return True
        if cond3: # * If in the same half of the same PDF page
            if cond4 and cond5:
                return True
        if cond7: # * If pre_line is a cross-column text line
            if cond4 and cond5:
                return True
    return False

def font_bold(s, section_suffix = ['-Medi', 'Bold', 'BoldMT']):
    for i in section_suffix:
        if s.endswith(i):
            return True
    return False
    
def find_bold_section(content_lines, page2img_size):
    def before(p1, c1, p2, c2): # whether (p1, c1) comes befor (p2, c2)
        return p1 < p2 if p1 != p2 else c1 < c2
    def merge_cl_using_chars(chars):
        str_ = ""
        max_box = max_bbox([ch[3] for ch in chars])
        last_right_x = chars[0][3][0]
        for ch in chars:
            if ch[3][0] <= last_right_x:
                str_ = str_ + ch[0]
            else:
                str_ = str_ + ' ' + ch[0]
            last_right_x = ch[3][2]
        return [str_, max_box, chars]
    for page_id in range(len(content_lines)):
        cl_id = 0
        while cl_id < len(content_lines[page_id]):
            cl = content_lines[page_id][cl_id]
            if cl_id > 0 and is_in_same_paragraph(page2img_size, content_lines[page_id][cl_id-1], cl, page_id, page_id):
                cl_id += 1
                continue
            if not font_bold(cl[2][0][1]):
                cl_id += 1
                continue
            # * Now meets a section contained in FirstLine
            fonts_are_bold = [font_bold(x) for x in [x[1] for x in cl[2]]]
            if False not in fonts_are_bold: # * The first line is all of one section
                cl_id += 1
                cur_cl = content_lines[page_id][cl_id]
                cur_cl_fonts_are_bold = [font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                while False not in cur_cl_fonts_are_bold: # * All lines are bold, means all char are bold font
                    cl_id += 1
                    cur_cl = content_lines[page_id][cl_id]
                    cur_cl_fonts_are_bold = [font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                # * Current meet one line that is not full of bold font
                bold_font_max_id = cur_cl_fonts_are_bold.index(False)
                if bold_font_max_id == 0: # * All chars in this cl are regarded as common content line
                    cl_id += 1
                    continue
                new_cl1 = merge_cl_using_chars(cur_cl[2][:bold_font_max_id])
                new_cl2 = merge_cl_using_chars(cur_cl[2][bold_font_max_id:])
                new_merged_cls = content_lines[page_id][:cl_id]
                new_merged_cls.append(new_cl1)
                new_merged_cls.append(new_cl2)
                new_merged_cls.extend(content_lines[page_id][cl_id+1:])
                content_lines[page_id] = new_merged_cls
                cl_id += 2
            else: # * The first line contain section, but not all are section
                bold_font_max_id = fonts_are_bold.index(False)
                new_cl1 = merge_cl_using_chars(cl[2][:bold_font_max_id])
                new_cl2 = merge_cl_using_chars(cl[2][bold_font_max_id:])
                new_merged_cls = content_lines[page_id][:cl_id]
                new_merged_cls.append(new_cl1)
                new_merged_cls.append(new_cl2)
                new_merged_cls.extend(content_lines[page_id][cl_id+1:])
                content_lines[page_id] = new_merged_cls
                cl_id += 2
    return content_lines

def save_info(content_lines, json_path):
    anno_json = []
    for page_id in range(content_lines.__len__()):
        page_cl = content_lines[page_id]
        for cl_id in range(page_cl.__len__()):
            cl = page_cl[cl_id]
            anno_json.append({
                "text": cl[0],
                "box": [int(x) for x in cl[1]],
                "page": page_id
            })
    json.dump(anno_json, open(json_path, "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pdf meta infomations of ACL papers')
    parser.add_argument('--pdf_folder', type=str, 
                        default="pdf_parser/acl",
                        help='The path to all pdf files')
    parser.add_argument('--save_json_path_stg1', type=str, 
                        default="pdf_parser/acl_stage1.json",
                        help='The path to save informations about image paths/json annotations of pdfs')
    args = parser.parse_args()

    pdf_list = get_pdf_paths(args.pdf_folder, recursive=True)
    pbar = tqdm.tqdm(pdf_list)
    all_info = {}
    for pdf_path in pbar:
        pbar.set_description("Processing {}".format(osp.basename(pdf_path)))
        try:
            vis_folder = pdf_path+"_vis" if not pdf_path.endswith(".pdf") else pdf_path[:-4] + "_vis"
            raw_image_paths = convert_pdf2img(pdf_path, vis_folder)
            content_lines = extract_pdf_line(pdf_path, visual=False) # Get text line information for each page
            pdf = pdfplumber.open(pdf_path)
            page2img_size = get_page2img_size(pdf)
            space_spliters = find_two_column_spliter(content_lines, page2img_size)
            content_lines = merge_cl_lines(content_lines, space_spliters)
            content_lines = find_bold_section(content_lines, page2img_size)
            json_path = pdf_path+".raw.json" if not pdf_path.endswith(".pdf") else pdf_path[:-4] + ".raw.json"
            save_info(content_lines, json_path)
            all_info[pdf_path] = { "images": raw_image_paths, "annotation": json_path }
            # visual_cl(content_lines, vis_folder)
        except Exception as e:
            print(repr(e))
            continue
    json.dump(all_info, open(args.save_json_path_stg1, "w"), indent=4)