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
from PIL import Image, ImageDraw, ImageFont
from utils import find_lcsubstr, extract_pdf_line, cal_wer

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
    def __init__(self, pdf_path, visual=False) -> None:
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
            "Section_Lines":{"color_rgb": (0,0,0), "short":'SecLine'} #black
        }
        self.regular = {
            "Image_Caption": re.compile(r"Figure \d+[:.]"),
            "Table_Caption": re.compile(r"Table \d+[:.]"),
            "Caption": re.compile(r"(Table|Figure) \d+[:.]")
        }
        self.special_section_name = ['Abstract', 'Acknowledgments', 'References']
        self.ave_char_width = 5 # !-^-! WARNING ! Predefined Parameter, This May Cause Problem !-^-!
        self.normal_line_height = 11 # !-^-! WARNING ! Predefined Parameter, This May Cause Problem !-^-!
        self.pdf_path = pdf_path # Current dealing pdf file name
        self.img_savedir = pdf_path[:-4]+'_vis' # Directory to store visualization results
        self.visual = visual
        if self.visual:
            self.convert_pdf2img()
        self.pdf = pdfplumber.open(pdf_path) # An instance for pdfplumber
        self.page2img_size = self.get_page2img_size() # List of [width, height] for each page
        self.content_lines = extract_pdf_line(pdf_path, visual=False) # Get text line information for each page
        self.remove_invalid_cl() # Remove those out of page border content line
        self.class_lines = {} # No need to store Image and Table
        self.alg_infos = [] # BBoxes/cl_ids for Alg area
        for k in self.ALL_CLASS:
            self.class_lines[k] = []

        self.space_spliters = self.find_two_column_spliter() # ! Should be done after exclude text in table and figure
        # Element boxes, including pdf.images/curves/edges
        # The last int symbol when >=0 means the ele box belong to which img/tab id in cap_infos
        # When == -1 means the ele box is the footnote spliter
        self.ele_boxes = self.extract_ele_boxes(visual=self.visual)
        self.cap_infos = self.extract_caption() # Extract Caption region in each page

        self.content_lines_merged = [] # * This is new content_lines after merged, used to locate section/content_line
        for _ in range(len(self.content_lines)):
            self.content_lines_merged.append([])
        self.references_page_id = [-1, -1]
        self.appendix_page_id = [-1, -1]
        self.paragraph_groups = [] # [[page_id, cl_id]] for each page
        self.cl2clm_mapping = []
        self.section_hierarchy = []
        self.delta = 2 # tolerant error
        self.section_suffix = ['-Medi', 'Bold', 'BoldMT']

    def get_page2img_size(self):
        page2img_size = [] # List of [width, height] for each page
        for page in self.pdf.pages:
            page2img_size.append([int(page.width), int(page.height)])
        return page2img_size

    def remove_invalid_cl(self):
        for page_id in range(self.content_lines.__len__()):
            for cl_id in range(self.content_lines[page_id].__len__()-1, -1, -1):
                cl = self.content_lines[page_id][cl_id]
                box = self.content_lines[page_id][cl_id][1] # [x0,y0,x1,y1]
                w, h = self.page2img_size[page_id][0], self.page2img_size[page_id][1]
                if (not self.valid_box(box)) or (not self.box_contained_in(box, [0,0,w,h])) or len(cl[2]) == 0:
                    self.content_lines[page_id].pop(cl_id)

    def find_two_column_spliter(self):
        spliter = [] # List of [left_space_left_border, left_space_right_border, right_space_left_border, right_space_right_border]
        for page_id in range(len(self.content_lines)):
            boxes = [x[1] for x in self.content_lines[page_id]] # [x0,y0,x1,y1]
            page_width = self.page2img_size[page_id][0]
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

    def extract_ele_boxes(self, visual=False):
        def is_valid_ele(ele):
            if ele['stroke']:
                return True
            elif isinstance(ele['non_stroking_color'], tuple):
                if sum(ele['non_stroking_color']) != 0:
                    return True
            return False
        ele_boxes = []
        for i in range(self.content_lines.__len__()):
            ele_boxes.append([])
        for img in self.pdf.images:
            page_id = img["page_number"]-1
            bbox = [int(img['x0']), int(img['top']), int(img['x1']), int(img['bottom']), 'img']
            ele_boxes[page_id].append(bbox)
        for curv in self.pdf.curves:
            if not is_valid_ele(curv): # Simplified condition
                continue
            page_id = curv["page_number"]-1
            points_x = [int(i[0]) for i in curv["points"]]
            points_y = [int(i[1]) for i in curv["points"]]
            if len(points_x) == 0 or len(points_y) == 0:
                continue
            bbox = [min(points_x), min(points_y), max(points_x), max(points_y), 'curv']
            ele_boxes[page_id].append(bbox)
        for edge_id, edge in enumerate(self.pdf.edges):
            page_id = edge["page_number"]-1
            # Exclude those non-visible edge, exsample in '2020.acl-main.340.pdf' page 8
            # if not edge['stroke'] or (edge['stroking_color'] in [None, 0, [0]] and edge['non_stroking_color'] in [(0,0,0,0), [0], 0]:
            if not is_valid_ele(edge): # Simplified condition
                continue
            bbox = [int(edge['x0']), int(edge['top']), int(edge['x1']), int(edge['bottom']), 'edge']
            x_, y_ = [bbox[0], bbox[2]], [bbox[1], bbox[3]]
            w, h = self.page2img_size[page_id][0], self.page2img_size[page_id][1]
            if min(x_) < 0 or max(x_) > w or min(y_) < 0 or max(y_) > h:
                logger.debug("PDF:{} Page_id:{} has edge out of pdf border. Edge_id:{}, Edge_box:{}, PDF_border:{}".format(self.pdf_path, page_id, edge_id, str(bbox), str([0,0,w,h]))) # Detect those out of border edge
            ele_boxes[page_id].append(bbox)

        if visual:
            ELE_COLOR_RGB = {
                'curv': (255,0,0),   #Red
                'edge': (0,255,0),   #Green
                'img':  (0,0,255),   #Blue
                'rect': (0,255,255), #Cyan
            }
            for page_id in range(len(ele_boxes)):
                img = Image.open(os.path.join(self.img_savedir, "{}.png".format(page_id))) # RGB format
                draw = ImageDraw.Draw(img)
                for ele_box in ele_boxes[page_id]:
                    box = ele_box[:4]
                    cls_ = ele_box[4]
                    draw.rectangle(box, outline=ELE_COLOR_RGB[cls_], width=1)
                img.save(os.path.join(self.img_savedir, "{}_ele_boxes.png".format(page_id)), "png")
        return ele_boxes

    def extract_all(self):
        self.extract_first_page_top_meta_info() # * Extract title/author/mail/affiliation,  [cl_boxes] classified
        self.extract_page_conf()                # * Extract page/conference,                [cl_boxes] classified
        self.extract_image_table()              # * Extract image/table, link to caption,   [ele_boxes]/[cl_boxes] classified
        self.extract_remained_ele()             # * Extract no-label ele, such as Alg.,     [ele_boxes]/[cl_boxes] classified
        self.extract_footnote()                 # * Extract footnote,                       [ele_boxes]/[cl_boxes] classified
        self.merge_cl_lines()                   # ! ----------Merge contents in same line.----------------------------------!
        self.modify_cl2clm()                    # ! ----------Use self.content_lines_merged in following function.----------!
        self.extract_section()                  # * Extract sections,                       [cl_boxes] classified
        self.extract_reference()                # * Extract references,                     [cl_boxes] classified
        self.extract_paragraph()                # * Extract paragraph in main body,         [cl_boxes] classified, Add Equation classify in 2022-5-17
        self.modify_first_line()                # * Extract fst line of title/footnote/image_caption/table_caption,  [cl_boxes] renewed
        self.find_bold_section()                # * Extract bold-text as section,           [cl_boxes] renewed
        if self.visual:
            self.visual_labeled_cl()            # * Visualization
        pass

    def dist(self, x1, y1, x2, y2):
        distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        return distance

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

    def rgb2bgr(self, rgb):
        return (rgb[2], rgb[1], rgb[0])
    
    def box_contained_in(self, small_box, big_box): # Judge whether small_box contained in big_box
        return True if small_box[0]>=big_box[0] and small_box[1]>=big_box[1] and small_box[2]<=big_box[2] and small_box[3]<=big_box[3] else False
    
    def box_overlap(self, bx1, bx2): # Judge whether box1 has overlap with box2
        x1, y1, x1b, y1b = bx1[0], bx1[1], bx1[2], bx1[3]
        x2, y2, x2b, y2b = bx2[0], bx2[1], bx2[2], bx2[3]
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return False
        elif left and bottom:
            return False
        elif bottom and right:
            return False
        elif right and top:
            return False
        elif left:
            return False
        elif right:
            return False
        elif bottom:
            return False
        elif top:
            return False
        else:  # rectangles intersect
            return True
    
    def valid_box(self, box):
        return box[0] <= box[2] and box[1] <= box[3]
            
    def in_which_region(self, bbox, page_id):
        page_width = self.page2img_size[page_id][0]
        assert bbox[0]<=bbox[2] and bbox[1]<=bbox[3]
        if bbox[2] <= page_width/2:
            return 'Left'
        if bbox[0] >= page_width/2:
            return 'Right'
        return 'Mid'

    def max_bbox(self, boxes):
        x = [b[0] for b in boxes if b[0]>0]+[b[2] for b in boxes if b[2]>0]
        y = [b[1] for b in boxes if b[1]>0]+[b[3] for b in boxes if b[3]>0]
        if len(x) == 0 or len(y) == 0: return [-1,-1,-1,-1]
        return [min(x), min(y), max(x), max(y)]
    
    def convert_pdf2img(self, zoom_x=1.0, zoom_y=1.0, rotate=0):
        if not os.path.exists(self.img_savedir):
            # logger.info("Making dir: {}".format(self.img_savedir))
            os.makedirs(self.img_savedir)
        pdfDoc = fitz.open(self.pdf_path)
        for page_index in range(pdfDoc.pageCount): # Image name start from zero
            page = pdfDoc[page_index]
            mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            pix = page.getPixmap(matrix=mat, alpha=False)
            pix.writePNG(os.path.join(self.img_savedir, '%s.png' % (page_index)))

    def extract_first_page_top_meta_info(self):
        """Extract `title/author/mail/affiliation`

        Args:
            content_lines (list): all content lines obtained from `extract_pdf_line`
            metadata (dict):  pdfplumber.open("pdf file").metadata
        """
        def contain_author(text, authors):
            contain = False
            for author in authors:
                if contain:
                    return contain
                if author.replace(" ","").lower() in text.replace(" ", "").lower(): #contain one author name
                    contain = True
                # if find_lcsubstr(text, author)
                if '.' in text and cal_wer(author, text) < 0.2:
                    contain = True
            return contain

        fst_page = self.content_lines[0]
        # Find the "Abstract section"
        abs_idx = 0
        while abs_idx < len(fst_page) and fst_page[abs_idx][0].strip() != "Abstract":
            abs_idx += 1
        if fst_page[abs_idx][0].strip() != "Abstract":
            logger.error("PDF:{} has find no Abstract".format(self.pdf_path))
        # All lines above "Abstract" is top head meta infos, including title\author\mail\affiliation
        top_meta_idx = [i for i in range(len(fst_page)) if fst_page[i][1][3] < fst_page[abs_idx][1][1]]
        title = self.pdf.metadata['Title']
        authors = [i.strip() for i in self.pdf.metadata['Author'].split(";")]

        if len(top_meta_idx) > 20:
            logger.warning("PDF:{} has more than 20 top info text lines (Title/Author/Affili), should check Abstract location:{}".format(self.pdf_path, str(fst_page[abs_idx][1])))

        for idx in top_meta_idx: # Give class to all top_meta_idx
            text = fst_page[idx][0]
            lcs, lcs_len = find_lcsubstr(text, title)
            if len(text) > 10 and (lcs_len > 20 or lcs_len/len(text) > 0.8):
                fst_page[idx].append("Title")
                self.class_lines["Title"].append(fst_page[idx])
            elif '@' in text:
                fst_page[idx].append("Mail")
                self.class_lines["Mail"].append(fst_page[idx])
            elif contain_author(text, authors):
                fst_page[idx].append("Author")
                self.class_lines["Author"].append(fst_page[idx])
            else:
                fst_page[idx].append("Affili")
                self.class_lines["Affili"].append(fst_page[idx])

    def extract_page_conf(self):
        """Extract `page/conference`

        Args:
            content_lines (list): all content lines obtained from `extract_pdf_line`
            page2img_size (list): list of [width, height] for each page
        """
        # Special rule only for ACL, for page/conference
        has_page = [False]*len(self.content_lines)
        Top_Num = 10
        for page_idx, page in enumerate(self.content_lines):
            cl_id_top_y = [[i, page[i][1][1]] for i in range(len(page))]
            cl_id_top_y.sort(key=lambda x:-x[1]) # Sort from bottom to top as y-axis
            page_entity_id = -1
            width, height = self.page2img_size[page_idx][0], self.page2img_size[page_idx][1]
            for i in range(Top_Num):
                i_top = cl_id_top_y[i][0] # cl id in page
                if has_page[page_idx]:
                    break # jump out of inner loop, continue outer loop
                if page[i_top][0].isdigit() and page[i_top][1][1] > 0.8*height and (page[i_top][1][2]-page[i_top][1][0])<0.1* width and page[i_top][1][0] > 0.4* width:
                    has_page[page_idx] = True
                    page_entity_id = i_top
                    page[i_top].append("Page") # found Page entity
                    self.class_lines["Page"].append(page[i_top])
            if page_idx == 0 and has_page[page_idx]:
                below_page_idx = [i for i in range(Top_Num) if page[i][1][1] > page[page_entity_id][1][3]]
                for i in below_page_idx:
                    page[i].append("Conference")  # found Conference entity
                    self.class_lines["Conference"].append(page[i])
        if sum(has_page) == len(has_page):
            return
        else:
            logger.warning("{} has no page entity in page {}!".format(self.pdf_path, str([i+1 for i in range(len(has_page)) if not has_page[i]])))
        # TODO: General rule for ACL

    def find_ele_box_in_area(self, area, page_id): # should exclude those content_lines that already has label; has overlap or in
        in_area_ele_ids = [ele_id for ele_id in range(self.ele_boxes[page_id].__len__()) \
            # if self.box_overlap(self.ele_boxes[page_id][ele_id][:4], area) and isinstance(self.ele_boxes[page_id][ele_id][-1], str)]
            if self.box_contained_in(self.ele_boxes[page_id][ele_id][:4], area) and isinstance(self.ele_boxes[page_id][ele_id][-1], str)]
        bboxes = [self.ele_boxes[page_id][ele_id][:4] for ele_id in in_area_ele_ids] 
        return in_area_ele_ids, self.max_bbox(bboxes)

    def find_cl_in_area(self, area, page_id): # should exclude those ele_box that already has corrosponding caption; has overlap or in
        in_area_cls = [cl for cl in range(self.content_lines[page_id].__len__()) \
            # if self.box_overlap(self.content_lines[page_id][cl][1], area) and isinstance(self.content_lines[page_id][cl][-1], list)]
            if self.box_contained_in(self.content_lines[page_id][cl][1], area) and isinstance(self.content_lines[page_id][cl][-1], list)]
        bboxes = [self.content_lines[page_id][cl][1] for cl in in_area_cls] 
        return in_area_cls, self.max_bbox(bboxes)

    def is_in_same_paragraph(self, pre_line, next_line, pre_line_page_id, next_line_page_id):
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
        # top_5_char_width = [c[3][2]-c[3][0] for c in \
        #     sample((pre_line[2]+next_line[2]), min(10, len(pre_line[2]+next_line[2])))] # choose at most 5 chars from all chars in pre_line and next_line
        # ave_char_width = sum(top_5_char_width)/len(top_5_char_width)
        pre_in_ls  = self.in_which_region(pre_line_box, pre_line_page_id) == 'Left' # preline in left half space
        pre_in_rs  = self.in_which_region(pre_line_box, pre_line_page_id) == 'Right'
        next_in_ls = self.in_which_region(next_line_box, next_line_page_id) == 'Left'
        next_in_rs = self.in_which_region(next_line_box, next_line_page_id) == 'Right'
        cond2 = (pre_in_ls and next_in_rs and pre_line_page_id == next_line_page_id) \
             or (pre_in_rs and next_in_ls  and pre_line_page_id == next_line_page_id-1) # * Belong two different column, either in the same or different page
        cond3 = ((pre_in_ls and next_in_ls) or (pre_in_rs and next_in_rs)) and pre_line_page_id == next_line_page_id # * Same column and same page
        cond4 = next_line_box[1] - pre_line_box[3] < 0.25*(pre_line_h + next_line_h) # * Should be close in vertical direction
        cond5 = (pre_line_box[0] - next_line_box[0]) > -self.ave_char_width \
            and (pre_line_box[0] - next_line_box[0]) < 4*self.ave_char_width \
            and (pre_line_box[2] - next_line_box[2]) > -self.ave_char_width # * Should follow special pattern in horizontal direction
        # TODO: Need special judge for shorter paragraph, such as Abstract/Reference
        if next_in_rs: # * Special demand for next content line, should not be first line of new para, x0 of next line and x1 of pre line should be close to space_spliters
            cond6 = abs(next_line_box[0]-self.space_spliters[next_line_page_id][2]) < self.ave_char_width and abs(pre_line_box[2]-self.space_spliters[pre_line_page_id][1]) < self.ave_char_width
        elif next_in_ls:
            cond6 = abs(next_line_box[0]-self.space_spliters[next_line_page_id][0]) < self.ave_char_width and abs(pre_line_box[2]-self.space_spliters[pre_line_page_id][3]) < self.ave_char_width
        cond7 = (pre_line_box[2]-pre_line_box[0]) > 0.6*self.page2img_size[pre_line_page_id][0] \
            and abs((pre_line_box[2]+pre_line_box[0])/2 - self.page2img_size[pre_line_page_id][0]/2) < 0.1*self.page2img_size[pre_line_page_id][0] # * Cross two column condition
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

    def extract_caption(self):
        """ Extract caption region, including figure caption and table caption """
        cap_infos = []
        for page_id in range(self.content_lines.__len__()): # for each page
            caption_start_ids = [x for x in range(self.content_lines[page_id].__len__()) if self.regular['Caption'].match(self.content_lines[page_id][x][0]) != None]
            caption_start_ids = [x for x in caption_start_ids \
                if not (x > 0 and self.is_in_same_paragraph(self.content_lines[page_id][x-1], self.content_lines[page_id][x], page_id, page_id))] # Remove in para caption
            cap_info = {
                "classes": [],    # * List of class of caption, including Image_Caption and Table_Caption
                "groups": [],     # * List of [ids of same caption region]
                "bboxes": [],     # * List of Max_Bbox of each group
                "locations": [],  # * List of where each group are located in (Left/Right/Mid)
                "orders": []      # * Reading order of each group
            }
            for cap_st_id in caption_start_ids:
                if self.regular['Image_Caption'].match(self.content_lines[page_id][cap_st_id][0]) != None:
                    cap_info["classes"].append('Image_Caption')
                elif self.regular['Table_Caption'].match(self.content_lines[page_id][cap_st_id][0]) != None:
                    cap_info["classes"].append('Table_Caption')
                else:logger.error("PDF:{} Page:{} has mismatched content: {}".format(self.pdf_path, page_id, self.content_lines[page_id][cap_st_id][0]))
                cap_group = [cap_st_id]
                for next_id in range(cap_st_id+1, self.content_lines[page_id].__len__()):
                    if self.is_in_same_paragraph(self.content_lines[page_id][next_id-1], self.content_lines[page_id][next_id], page_id, page_id):
                        cap_group.append(next_id)
                    else: break
                for x in cap_group:
                    self.content_lines[page_id][x].append(cap_info["classes"][-1])
                self.class_lines[cap_info["classes"][-1]].append([self.content_lines[page_id][x] for x in cap_group])
                boxes = [self.content_lines[page_id][x][1] for x in cap_group]
                cap_info["bboxes"].append(self.max_bbox(boxes))
                cap_info["groups"].append(cap_group)
                cap_info["locations"].append(self.in_which_region(cap_info["bboxes"][-1], page_id))
            mid_y_ids = [[(cap_info['bboxes'][i][1]+cap_info['bboxes'][i][3])/2, i] for i in range(len(cap_info['bboxes']))]
            mid_y_ids.sort(key=lambda x:x[0])
            in_mid_reg_img_ids = [i for i in range(len(cap_info['locations'])) if cap_info['locations'][i]=='Mid']
            id_groups_divided_by_mid_id = []
            lr_rr_stack = []
            for sorted_id in [x[1] for x in mid_y_ids]:
                if sorted_id in in_mid_reg_img_ids:
                    if len(lr_rr_stack):
                        id_groups_divided_by_mid_id.append(lr_rr_stack)
                    lr_rr_stack = []
                    id_groups_divided_by_mid_id.append(sorted_id)
                else:
                    lr_rr_stack.append(sorted_id)
            id_groups_divided_by_mid_id.append(lr_rr_stack)
            for group in id_groups_divided_by_mid_id:
                if isinstance(group, int):
                    cap_info["orders"].append(group)
                else: # arrange inner-group order
                    cap_info["orders"].extend([x for x in group if cap_info['locations'][x] == 'Left']) # Already sorted
                    cap_info["orders"].extend([x for x in group if cap_info['locations'][x] == 'Right'])
            cap_infos.append(cap_info)
        return cap_infos

    def extract_image_table(self):
        def cl_bbox_in_area(area_box, page_id):
            """ Find the max bounding box of content_lines in area_box which may be part of normal paragraph """
            potential_cl_ids = [cl_id for cl_id in range(len(self.content_lines[page_id])) if self.box_contained_in(self.content_lines[page_id][cl_id][1], area_box)]
            cl_box_ids = []
            
            for potential_cl_id in potential_cl_ids:
                box = self.content_lines[page_id][potential_cl_id][1]
                if box[2] < self.page2img_size[page_id][0]/2: # * In Left region
                    if abs(box[0]-self.space_spliters[page_id][0]) < self.ave_char_width and abs(box[2]-self.space_spliters[page_id][1]) < self.ave_char_width:
                        cl_box_ids.append(potential_cl_id)
                if box[0] > self.page2img_size[page_id][0]/2: # * In Right region
                    if abs(box[0]-self.space_spliters[page_id][2]) < self.ave_char_width and abs(box[2]-self.space_spliters[page_id][3]) < self.ave_char_width:
                        cl_box_ids.append(potential_cl_id)
            if len(cl_box_ids) == 0:
                return None
            cl_boxes = [self.content_lines[page_id][i][1] for i in cl_box_ids]
            return self.max_bbox(cl_boxes)

        def pre_cap_y_bottom_loc(cap_info, order_id): 
            """ Find the bottom y-axis of pre caption region, either in the same colum or a Mid region """
            pre_y_loc = 0
            for pre_id in range(order_id-1, -1, -1):
                if cap_info["locations"][pre_id] == 'Mid':
                    pre_y_loc = cap_info["bboxes"][pre_id][3]
                    break
                if cap_info["locations"][pre_id] == cap_info["locations"][order_id]:
                    pre_y_loc = cap_info["bboxes"][pre_id][3]
                    break
            return pre_y_loc

        def next_cap_y_upper_loc(cap_info, order_id, page_id):
            """ Find the upper y-axis of next caption region, either in the same colum or a Mid region """
            nx_y_loc = self.page2img_size[page_id][1]
            for nx_id in range(order_id+1, len(cap_info["orders"])):
                if cap_info["locations"][nx_id] == 'Mid':
                    nx_y_loc = cap_info["bboxes"][nx_id][1]
                    break
                if cap_info["locations"][nx_id] == cap_info["locations"][order_id]:
                    nx_y_loc = cap_info["bboxes"][nx_id][1]
                    break
            return nx_y_loc

        for page_id, cap_info in enumerate(self.cap_infos):
            page_width = self.page2img_size[page_id][0]
            cap_info['cl_ids'] = []
            cap_info['img_or_tab_bboxes'] = []
            cap_info['layout'] = [-1]*len(cap_info["orders"])
            for i in range(len(cap_info['classes'])):
                cap_info['cl_ids'].append([])
                cap_info['img_or_tab_bboxes'].append([])
            for order_id in cap_info["orders"]:
                cap_box = cap_info["bboxes"][order_id]
                cap_cls = cap_info["classes"][order_id]
                cap_loc = cap_info["locations"][order_id]
                pre_y = pre_cap_y_bottom_loc(cap_info, order_id)
                next_y = next_cap_y_upper_loc(cap_info, order_id, page_id)
                left_x_border = 0 if cap_loc in ['Mid', 'Left'] else max(int(page_width/2), self.space_spliters[page_id][1])
                right_x_border = page_width if cap_loc in ['Mid', 'Right'] else max(int(page_width/2), self.space_spliters[page_id][2])
                potential_ele_box_areas = [[left_x_border, pre_y, right_x_border, cap_box[1]], [left_x_border, cap_box[3], right_x_border, next_y]]

                # * To narrow down the potential area using the content line in common paragraph
                ''' *** This Part of Code is meant to avoid including curv/edge of equation to image/table
                    *** But it will introduce more problems in (see Figure4 Page10 of EMNLP-D12-1001.pdf)
                    # TODO: Should be more specific to narrow down potentical area *^[When encounter bad cases]^*
                # cl_bbox_in_fst_potential_area = cl_bbox_in_area(potential_ele_box_areas[0],page_id)
                # if cl_bbox_in_fst_potential_area != None:
                #     potential_ele_box_areas[0] = [potential_ele_box_areas[0][0], max(potential_ele_box_areas[0][1], cl_bbox_in_fst_potential_area[3]), \
                #                                 potential_ele_box_areas[0][2], potential_ele_box_areas[0][3]]
                # cl_bbox_in_sec_potential_area = cl_bbox_in_area(potential_ele_box_areas[0],page_id)
                # if cl_bbox_in_sec_potential_area != None:
                #     potential_ele_box_areas[1] = [potential_ele_box_areas[1][0], potential_ele_box_areas[1][1], \
                #                                 potential_ele_box_areas[1][2], min(potential_ele_box_areas[1][3], cl_bbox_in_sec_potential_area[1])]
                # '''

                ele_area_flag = -1 # 0 for in first potential area, 1 for in second potential area, -1 for not find any
                # * Dealing the first potential area, ele box in the upper of caption region
                in_area_ele_ids, max_ele_bbox = self.find_ele_box_in_area(potential_ele_box_areas[0], page_id)
                if len(in_area_ele_ids) != 0: # * if has ele_boxes found in the first potential area
                    ele_area_flag = 0
                    if cap_cls == 'Image_Caption': 
                        # upper_bound decided by max_ele_bbox, other remain the same
                        cl_area = [potential_ele_box_areas[0][0], max(potential_ele_box_areas[0][1], max_ele_bbox[1]), \
                                   potential_ele_box_areas[0][2], potential_ele_box_areas[0][3]]
                        in_area_cl_ids, max_cl_box = self.find_cl_in_area(cl_area, page_id)
                        for ele_id in in_area_ele_ids:
                            self.ele_boxes[page_id][ele_id].append(order_id)
                        for cl_id in in_area_cl_ids:
                            self.content_lines[page_id][cl_id].append('Image')
                            cap_info['cl_ids'][order_id].append(cl_id)
                        cap_info['img_or_tab_bboxes'][order_id] = self.max_bbox([max_cl_box, max_ele_bbox])
                    if cap_cls == 'Table_Caption':
                        if cap_loc == 'Mid': # All ele and content in potential_ele_box_areas[0] should be considered as part of table
                            in_area_cl_ids, max_cl_box = self.find_cl_in_area(potential_ele_box_areas[0], page_id)
                            for ele_id in in_area_ele_ids:
                                self.ele_boxes[page_id][ele_id].append(order_id)
                            for cl_id in in_area_cl_ids:
                                self.content_lines[page_id][cl_id].append('Table')
                                cap_info['cl_ids'][order_id].append(cl_id)
                            cap_info['img_or_tab_bboxes'][order_id] = self.max_bbox([max_ele_bbox, max_cl_box])
                        else:
                            # Special rule for cap_loc in ['Left', 'Right']
                            ele_boxes = [self.ele_boxes[page_id][i][:4] for i in in_area_ele_ids]
                            horizontal_line_boxes = [b for b in ele_boxes if b[1]==b[3]]
                            if len(horizontal_line_boxes) == 0:
                                logger.warning("PDF:{} Page:{} table order_id:{} has no horizontal edge but has other matched ele_box in first potencial area:{}.".format(self.pdf_path, page_id, order_id, str(potential_ele_box_areas[0])))
                                ele_area_flag = -1 # Match failed, try to search in second potencial area
                            else:
                                horizontal_line_boxes.sort(key=lambda b:b[1])
                                max_y = horizontal_line_boxes[-1][1]
                                bottom_horizontal_line_boxes = [b for b in horizontal_line_boxes if b[1]==max_y]
                                bottom_horizontal_line_xs = [b[0] for b in bottom_horizontal_line_boxes]+[b[2] for b in bottom_horizontal_line_boxes]
                                min_x, max_x = min(bottom_horizontal_line_xs), max(bottom_horizontal_line_xs)
                                same_min_x_line_boxes = [b for b in horizontal_line_boxes if b[0] == min_x]
                                same_max_x_line_boxes = [b for b in horizontal_line_boxes if b[2] == max_x]
                                min_y = min([b[1] for b in (same_min_x_line_boxes+same_max_x_line_boxes)])
                                # cap_loc in ['Left', 'Right'], only need to constrain upper bound(y0), while lower bound remains same as potencial area
                                box_area = [potential_ele_box_areas[0][0], max(potential_ele_box_areas[0][1], min_y)-self.delta, \
                                            potential_ele_box_areas[0][2], potential_ele_box_areas[0][3]]
                                in_area_ele_ids_real, max_ele_bbox_real = self.find_ele_box_in_area(box_area, page_id)
                                in_area_cl_ids, max_cl_box = self.find_cl_in_area(box_area, page_id)
                                for ele_id in in_area_ele_ids_real:
                                    self.ele_boxes[page_id][ele_id].append(order_id)
                                for cl_id in in_area_cl_ids:
                                    self.content_lines[page_id][cl_id].append('Table')
                                    cap_info['cl_ids'][order_id].append(cl_id)
                                cap_info['img_or_tab_bboxes'][order_id] = self.max_bbox([max_ele_bbox_real, max_cl_box])
                # * Dealing the second potential area, ele box in the lower of caption region
                if len(in_area_ele_ids) == 0 or ele_area_flag == -1:
                    in_area_ele_ids, max_ele_bbox = self.find_ele_box_in_area(potential_ele_box_areas[1], page_id)
                    if len(in_area_ele_ids) != 0: # * if has ele_boxes found in the first potential area
                        ele_area_flag = 1
                        if cap_cls == 'Image_Caption': 
                            # lower_bound decided by max_ele_bbox, other remain the same
                            cl_area = [potential_ele_box_areas[1][0], potential_ele_box_areas[1][1], \
                                       potential_ele_box_areas[1][2], min(potential_ele_box_areas[1][3], max_ele_bbox[3])]
                            in_area_cl_ids, max_cl_box = self.find_cl_in_area(cl_area, page_id)
                            for ele_id in in_area_ele_ids:
                                self.ele_boxes[page_id][ele_id].append(order_id)
                            for cl_id in in_area_cl_ids:
                                self.content_lines[page_id][cl_id].append('Image')
                                cap_info['cl_ids'][order_id].append(cl_id)
                            cap_info['img_or_tab_bboxes'][order_id] = self.max_bbox([max_cl_box, max_ele_bbox])
                        if cap_cls == 'Table_Caption':
                            ele_boxes = [self.ele_boxes[page_id][i][:4] for i in in_area_ele_ids]
                            horizontal_line_boxes = [b for b in ele_boxes if b[1]==b[3]]
                            if len(horizontal_line_boxes) == 0:
                                logger.warning("PDF:{} Page:{} table order_id:{} has no horizontal edge but has other matched ele_box in second potencial area:{}.".format(self.pdf_path, page_id, order_id, str(potential_ele_box_areas[1])))
                                ele_area_flag = -1 # Match failed
                            else:
                                horizontal_line_boxes.sort(key=lambda b:b[1])
                                min_y = horizontal_line_boxes[0][1]
                                upper_horizontal_line_boxes = [b for b in horizontal_line_boxes if b[1]==min_y]
                                upper_horizontal_line_xs = [b[0] for b in upper_horizontal_line_boxes]+[b[2] for b in upper_horizontal_line_boxes]
                                min_x, max_x = min(upper_horizontal_line_xs), max(upper_horizontal_line_xs)
                                same_min_x_line_boxes = [b for b in horizontal_line_boxes if b[0] == min_x]
                                same_max_x_line_boxes = [b for b in horizontal_line_boxes if b[2] == max_x]
                                max_y = max([b[1] for b in (same_min_x_line_boxes+same_max_x_line_boxes)])
                                # Only need to constrain the lower bound
                                box_area = [potential_ele_box_areas[1][0], potential_ele_box_areas[1][1], \
                                            potential_ele_box_areas[1][2], min(potential_ele_box_areas[1][3], max_y)+self.delta]
                                in_area_ele_ids_real, max_ele_bbox_real = self.find_ele_box_in_area(box_area, page_id)
                                in_area_cl_ids, max_cl_box = self.find_cl_in_area(box_area, page_id)
                                for ele_id in in_area_ele_ids_real:
                                    self.ele_boxes[page_id][ele_id].append(order_id)
                                for cl_id in in_area_cl_ids:
                                    self.content_lines[page_id][cl_id].append('Table')
                                    cap_info['cl_ids'][order_id].append(cl_id)
                                cap_info['img_or_tab_bboxes'][order_id] = self.max_bbox([max_ele_bbox_real, max_cl_box])
                if ele_area_flag == -1:
                    logger.warning("PDF:{} Page:{} {} order_id:{} find no corresponding Table/Img area.".format(self.pdf_path, page_id, cap_cls, order_id))
                elif ele_area_flag == 0: # found in first potential area
                    cap_info['layout'][order_id] = "Caption under ele"
                elif ele_area_flag == 1: # found in second potential area
                    cap_info['layout'][order_id] = "Caption above ele"

    def extract_remained_ele(self):
        for page_id in range(self.content_lines.__len__()): # for each page
            ele_potential_alg_ids = []
            ss = self.space_spliters[page_id]
            for ele_id in range(self.ele_boxes[page_id].__len__()):
                x = self.ele_boxes[page_id][ele_id]
                if x[4]=='edge' and x[3]==x[1] and isinstance(x[-1], str):
                    if (x[2]-x[0]) > max([ss[1]-ss[0], ss[3]-ss[2]])*0.6:
                        if self.in_which_region(x[:4], page_id) in ['Left', 'Right']:
                            ele_potential_alg_ids.append(ele_id)
            for ele_id in ele_potential_alg_ids:
                x = self.ele_boxes[page_id][ele_id]
                if not isinstance(x[-1], str):
                    continue
                same_x_id = []
                for ele_id_ref in ele_potential_alg_ids:
                    x_ref = self.ele_boxes[page_id][ele_id_ref]
                    if x_ref[0] == x[0] and x_ref[2] == x[2]:
                        same_x_id.append(ele_id_ref)
                if len(same_x_id) > 1:
                    min_y = min([self.ele_boxes[page_id][i][1] for i in same_x_id])
                    max_y = max([self.ele_boxes[page_id][i][1] for i in same_x_id])
                    box_area = [x_ref[0]-1, min_y-1, x_ref[2]+1, max_y+1]
                    in_area_ele_ids_real, max_ele_bbox_real = self.find_ele_box_in_area(box_area, page_id)
                    in_area_cl_ids, max_cl_box = self.find_cl_in_area(box_area, page_id)
                    for ele_id in in_area_ele_ids_real:
                        self.ele_boxes[page_id][ele_id].append(-2) # -2 represent alg
                    for cl_id in in_area_cl_ids:
                        self.content_lines[page_id][cl_id].append('Algorithm')
                    self.alg_infos.append({
                        "page": page_id,
                        "bbox": self.max_bbox([max_ele_bbox_real, max_cl_box]),
                        "cl_ids": in_area_cl_ids,
                        "ele_ids": in_area_ele_ids_real
                    })
        # pass

    def extract_footnote(self):
        for page_id in range(self.content_lines.__len__()): # for each page
            ele_potential_fn_ids = []
            page_width, page_height = self.page2img_size[page_id][0], self.page2img_size[page_id][1]
            ss = self.space_spliters[page_id]
            for ele_id in range(self.ele_boxes[page_id].__len__()):
                x = self.ele_boxes[page_id][ele_id]
                if x[4]=='edge' and x[3]==x[1] and x[3] > 0.7*page_height and isinstance(x[-1], str):
                    if abs(x[0]-ss[0]) < self.ave_char_width or abs(x[0]-ss[2]) < self.ave_char_width:
                        if (x[2]-x[0])< max([ss[1]-ss[0], ss[3]-ss[2]])*0.8:
                            if self.in_which_region(x[:4], page_id) in ['Left', 'Right']:
                                ele_potential_fn_ids.append(ele_id)
            for ele_id in ele_potential_fn_ids:
                x = self.ele_boxes[page_id][ele_id]
                if x[2] <= page_width/2: # in Left
                    cl_box = [0, x[1], int(page_width/2), page_height]
                elif x[0] >= page_width/2: # in Right
                    cl_box = [int(page_width/2), x[1], page_width, page_height]
                else:
                    logger.warning("PDF:{} Page:{} has error dealing footnote, ele_box:{}.".format(self.pdf_path, page_id, str(x[:4])))
                in_area_cl_ids, _ = self.find_cl_in_area(cl_box, page_id)
                self.ele_boxes[page_id][ele_id].append(-1)
                for cl_id in in_area_cl_ids:
                    self.content_lines[page_id][cl_id].append("Footnote")

    def merge_cl_lines(self):
        # ! NOTE! This function should be called after
        # ! [extract_first_page_top_meta_info / extract_page_conf / extract_image_table / extract_footnote] are called
        # ! CLs close to each other (except in image/table) should be merged into one CL
        def overlap_len(min1, len1, min2, len2):
            min_ = min1
            max_ = min1 + len1
            if min1 > min2:
                min_ = min2
            if (min1 + len1) < (min2 + len2):
                max_ = min2 + len2
            return max(0, len1+len2-(max_-min_))
        def needs_merge(cl1, cl2, page_id, th = 0.2):
            if cl1[-1] == 'Footnote' and cl2[-1] == 'Footnote':
                overlap_l = overlap_len(cl1[1][1], cl1[1][3]-cl1[1][1], cl2[1][1], cl2[1][3]-cl2[1][1])
                if overlap_l/max(min(cl1[1][3]-cl1[1][1], cl2[1][3]-cl2[1][1]), 1) > th: # overlap big enough
                    return True
            width = self.page2img_size[page_id][0]
            # if not ((cl1[1][2]<width/2 and cl2[1][2]<width/2) or (cl1[1][0]>width/2 and cl2[1][0]>width/2)): # Not in same column
            #     return False
            if abs(cl1[1][0]-self.space_spliters[page_id][0])+abs(cl1[1][2]-self.space_spliters[page_id][1]) < 4*self.ave_char_width or abs(cl1[1][0]-self.space_spliters[page_id][2])+abs(cl1[1][2]-self.space_spliters[page_id][3]) < 4*self.ave_char_width:
                return False
            if abs(cl2[1][0]-self.space_spliters[page_id][0])+abs(cl2[1][2]-self.space_spliters[page_id][1]) < 4*self.ave_char_width or abs(cl2[1][0]-self.space_spliters[page_id][2])+abs(cl2[1][2]-self.space_spliters[page_id][3]) < 4*self.ave_char_width:
                return False
            if (cl1[1][2]-cl1[1][0]) > width/6 and (cl2[1][2]-cl2[1][0]) > width/6:
                return False
            if self.rect_distance(cl1[1][:4], cl2[1][:4]) > 5*self.ave_char_width:
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
            max_box = self.max_bbox([cl[1] for cl in cl_s])
            chars = []
            for cl_i in sorted_cl_ids:
                chars.extend(cl_s[cl_i][2])
            labels = list(set([cl[-1] for cl in cl_s if isinstance(cl[-1], str)]))
            if len(labels) > 1:
                if 'Affili' in labels:
                    labels.remove('Affili')
                logger.warning("PDF:{} page:{} merged more than one class cls: {} , text: {} , using cls: {}".format(self.pdf_path,page_id,str(labels),str(strings),labels[0]))
                return [' '.join(strings), max_box, chars, labels[0]]
            if len(labels) == 1:
                return [' '.join(strings), max_box, chars, labels[0]]
            return [' '.join(strings), max_box, chars]

        for page_id in range(self.content_lines.__len__()):
            cur_cl2clm_mapping = {}
            need_merge_line_id_groups = []
            RANGE = 5
            for cl_id in range(self.content_lines[page_id].__len__()):
                cur_group = []
                if isinstance(self.content_lines[page_id][cl_id][-1], str):
                    if self.content_lines[page_id][cl_id][-1] in ['Image', 'Table', 'Algorithm']:
                        continue
                for cur_cl_id in range(max(0, cl_id-RANGE), min(len(self.content_lines[page_id]), cl_id+RANGE)):
                    if isinstance(self.content_lines[page_id][cur_cl_id][-1], str):
                        if self.content_lines[page_id][cl_id][-1] in ['Image', 'Table', 'Algorithm']:
                            continue
                    if needs_merge(self.content_lines[page_id][cl_id], self.content_lines[page_id][cur_cl_id], page_id):
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
            for cl_id in range(len(self.content_lines[page_id])):
                if cl_id not in all_need_to_merge_ids:
                    cur_cl2clm_mapping[cl_id] = len(new_cl_this_page)
                    new_cl_this_page.append(self.content_lines[page_id][cl_id])
                else:
                    group_id = which_group(cl_id, need_merge_line_id_groups)
                    if not merged_to_new[group_id]:
                        for i in need_merge_line_id_groups[group_id]:
                            cur_cl2clm_mapping[i] = len(new_cl_this_page)
                        cl_lists = [self.content_lines[page_id][i] for i in need_merge_line_id_groups[group_id]]
                        new_cl_this_page.append(cl_join(cl_lists, page_id))
                        merged_to_new[group_id] = True
            self.cl2clm_mapping.append(cur_cl2clm_mapping)
            self.content_lines_merged[page_id] = new_cl_this_page

    def modify_cl2clm(self): # Modify the including cl_ids in each img/tab to clm_ids
        for page_id, cap_info in enumerate(self.cap_infos):
            for cl_ids in cap_info["cl_ids"]:
                for i in range(len(cl_ids)):
                    cl_ids[i] = self.cl2clm_mapping[page_id][cl_ids[i]]
            for cl_ids in cap_info["groups"]:
                for i in range(len(cl_ids)):
                    cl_ids[i] = self.cl2clm_mapping[page_id][cl_ids[i]]
        for alg_info in self.alg_infos:
            for i in range(len(alg_info["cl_ids"])):
                alg_info["cl_ids"][i] = self.cl2clm_mapping[alg_info["page"]][alg_info["cl_ids"][i]]

    def font_bold(self, s):
        for i in self.section_suffix:
            if s.endswith(i):
                return True
        return False
        
    def extract_section(self):
        def is_special_name(text):
            for s in self.special_section_name:
                if cal_wer(text, s) < 0.4:
                    return True
            return False
        # Using content_lines_merged
        ALL_sections_cl_page_id = []
        for page_id in range(self.content_lines_merged.__len__()):
            for cl_id in range(self.content_lines_merged[page_id].__len__()):
                if isinstance(self.content_lines_merged[page_id][cl_id][-1], str):
                    continue
                cl = self.content_lines_merged[page_id][cl_id]
                fonts = [x[1] for x in cl[2]]
                fonts_are_bold = [self.font_bold(x) for x in fonts]
                if False not in fonts_are_bold: # All fonts are bold
                    cl.append('Section')
                    ALL_sections_cl_page_id.append([page_id, cl_id])
                    if cal_wer(cl[0], 'References') < 0.4:
                        if self.references_page_id != [-1, -1]:
                            logger.warning("PDF:{} Page:{} has multiple References section, first:{}, current:{}.".format(self.pdf_path, page_id, str(self.references_page_id), str([page_id, cl_id])))
                        else:
                            self.references_page_id = [page_id, cl_id]
                    if cal_wer(cl[0], 'Appendix') < 0.4:
                        if self.appendix_page_id != [-1, -1]:
                            logger.warning("PDF:{} Page:{} has multiple Appendix section, first:{}, current:{}.".format(self.pdf_path, page_id, str(self.references_page_id), str([page_id, cl_id])))
                        else:
                            self.appendix_page_id = [page_id, cl_id]
        if len(ALL_sections_cl_page_id) < 5:
            all_section_names = [self.content_lines_merged[x[0]][x[1]][0] for x in ALL_sections_cl_page_id]
            logger.error("PDF:{} has less than 5 section name:{}, please check bold-font rule!".format(self.pdf_path, str(all_section_names)))
        if self.references_page_id == [-1, -1]:
            logger.error("PDF:{} has No References section found!".format(self.pdf_path))
        # * Before Reference handling process is different than after Reference
        ALL_sections_cl = [self.content_lines_merged[x[0]][x[1]] for x in ALL_sections_cl_page_id]
        for i in range(len(ALL_sections_cl_page_id)):
            cl = ALL_sections_cl[i]
            page_id, cl_id = ALL_sections_cl_page_id[i][0], ALL_sections_cl_page_id[i][1]
            if page_id < self.references_page_id[0] or (page_id == self.references_page_id[0] and cl_id < self.references_page_id[1]): # Main body
                if is_special_name(cl[0]):
                    self.section_hierarchy.append({
                        "depth": 1,
                        "content": cl[0],
                        "page_id": page_id,
                        "cl_ids": [cl_id],
                        "parent": [-1, -1] # use page_id, cl_id to repr unique section node
                    })
                    cl[-1] = "Section_1"
                elif cl[0][:1].isdigit():
                    depth = cl[0].split()[0].count('.') + 1
                    if depth > 1:
                        parent_sec_hier_id = len(self.section_hierarchy)-1
                        while parent_sec_hier_id >= 0 and self.section_hierarchy[parent_sec_hier_id]["depth"] != depth-1:
                            parent_sec_hier_id -= 1
                        if parent_sec_hier_id == 0: logger.warning("PDF:{} section:{} has error finding parent section".format(self.pdf_path, cl[0]))
                        parent_page_id = [self.section_hierarchy[parent_sec_hier_id]["page_id"], self.section_hierarchy[parent_sec_hier_id]["cl_ids"][0]]
                    else: parent_page_id = [-1, -1]
                    self.section_hierarchy.append({
                        "depth": depth,
                        "content": cl[0],
                        "page_id": page_id,
                        "cl_ids": [cl_id],
                        "parent": parent_page_id # use page_id, cl_id to repr unique section node
                    })
                    if depth in [1,2,3]: label = 'Section_'+str(depth)
                    else: label = 'Section_X'
                    cl[-1] = label
                else:
                    if page_id == ALL_sections_cl_page_id[i-1][0] and cl_id == ALL_sections_cl_page_id[i-1][1] + 1 and self.rect_distance(ALL_sections_cl[i-1][1][:4], cl[1][:4]) < min(ALL_sections_cl[i-1][1][3]-ALL_sections_cl[i-1][1][1], cl[1][3]-cl[1][1])*0.7: # Be part of section
                        self.section_hierarchy[-1]["content"] = self.section_hierarchy[-1]["content"] + " " + cl[0]
                        self.section_hierarchy[-1]["cl_ids"].append(cl_id)
                        cl[-1] = "Section_Lines"
                    else:
                        cl.pop()
            elif page_id == self.references_page_id[0] and cl_id == self.references_page_id[1]: # References
                self.section_hierarchy.append({
                    "depth": 1,
                    "content": cl[0],
                    "page_id": page_id,
                    "cl_ids": [cl_id],
                    "parent": [-1, -1] # use page_id, cl_id to repr unique section node
                })
                cl[-1] = "Section_1"
            else: # after References section, Appendix or later
                if self.rect_distance(ALL_sections_cl[i-1][1][:4], cl[1][:4]) < min(ALL_sections_cl[i-1][1][3]-ALL_sections_cl[i-1][1][1], cl[1][3]-cl[1][1])*0.7: # Be part of section
                    self.section_hierarchy[-1]["content"] = self.section_hierarchy[-1]["content"] + " " + cl[0]
                    self.section_hierarchy[-1]["cl_ids"].append(cl_id)
                    cl[-1] = "Section_Lines"
                else:
                    if self.appendix_page_id[0] > 0 and ( page_id > self.appendix_page_id[0] or (page_id == self.appendix_page_id[0] and cl_id > self.appendix_page_id[1])): # has appendix and after appendix section
                        self.section_hierarchy.append({
                            "depth": 1,
                            "content": cl[0],
                            "page_id": page_id,
                            "cl_ids": [cl_id],
                            "parent": self.appendix_page_id # use page_id, cl_id to repr unique section node
                        })
                        cl[-1] = "Section_2"
                    else: # no appendix or is appendix
                        self.section_hierarchy.append({
                            "depth": 1,
                            "content": cl[0],
                            "page_id": page_id,
                            "cl_ids": [cl_id],
                            "parent": [-1, -1] # use page_id, cl_id to repr unique section node
                        })
                        cl[-1] = "Section_1"

    def extract_reference(self):
        def is_same_reference(pre_page_cl_id, next_page_cl_id):
            pre_cl = self.content_lines_merged[pre_page_cl_id[0]][pre_page_cl_id[1]]
            next_cl = self.content_lines_merged[next_page_cl_id[0]][next_page_cl_id[1]]
            if pre_page_cl_id[0]!=next_page_cl_id[0]: # at different page
                if (next_cl[1][2]-next_cl[1][0]) > (pre_cl[1][2]-pre_cl[1][0])+self.ave_char_width: # This means next_cl is a new refer
                    return False
                else: # ! Problem will occur when pre_cl is a single-line refer, however this rarely happens
                    return True
            page_width = self.page2img_size[pre_page_cl_id[0]][0]
            if (pre_cl[1][2] < page_width/2 and next_cl[1][2] < page_width/2) or (pre_cl[1][0] > page_width/2 and next_cl[1][0] > page_width/2): # Same column
                if next_cl[1][0] < pre_cl[1][0]-self.ave_char_width:
                    return False
                else: 
                    return True
            if pre_cl[1][2] < page_width/2 and next_cl[1][0] > page_width/2: # Differe column, pre in left, next in right
                if (next_cl[1][2]-next_cl[1][0]) > (pre_cl[1][2]-pre_cl[1][0])+self.ave_char_width: # This means next_cl is a new refer
                    return False
                else: # ! Problem will occur when pre_cl is a single-line refer, however this rarely happens
                    return True
            return False
        if self.references_page_id == [-1, -1]:
            return # No References section exists
        cur_group = []
        finish = False
        for page_id in range(self.references_page_id[0],len(self.content_lines_merged)):
            if finish: break
            cl_start_id = self.references_page_id[1]+1 if page_id == self.references_page_id[0] else 0
            for cl_id in range(cl_start_id, len(self.content_lines_merged[page_id])):
                cl = self.content_lines_merged[page_id][cl_id]
                if 'Section' in cl[-1]:
                    finish = True
                    break
                if isinstance(cl[-1], str): # ignore page/img/tab etc. labeled cl
                    continue
                if len(cur_group) == 0:
                    cur_group.append([page_id, cl_id])
                    cl.append('FirstLine')
                elif is_same_reference(cur_group[-1], [page_id, cl_id]):
                    cur_group.append([page_id, cl_id])
                    cl.append('ContentLine')
                else:
                    self.paragraph_groups.append(cur_group)
                    cur_group = []
                    cur_group.append([page_id, cl_id])
                    cl.append('FirstLine')
        if len(cur_group) > 0:
            self.paragraph_groups.append(cur_group)

    def extract_paragraph(self):
        cur_group = []
        for page_id in range(self.content_lines_merged.__len__()):
            no_label_cl_ids = [cl_id for cl_id in range(self.content_lines_merged[page_id].__len__()) if not isinstance(self.content_lines_merged[page_id][cl_id][-1], str)]
            for tmp_id in range(len(no_label_cl_ids)):
                cl_id = no_label_cl_ids[tmp_id]
                cl = self.content_lines_merged[page_id][cl_id]
                # * Dealing equation part * --- BEGIN --- * #
                if cl[1][3] - cl[1][1] > 2*self.normal_line_height:
                    cl.append('Equation')
                    continue
                region = self.in_which_region(cl[1], page_id)
                if region == "Left":
                    if abs(cl[1][0] - self.space_spliters[page_id][0]) > 8*self.ave_char_width:
                        if abs(cl[1][2] - self.space_spliters[page_id][1]) < self.ave_char_width and re.compile(r"\(\d+\)").match(cl[0]) != None:
                            cl.append('Equation_Label')
                        else: cl.append('Equation')
                        continue
                if region == "Right":
                    if abs(cl[1][0] - self.space_spliters[page_id][2]) > 8*self.ave_char_width:
                        if abs(cl[1][2] - self.space_spliters[page_id][3]) < self.ave_char_width and re.compile(r"\(\d+\)").match(cl[0]) != None:
                            cl.append('Equation_Label')
                        else: cl.append('Equation')
                        continue
                # * Dealing equation part * ---  END  --- * #
                if tmp_id == 0: # the first cl in page that has no label yet
                    if cl_id != 0 and 'Section' in self.content_lines_merged[page_id][cl_id-1][-1]:
                        if len(cur_group) > 0:
                            self.paragraph_groups.append(cur_group)
                            cur_group = []
                        cl.append('FirstLine')
                        cur_group.append([page_id, cl_id])
                    elif page_id == 0:
                        assert len(cur_group) == 0
                        cl.append('FirstLine')
                        cur_group.append([page_id, cl_id])
                    else:
                        if len(cur_group) == 0:
                            cl.append('FirstLine')
                            cur_group.append([page_id, cl_id])
                        elif self.is_in_same_paragraph(self.content_lines_merged[cur_group[-1][0]][cur_group[-1][1]], cl, cur_group[-1][0], page_id):
                            cl.append('ContentLine')
                            cur_group.append([page_id, cl_id])
                        else:
                            if len(cur_group) > 0:
                                self.paragraph_groups.append(cur_group)
                                cur_group = []
                            cl.append('FirstLine')
                            cur_group.append([page_id, cl_id])
                else:
                    pre_no_label_cl = self.content_lines_merged[page_id][no_label_cl_ids[tmp_id-1]]
                    pre_cl = self.content_lines_merged[page_id][cl_id-1]
                    if 'Section' in pre_cl[-1]:
                        if len(cur_group) > 0:
                            self.paragraph_groups.append(cur_group)
                            cur_group = []
                        cl.append('FirstLine')
                        cur_group.append([page_id, cl_id])
                    else:
                        if self.is_in_same_paragraph(pre_no_label_cl, cl, page_id, page_id):
                            cl.append('ContentLine')
                            cur_group.append([page_id, cl_id])
                        else:
                            if len(cur_group) > 0:
                                self.paragraph_groups.append(cur_group)
                                cur_group = []
                            cl.append('FirstLine')
                            cur_group.append([page_id, cl_id])
        if len(cur_group) > 0:
            self.paragraph_groups.append(cur_group)

    def modify_first_line(self):
        # Need to modify title/footnote/image_caption/table_caption
        # * Handling title
        title_cls = [cl for cl in self.content_lines_merged[0] if cl[-1] == 'Title']
        for cl in title_cls[1:]:
            cl[-1] = 'Title_Lines'
        # * Handling footnote
        for page_id in range(len(self.content_lines_merged)):
            page_width = self.page2img_size[page_id][0]
            fn_cls = [cl for cl in self.content_lines_merged[page_id] if cl[-1] == 'Footnote']
            for cl in fn_cls:
                if cl[1][2] < page_width/2 and abs(cl[1][0]-self.space_spliters[page_id][0]) < self.ave_char_width:
                    cl[-1] = 'Footnote_Lines'
                if cl[1][0] > page_width/2 and abs(cl[1][0]-self.space_spliters[page_id][2]) < self.ave_char_width:
                    cl[-1] = 'Footnote_Lines'
        # * Handling caption
        for page_id, cap_info in enumerate(self.cap_infos):
            for i in range(len(cap_info['classes'])):
                cls_ = cap_info['classes'][i]
                for cl_id in cap_info['groups'][i][1:]:
                    self.content_lines_merged[page_id][cl_id][-1] = 'Image_Caption_Lines' if cls_ == 'Image_Caption' else 'Table_Caption_Lines'
        # * Handling first_line
        for page_id in range(len(self.content_lines_merged)):
            page_width = self.page2img_size[page_id][0]
            fl_cl_ids = [cl_id for cl_id in range(len(self.content_lines_merged[page_id])) if self.content_lines_merged[page_id][cl_id][-1] == 'FirstLine']
            for fl_cl_id in fl_cl_ids:
                if self.content_lines_merged[page_id][fl_cl_id-1][-1] == 'FirstLine' and (self.content_lines_merged[page_id][fl_cl_id-1][1][2]-self.content_lines_merged[page_id][fl_cl_id-1][1][0]) > page_width/5: # Preline is also FirstLine
                    if self.content_lines_merged[page_id][fl_cl_id][0][0] in [chr(i) for i in range(97, 123)]+['{', '(']: # begin with lower-case alphabet
                        self.content_lines_merged[page_id][fl_cl_id][-1] = 'ContentLine'

    def find_bold_section(self):
        def before(p1, c1, p2, c2): # whether (p1, c1) comes befor (p2, c2)
            return p1 < p2 if p1 != p2 else c1 < c2
        def which_section_level(page_id, cl_id):
            sec_id = 0
            while sec_id < len(self.section_hierarchy) - 1:
                cur_sec = self.section_hierarchy[sec_id]
                next_sec = self.section_hierarchy[sec_id+1]
                if before(cur_sec['page_id'], cur_sec['cl_ids'][0], page_id, cl_id) and before(page_id, cl_id, next_sec['page_id'], next_sec['cl_ids'][0]):
                    break
                sec_id += 1
            parent_sec_id = sec_id
            parent_sec = self.section_hierarchy[parent_sec_id]
            if not before(parent_sec['page_id'], parent_sec['cl_ids'][0], page_id, cl_id):
                logger.warning("PDF:{} page:{} text:{} may have error parent section relation".format(self.pdf_path, page_id, self.merge_cl_lines[page_id][cl_id][0]))
            if parent_sec['depth'] == 1:
                return 'Section_2'
            if parent_sec['depth'] == 2:
                return 'Section_3'
            if parent_sec['depth'] >= 3:
                return 'Section_X'
        def merge_cl_using_chars(chars):
            str_ = ""
            max_box = self.max_bbox([ch[3] for ch in chars])
            last_right_x = chars[0][3][0]
            for ch in chars:
                if ch[3][0] <= last_right_x:
                    str_ = str_ + ch[0]
                else:
                    str_ = str_ + ' ' + ch[0]
                last_right_x = ch[3][2]
            return [str_, max_box, chars]
        def renew_cl_ids(page_id, cl_id):
            for sec in self.section_hierarchy:
                if page_id == sec['page_id'] and sec['cl_ids'][0] > cl_id:
                    for i in range(len(sec['cl_ids'])):
                        sec['cl_ids'][i] = sec['cl_ids'][i]+1
            for alg_info in self.alg_infos:
                if page_id == alg_info["page"]:
                    if len(alg_info['cl_ids']) == 0:
                        continue
                    min_cl_id, max_cl_id = min(alg_info['cl_ids']), max(alg_info['cl_ids'])
                    if cl_id >= min_cl_id and cl_id <= max_cl_id:
                        logger.warning("PDF:{} Page:{} meet error when find_bold_section changing algorithm to cl_id mapping".format(self.pdf_path, page_id))
                    if min_cl_id > cl_id:
                        for i in range(len(alg_info['cl_ids'])):
                            alg_info['cl_ids'][i] = alg_info['cl_ids'][i] + 1
            for tabimg_id in range(len(self.cap_infos[page_id]['classes'])):
                for tag in ['groups', 'cl_ids']:
                    cl_id_grp = self.cap_infos[page_id][tag][tabimg_id]
                    if len(cl_id_grp) == 0:
                        continue
                    min_cl_id, max_cl_id = min(cl_id_grp), max(cl_id_grp)
                    if cl_id >= min_cl_id and cl_id <= max_cl_id:
                        logger.warning("PDF:{} Page:{} meet error when find_bold_section changing tab/img caption to cl_id mapping, first text:{}".format(self.pdf_path, page_id, self.content_lines_merged[page_id][min_cl_id][0]))
                    if min_cl_id > cl_id:
                        for i in range(len(cl_id_grp)):
                            cl_id_grp[i] = cl_id_grp[i] + 1
        for page_id in range(len(self.content_lines_merged)):
            cl_id = 0
            while cl_id < len(self.content_lines_merged[page_id]):
                cl = self.content_lines_merged[page_id][cl_id]
                if cl[-1] != "FirstLine":
                    cl_id += 1
                    continue
                if not self.font_bold(cl[2][0][1]):
                    cl_id += 1
                    continue
                # * Now meets a section contained in FirstLine
                fonts_are_bold = [self.font_bold(x) for x in [x[1] for x in cl[2]]]
                if False not in fonts_are_bold: # * The first line is all of one section
                    cl[-1] = which_section_level(page_id, cl_id)
                    cl_id += 1
                    cur_cl = self.content_lines_merged[page_id][cl_id]
                    cur_cl_fonts_are_bold = [self.font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                    while False not in cur_cl_fonts_are_bold: # * All lines are bold, means all char are bold font
                        cur_cl[-1] = 'Section_Lines'
                        cl_id += 1
                        cur_cl = self.content_lines_merged[page_id][cl_id]
                        cur_cl_fonts_are_bold = [self.font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                    # * Current meet one line that is not full of bold font
                    bold_font_max_id = cur_cl_fonts_are_bold.index(False)
                    if bold_font_max_id == 0: # * All chars in this cl are regarded as common content line
                        cur_cl[-1] = "FirstLine"
                        cl_id += 1
                        continue
                    new_cl1 = merge_cl_using_chars(cur_cl[2][:bold_font_max_id])
                    new_cl2 = merge_cl_using_chars(cur_cl[2][bold_font_max_id:])
                    new_cl1.append("Section_Lines")
                    new_cl2.append("FirstLine")
                    new_merged_cls = self.content_lines_merged[page_id][:cl_id]
                    new_merged_cls.append(new_cl1)
                    new_merged_cls.append(new_cl2)
                    new_merged_cls.extend(self.content_lines_merged[page_id][cl_id+1:])
                    self.content_lines_merged[page_id] = new_merged_cls
                    renew_cl_ids(page_id, cl_id)
                    cl_id += 2
                else: # * The first line contain section, but not all are section
                    bold_font_max_id = fonts_are_bold.index(False)
                    new_cl1 = merge_cl_using_chars(cl[2][:bold_font_max_id])
                    new_cl2 = merge_cl_using_chars(cl[2][bold_font_max_id:])
                    new_cl1.append(which_section_level(page_id, cl_id))
                    new_cl2.append("FirstLine")
                    new_merged_cls = self.content_lines_merged[page_id][:cl_id]
                    new_merged_cls.append(new_cl1)
                    new_merged_cls.append(new_cl2)
                    new_merged_cls.extend(self.content_lines_merged[page_id][cl_id+1:])
                    self.content_lines_merged[page_id] = new_merged_cls
                    renew_cl_ids(page_id, cl_id)
                    cl_id += 2
 
    def visual_labeled_cl(self):
        def put_label(img, box, color_rgb, label, left_border=20, bg_color="#000000"):
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline=color_rgb, width=1)
            lable_box = [box[0]-left_border, box[1], box[0]-3, box[1]+10]
            draw.rectangle(lable_box, fill=bg_color, width=-1)
            font = ImageFont.truetype("msyh.ttf", size=8)
            draw.text((lable_box[0],lable_box[1]), label, font=font, fill="#FFFFFF")
        for page_id in range(self.content_lines_merged.__len__()):
            img = Image.open(os.path.join(self.img_savedir, "{}.png".format(page_id))) # RGB format
            start_id = 1
            for cl in self.content_lines_merged[page_id]:
                cl_class = cl[-1]
                if not isinstance(cl_class, str):
                    cl_class = 'ContentLine'
                if cl_class in ['Image', 'Table', 'Algorithm']:
                    continue
                if cl_class not in self.ALL_CLASS.keys():
                    logger.warning("PDF:{} Page:{} has error in visual, cl_class:{} not in self.ALL_CLASS!".format(self.pdf_path, page_id, cl_class))
                text = self.ALL_CLASS[cl_class]["short"]
                color_rgb = self.ALL_CLASS[cl_class]["color_rgb"]
                if cl_class in ["Footnote", "Image_Caption", "Table_Caption", "FirstLine", "Section_1", "Section_2", "Section_3", "Section_X"]:
                    text = str(start_id)+" "+text
                    start_id += 1
                    put_label(img, cl[1], color_rgb, text, left_border = 30, bg_color = "#FF0000")
                else:
                    put_label(img, cl[1], color_rgb, text)
            # plot img/table
            for i in range(len(self.cap_infos[page_id]['classes'])):
                cls_ = self.cap_infos[page_id]['classes'][i]
                box = self.cap_infos[page_id]['img_or_tab_bboxes'][i]
                if len(box) != 4:
                    continue
                cl_class = 'Image' if cls_=='Image_Caption' else 'Table'
                color_rgb = self.ALL_CLASS[cl_class]["color_rgb"]
                put_label(img, box, color_rgb, self.ALL_CLASS[cl_class]["short"])
            # plot alg
            alg_infos = [x for x in self.alg_infos if x["page"] == page_id]
            for alg_info in alg_infos:
                box = alg_info["bbox"]
                cl_class = "Algorithm"
                color_rgb = self.ALL_CLASS[cl_class]["color_rgb"]
                put_label(img, box, color_rgb, self.ALL_CLASS[cl_class]["short"])
            img.save(os.path.join(self.img_savedir, "{}_visual_labeled_cl.png".format(page_id)), "png")

    def save_info(self, delete_visual=False, delete_pdf=False):
        img_info = []
        tab_info = []
        alg_info = []
        for page_id, cap_info in enumerate(self.cap_infos):
            for order_id in cap_info['orders']:
                cls_ = cap_info['classes'][order_id]
                if cls_ == "Table_Caption":
                    tab_info.append({
                        'class' : 'Table',
                        'page' : page_id,
                        'box' : [int(x) for x in cap_info['img_or_tab_bboxes'][order_id]],
                        'layout' : cap_info['layout'][order_id],
                        'location' : cap_info['locations'][order_id],
                        'id' : len(tab_info),
                        'caption_ids' : [int(x) for x in cap_info['groups'][order_id]],
                    })
                elif cls_ == "Image_Caption":
                    img_info.append({
                        'class' : 'Image',
                        'page' : page_id,
                        'box' : [int(x) for x in cap_info['img_or_tab_bboxes'][order_id]],
                        'layout' : cap_info['layout'][order_id],
                        'location' : cap_info['locations'][order_id],
                        'id' : len(img_info),
                        'caption_ids' : [int(x) for x in cap_info['groups'][order_id]],
                    })
        for alg_info_ in self.alg_infos:
            page_id = alg_info_['page']
            cl_ids = alg_info_['cl_ids']
            for i in cl_ids:
                cl = self.content_lines_merged[page_id][i]
                if cl[-1] != 'Algorithm':
                    logger.error("PDF:{} Page:{} has error matching self.alg_infos to Algorithm cls: {}".format(self.pdf_path, page_id, cl[0]))
            alg_info.append({
                'class' : 'Algorithm',
                'page' : page_id,
                'box' : [int(x) for x in alg_info_['bbox']],
                'id' : len(alg_info),
                'cl_ids' : [int(x) for x in cl_ids],
            })
        def find_belong_id(infos, page_id, cl_id):
            for info in infos:
                if page_id == info['page'] and cl_id in info['caption_ids']:
                    return info['id']
            return None
        cur_cls = []
        for page_id in range(len(self.content_lines_merged)):
            cur_page_cls = {}
            for cl_id in range(len(self.content_lines_merged[page_id])):
                cl = self.content_lines_merged[page_id][cl_id]
                if not isinstance(cl[-1], str):
                    cur_page_cls[str(len(cur_page_cls.keys()))] = {"text": cl[0], "box": [int(x) for x in cl[1]], "class": 'ContentLine'}
                else:
                    add_new_cl = {"text": cl[0], "box": [int(x) for x in cl[1]], "class": cl[-1]}
                    if cl[-1] == "Table_Caption":
                        add_new_cl["parent"] = find_belong_id(tab_info, page_id, cl_id)
                        if add_new_cl["parent"] == None:
                            logger.error("PDF:{} Page:{} has error finding caption: {} corresponding Table. This is a serious error!".format(self.pdf_path, page_id, cl[0]))
                    if cl[-1] == "Image_Caption":
                        add_new_cl["parent"] = find_belong_id(img_info, page_id, cl_id)
                        if add_new_cl["parent"] == None:
                            logger.error("PDF:{} Page:{} has error finding caption: {} corresponding Image. This is a serious error!".format(self.pdf_path, page_id, cl[0]))
                    cur_page_cls[str(len(cur_page_cls.keys()))] = add_new_cl
            cur_cls.append(cur_page_cls)
                
        all_info = {
            "pdf": self.pdf_path,
            "images": img_info,
            "tables": tab_info,
            "algs": alg_info,
            "conetent_lines": cur_cls,
        }
        pdf_base_path, pdf_name = os.path.split(self.pdf_path)
        json_path = os.path.join(pdf_base_path, pdf_name.replace('.pdf', '.json'))
        json.dump(all_info, open(json_path, 'w'), indent=4)
        if delete_visual:
            all_ele_boxes_imgs = glob.glob(os.path.join(self.img_savedir, "*_ele_boxes.png"))
            for f in all_ele_boxes_imgs:
                os.remove(f)
        if delete_pdf:
            os.remove(self.pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pdf meta infomations of ACL papers')
    parser.add_argument('--pdf_folder', type=str, 
                        default="Path To Your PDF folder",
                        help='The path storing all pdf files')
    args = parser.parse_args()

    pdf_list = get_pdf_paths(args.pdf_folder, recursive=True)
    for pdf_path in tqdm.tqdm(pdf_list):
        try:
            doc_parser = DocParser(pdf_path, visual=True)
            doc_parser.extract_all()
            # doc_parser.save_info(delete_visual=True, delete_pdf=True)
        except Exception as e:
            print(repr(e))
            continue
    print('All Done!') 