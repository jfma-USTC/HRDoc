#!/usr/bin/env python
# -*- coding:utf-8 -*-

import subprocess
import os
import re
import shutil
import tqdm
import json
from utils import get_main_tex_path, open_tex_file, logger, Left_Brace, Begin_Brace

Insert_Colors = [ # some command will automatically transfer command to upper-cased format, use upper-case instead
    r"\usepackage{xcolor}",
    r"\definecolor{MYTITLE}{RGB}{0, 191, 250}",
    r"\definecolor{MYAUTHOR}{RGB}{0, 0, 139}",
    r"\definecolor{MYMAIL}{RGB}{0, 139, 139}",
    r"\definecolor{MYAFFILI}{RGB}{139, 139, 139}",
    r"\definecolor{MYABS}{RGB}{250, 182, 193}",
    r"\definecolor{MYCAP}{RGB}{0, 250, 0}",
    r"\definecolor{MYEQU}{RGB}{250, 0, 0}",
    r"\definecolor{MYFIG}{RGB}{250, 0, 250}",
    r"\definecolor{MYTAB}{RGB}{0, 250, 250}",
    r"\definecolor{MYALG}{RGB}{139, 0, 139}",
    r"\definecolor{MYFOOTER}{RGB}{250, 128, 0}",
    r"\definecolor{MYPARA}{RGB}{176, 196, 222}",
    r"\definecolor{MYREF}{RGB}{48, 144, 0}",
    r"\definecolor{MYSECTION}{RGB}{0, 0, 250}",
    r"\definecolor{MYSUBSECTION}{RGB}{48, 32, 112}",
    r"\definecolor{MYSUBSUBSECTION}{RGB}{125, 10, 0}",
]

Insert_Black = [ # some command will automatically transfer command to upper-cased format, use upper-case instead
    r"\usepackage{xcolor}",
    r"\definecolor{MYTITLE}{RGB}{0, 0, 0}",
    r"\definecolor{MYAUTHOR}{RGB}{0, 0, 0}",
    r"\definecolor{MYMAIL}{RGB}{0, 0, 0}",
    r"\definecolor{MYAFFILI}{RGB}{0, 0, 0}",
    r"\definecolor{MYABS}{RGB}{0, 0, 0}",
    r"\definecolor{MYCAP}{RGB}{0, 0, 0}",
    r"\definecolor{MYEQU}{RGB}{0, 0, 0}",
    r"\definecolor{MYFIG}{RGB}{0, 0, 0}",
    r"\definecolor{MYTAB}{RGB}{0, 0, 0}",
    r"\definecolor{MYALG}{RGB}{0, 0, 0}",
    r"\definecolor{MYFOOTER}{RGB}{0, 0, 0}",
    r"\definecolor{MYPARA}{RGB}{0, 0, 0}",
    r"\definecolor{MYREF}{RGB}{0, 0, 0}",
    r"\definecolor{MYSECTION}{RGB}{0, 0, 0}",
    r"\definecolor{MYSUBSECTION}{RGB}{0, 0, 0}",
    r"\definecolor{MYSUBSUBSECTION}{RGB}{0, 0, 0}",
]

Match_Regex = {
    "title_begin": re.compile(r"^\\((sub)*)title[^a-z]"),
    "abs_begin": re.compile(r"^\\abstract[^a-z]"),
    "author_begin": re.compile(r"^\\author[^a-z]"),
    "mail_begin": re.compile(r"^\\email[^a-z]"),
    "address_begin": re.compile(r"^\\address[^a-z]"),
    "institute_begin": re.compile(r"^\\institute[^a-z]"),
    "affili_begin": re.compile(r"^\\affiliation[^a-z]"),
    "sec_begin": re.compile(r"\\section[^a-z]"),
    "subsec_begin": re.compile(r"\\subsection[^a-z]"),
    "subsubsec_begin": re.compile(r"\\subsubsection[^a-z]"),
    "cap_begin": re.compile(r"\\caption[^a-z]"),
    "footnote_begin1": re.compile(r"\\footnote[^a-z]"),
    "footnote_begin2": re.compile(r"\\footnotetext[^a-z]"),
    "thanks_begin": re.compile(r"\\thanks[^a-z]"),
    "thanksref_begin": re.compile(r"\\thanksref[^a-z]"),
    "ref_1": re.compile(r"\\bibliography[^a-z]"),
    "ref_2": re.compile(r"\\begin\{thebibliography\}"),
}

def append_predefined_color(latex_contents, latex_file, Insert_Colors):
    """Add package to tex lines
    """
    cur_id = 0
    while cur_id < len(latex_contents):
        if r"\documentclass" in latex_contents[cur_id] or r"\usepackage" in latex_contents[cur_id]:
            find_id = cur_id
            break
        cur_id += 1
    if cur_id == len(latex_contents):
        logger.error(r"No \documentclass or \usepackage found in ## {} ##".format(latex_file))
    latex_contents = latex_contents[:cur_id+1] + Insert_Colors + latex_contents[cur_id+1:]
    return latex_contents

def add_color_brace(latex_contents, brace_node, command_start, color_name, inner_outer = 'inner'): 
    """ Add \color{NAME} to right of {  *OR*  Add \color{NAME} to warp this command

    Args:
        latex_contents (List of string): all tex lines in tex file
        brace_node (Left_Brace): contain the outer brace info
        command_start (int, int): contain begining line and begining position of this command
        color_name (string): name of color
        inner_outer (optional, `inner` or `outer`)
    Exsample:
        inner_outer = 'inner' : 
            "\title{Attention is all you need} -> \title{{\color{RED}Attention is all you need}}"
        inner_outer = 'outer' : 
            "\bibliography{References} -> {\color{RED}\bibliography{References}}"
    """
    if inner_outer == 'inner':
        end_id = brace_node.right_brace_line
        end_pos = brace_node.right_brace_pos
        start_id = brace_node.line_id
        latex_contents[start_id] = latex_contents[start_id][:brace_node.position+1]\
            + "{\\color{" + color_name + "}" + latex_contents[start_id][brace_node.position+1:]
        if end_id == start_id:
            end_pos += len("{\\color{" + color_name + "}")
        latex_contents[end_id] = latex_contents[end_id][:end_pos+1] + \
            "}" + latex_contents[end_id][end_pos+1:]
    elif inner_outer == 'outer':
        command_line, command_pos = command_start
        end_id = brace_node.right_brace_line
        end_pos = brace_node.right_brace_pos
        latex_contents[command_line] = latex_contents[command_line][:command_pos] + \
            "{\\color{" + color_name + "}" + latex_contents[command_line][command_pos:]
        if end_id == command_line:
            end_pos += len("{\\color{" + color_name + "}")
        latex_contents[end_id] = latex_contents[end_id][:end_pos+1] + \
            "}" + latex_contents[end_id][end_pos+1:]
    return latex_contents

def add_color_begin_end_command(latex_contents, begin_brace, color_name, inner_outer = 'outer'):
    """ Add \color{NAME} to warp this start/end command

    Args:
        latex_contents (List of string): all tex lines in tex file
        begin_brace (Begin_Brace): contain the outer brace info
        color_name (string): name of color
    Exsample:
        "\begin{equ}{99} XXXX \end{equ} -> {\color{RED}\begin{equ}{99} XXXX \end{equ}}"
    """
    if inner_outer == 'outer':
        start_id = begin_brace.line_id
        latex_contents[start_id] = latex_contents[start_id][:begin_brace.st]\
            + "{\\color{" + color_name + "}" + latex_contents[start_id][begin_brace.st:]
        end_id = begin_brace.end_brace_line
        end_pos = begin_brace.end_brace_ed
        if end_id == start_id:
            end_pos += len("{\\color{" + color_name + "}")
        latex_contents[end_id] = latex_contents[end_id][:end_pos+1] + \
            "}" + latex_contents[end_id][end_pos+1:]
    elif inner_outer == 'inner':
        start_id = begin_brace.line_id
        latex_contents[start_id] = latex_contents[start_id][:begin_brace.ed+1]\
            + "\\color{" + color_name + "}" + latex_contents[start_id][begin_brace.ed+1:]
        # end_id = begin_brace.end_brace_line
        # end_pos = begin_brace.end_brace_st
        # if end_id == start_id:
        #     end_pos += len("{\\color{" + color_name + "}")
        # latex_contents[end_id] = latex_contents[end_id][:end_pos] + \
        #     "}" + latex_contents[end_id][end_pos:]
    return latex_contents

def get_all_brace_nodes(latex_contents, latex_file, st_line, st_pos): # Find the outer brace pair behind [st_line, st_pos]
    all_left_brace_nodes = []
    left_brace_stack = []
    not_find = True
    cur_id = st_line
    while cur_id < len(latex_contents) and not_find:
        for c_i, c in enumerate(latex_contents[cur_id]):
            if cur_id == st_line and c_i < st_pos: continue
            if c == '{':
                left_brace_stack.append(Left_Brace(cur_id, c_i, len(left_brace_stack))) 
                # use len(left_brace_stack) to represent depth of {}, means how many out-nested {} are in the stack
            elif c == '}':
                if len(left_brace_stack) == 0:
                    logger.error("Text containing unclosed brace pair! {} : {}".format(latex_contents[cur_id], latex_file))
                    continue
                left_brace_stack[-1].set_right_brace(cur_id, c_i)
                all_left_brace_nodes.append(left_brace_stack.pop())
            if len(all_left_brace_nodes) > 0 and len(left_brace_stack) == 0:
                not_find = False
                break
        cur_id += 1
    return all_left_brace_nodes

def color_brace(latex_contents, latex_file, reg_name, color_name, inner_outer = 'inner'):
    cur_id = 0
    all_brace_node = []
    command_start = []
    while cur_id < len(latex_contents):
        matched_ = Match_Regex[reg_name].search(latex_contents[cur_id])
        if matched_ != None:
            st, ep = matched_.span()
            all_left_brace_nodes = get_all_brace_nodes(latex_contents, latex_file, cur_id, ep-1)
            all_brace_node.append(all_left_brace_nodes)
            command_start.append((cur_id, st))
        cur_id += 1
    for node_idx in range(len(all_brace_node)):
        latex_contents = add_color_brace(latex_contents, all_brace_node[node_idx][-1], command_start[node_idx], color_name, inner_outer = inner_outer)
    return latex_contents

def get_all_begin_brace_nodes(latex_contents, latex_file, str = None, search_str_bg = None, search_str_ed = None):
    # Search begin_end (depth=0)
    # Color begin_end (depth=0)
    cur_id = 0
    if str:
        search_str_bg = "\\begin{"+str+"}"
        search_str_ed = "\\end{"+str+"}"
    elif not (search_str_bg and search_str_ed):
        raise SystemExit('You can only specify search_str_bg and search_str_ed both or none')
    all_begin_brace_list = []
    begin_brace_stack = []
    begin_brace_list = []
    while cur_id < len(latex_contents):
        begin_search = 0
        pos_bg = latex_contents[cur_id].find(search_str_bg, begin_search)
        pos_ed = latex_contents[cur_id].find(search_str_ed, begin_search)
        while not (pos_bg < 0 and pos_ed < 0): # if find no begin and no end, escape this line
            if pos_bg > 0 and latex_contents[cur_id][pos_bg-1] == "\\": # don't handle \\\[0.15cm]
                begin_search = pos_bg+len(search_str_bg)
            elif pos_bg >= 0 and pos_ed < 0: # has begin but no end
                begin_brace_stack.append(Begin_Brace(cur_id, pos_bg, pos_bg+len(search_str_bg)-1, len(begin_brace_stack)))
                begin_search = pos_bg + len(search_str_bg)
            elif pos_bg < 0 and pos_ed >= 0: # has end but no begin
                if len(begin_brace_stack) == 0:
                    logger.error("Text containing unclosed begin brace pair! {} : {}".format(latex_contents[cur_id], latex_file))
                else:
                    begin_brace_stack[-1].set_end_brace(cur_id, pos_ed, pos_ed+len(search_str_ed)-1)
                    begin_brace_list.append(begin_brace_stack.pop())
                begin_search = pos_ed + len(search_str_ed)
            else: # has both
                assert pos_bg != pos_ed
                if pos_bg < pos_ed:
                    begin_brace_stack.append(Begin_Brace(cur_id, pos_bg, pos_bg+len(search_str_bg)-1, len(begin_brace_stack)))
                    begin_search = pos_bg + len(search_str_bg)
                elif pos_ed < pos_bg:
                    if len(begin_brace_stack) == 0:
                        logger.error("Text containing unclosed begin brace pair! {} : {}".format(latex_contents[cur_id], latex_file))
                    else:
                        begin_brace_stack[-1].set_end_brace(cur_id, pos_ed, pos_ed+len(search_str_ed)-1)
                        begin_brace_list.append(begin_brace_stack.pop())
                    begin_search = pos_ed + len(search_str_ed)
            if len(begin_brace_list) > 0 and len(begin_brace_stack) == 0:
                all_begin_brace_list.append(begin_brace_list[:])
                begin_brace_list = []
            pos_bg = latex_contents[cur_id].find(search_str_bg, begin_search)
            pos_ed = latex_contents[cur_id].find(search_str_ed, begin_search)
        cur_id += 1
    return all_begin_brace_list


def color_begin_end(latex_contents, latex_file, str, color_name, inner_outer = 'outer'): # This function needs the target command has closed brace in the same line
    all_begin_brace_list = get_all_begin_brace_nodes(latex_contents, latex_file, str=str)
    for begin_brace_list in all_begin_brace_list:
        latex_contents = add_color_begin_end_command(latex_contents, begin_brace_list[-1], color_name, inner_outer=inner_outer)
    return latex_contents

def get_all_dollar_symbol(latex_contents, latex_file):
    all_single_dollar_nodes = []
    left_dollar_stack = []
    cur_id = 0
    while cur_id < len(latex_contents):
        for c_i, c in enumerate(latex_contents[cur_id]):
            if c == '$':
                if len(left_dollar_stack) == 0:
                    left_dollar_stack.append(Left_Brace(cur_id, c_i, len(left_dollar_stack)))
                else:
                    left_dollar_stack[-1].set_right_brace(cur_id, c_i)
                    all_single_dollar_nodes.append(left_dollar_stack.pop())
        cur_id += 1
    if len(left_dollar_stack) != 0: # scanned the whole document, but still has one unpaired dollar command
        logger.error("Text containing unclosed dollar pair! {} : {}".format(latex_contents[left_dollar_stack[-1].line_id], latex_file))
    # Getting the $$ symbol
    all_double_dollar_node = []
    begin_brace_double_dollar = []
    for dollar_node in all_single_dollar_nodes:
        content = dollar_node.get_braced_content(latex_contents)
        if len(content) == 1 and content[0] == "$$":
            all_double_dollar_node.append(dollar_node)
    if len(all_double_dollar_node)%2 == 1: 
        logger.error("Text containing odd double-dollar pair! {} : {}".format(latex_contents[all_double_dollar_node[-1].line_id], latex_file))
    for idx in range(len(all_double_dollar_node)//2):
        ldd = all_double_dollar_node[2*idx] # left_double_dollar
        rdd = all_double_dollar_node[2*idx+1] # right_double_dollar
        begin_brace_double_dollar.append(
            Begin_Brace(ldd.line_id, ldd.position, ldd.right_brace_pos, ldd.depth, \
                rdd.line_id, rdd.position, rdd.right_brace_pos))
    return begin_brace_double_dollar

def color_normal_eqution(latex_contents, latex_file, color_name): # This function needs the target command has closed brace in the same line
    # Part 1: dealing the \[EUQ\] braced equation
    all_begin_brace_list = get_all_begin_brace_nodes(latex_contents, latex_file, search_str_bg = '\[', search_str_ed = '\]')
    for begin_brace_list in all_begin_brace_list:
        begin_brace = begin_brace_list[-1]
        content = begin_brace.get_braced_content(latex_contents)
        if len("".join(content)) <= 5: continue
        latex_contents = add_color_begin_end_command(latex_contents, begin_brace, color_name, inner_outer='outer')
        
    # Part 2: dealing the $$EUQ$$ braced equation, using the Left_Brace to complete
    begin_brace_double_dollar = get_all_dollar_symbol(latex_contents, latex_file)
    for begin_brace in begin_brace_double_dollar:
        content = begin_brace.get_braced_content(latex_contents)
        triggrt_word = ["\includegraphics"]
        is_equ = True
        for w in triggrt_word:
            if w in "".join(content):
                is_equ = False
        if not is_equ: continue
        latex_contents = add_color_begin_end_command(latex_contents, begin_brace, color_name, inner_outer='outer')
    return latex_contents

def match_each_line(latex_contents, reg):
    cur_id = 0
    while cur_id < len(latex_contents):
        match_ = reg.search(latex_contents[cur_id])
        if match_:
            return True
        cur_id += 1
    return False

def replace_begin_end_blank_line(latex_contents, latex_file, command_str, new_str):
    all_begin_brace_list = get_all_begin_brace_nodes(latex_contents, latex_file, str=command_str)
    for begin_brace_list in all_begin_brace_list:
        outer_begin_brace = begin_brace_list[-1]
        blank_line_ids = [
            i for i in range(outer_begin_brace.line_id, outer_begin_brace.end_brace_line) \
                if latex_contents[i] == ""
        ]
        blank_line_ids.sort(reverse=True)
        if new_str == "":
            for i in blank_line_ids:
                latex_contents = latex_contents[:i] + latex_contents[i+1:]
        else:
            for i in blank_line_ids:
                latex_contents[i] = new_str
    return latex_contents

def get_all_section_brace_node(latex_contents, latex_file):
    sec_reg_key = ["sec_begin", "subsec_begin", "subsubsec_begin"]
    cur_id = 0
    all_brace_node = []
    sec_reg_cors = []
    while cur_id < len(latex_contents):
        for sec_reg in sec_reg_key:
            matched_ = Match_Regex[sec_reg].search(latex_contents[cur_id])
            if matched_ != None:
                _, ep = matched_.span()
                all_left_brace_nodes = get_all_brace_nodes(latex_contents, latex_file, cur_id, ep-1)
                all_brace_node.append(all_left_brace_nodes)
                sec_reg_cors.append(sec_reg)
                break
        cur_id += 1
    return all_brace_node, sec_reg_cors

def color_para(latex_contents, latex_file):
    all_brace_node, sec_reg_cors = get_all_section_brace_node(latex_contents, latex_file)
    for i in range(len(all_brace_node)-1):
        cur_line = all_brace_node[i][-1].right_brace_line
        cur_pos = all_brace_node[i][-1].right_brace_pos
        latex_contents[cur_line] = latex_contents[cur_line][:cur_pos+1] \
            + r"{\color{MYPARA}" +  latex_contents[cur_line][cur_pos+1:]
        # Dealing next section position
        next_line = all_brace_node[i+1][-1].line_id
        next_sec_outer_brace = all_brace_node[i+1][-1]
        matched_ = Match_Regex[sec_reg_cors[i+1]].search(latex_contents[next_sec_outer_brace.line_id])
        st, _ = matched_.span()
        latex_contents[next_line] = latex_contents[next_line][:st] + "}" + latex_contents[next_line][st:]
        next_sec_outer_brace.shift(1)
    if len(all_brace_node) > 0:
        cur_line = all_brace_node[-1][-1].right_brace_line
        cur_pos = all_brace_node[-1][-1].right_brace_pos
        latex_contents[cur_line] = latex_contents[cur_line][:cur_pos+1] \
            + r"\color{MYPARA}" +  latex_contents[cur_line][cur_pos+1:]

    return latex_contents
    

def convert_latex(latex_file, colors_head):
    latex_contents = open_tex_file(latex_file) # get all latex lines with all annotations removed
    latex_contents = append_predefined_color(latex_contents, latex_file, colors_head)
    latex_contents = color_brace(latex_contents, latex_file, "title_begin", "MYTITLE", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "abs_begin", "MYPARA", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "abs_begin", "MYABS", inner_outer = 'outer')
    latex_contents = color_brace(latex_contents, latex_file, "author_begin", "MYAUTHOR", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "mail_begin", "MYMAIL", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "institute_begin", "MYAFFILI", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "address_begin", "MYAFFILI", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "affili_begin", "MYAFFILI", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "sec_begin", "MYSECTION", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "subsec_begin", "MYSUBSECTION", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "subsubsec_begin", "MYSUBSUBSECTION", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "footnote_begin1", "MYFOOTER", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "footnote_begin2", "MYFOOTER", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "thanks_begin", "MYFOOTER", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "thanksref_begin", "MYFOOTER", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "cap_begin", "MYCAP", inner_outer = 'inner')
    latex_contents = color_brace(latex_contents, latex_file, "ref_1", "MYREF", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "array", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "eqnarray", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "eqnarray*", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "align", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "align*", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "equation", "MYEQU", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "equation*", "MYEQU", inner_outer = 'outer')
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "array", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "eqnarray", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "eqnarray*", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "align", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "align*", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "equation", "%")
    latex_contents = replace_begin_end_blank_line(latex_contents, latex_file, "equation*", "%")
    latex_contents = color_normal_eqution(latex_contents, latex_file, "MYEQU")
    latex_contents = color_begin_end(latex_contents, latex_file, "thebibliography", "MYREF", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "algorithm", "MYALG", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "algorithm*", "MYALG", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "algorithmic", "MYALG", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "tabular", "MYTAB", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "tabular*", "MYTAB", inner_outer = 'outer')
    latex_contents = color_begin_end(latex_contents, latex_file, "abstract", "MYPARA", inner_outer = 'inner')
    latex_contents = color_begin_end(latex_contents, latex_file, "abstract", "MYABS", inner_outer = 'outer')
    latex_contents = color_para(latex_contents, latex_file)
    return latex_contents

def run_cmd(cmd_lists, timeout=15):
    # Run shell command
    try:
        out_bytes = subprocess.check_output(cmd_lists, stderr=subprocess.STDOUT, timeout=timeout)
        res_code = 0
        msg = out_bytes.decode('utf-8')
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        try: 
            out_bytes = out_bytes.decode('utf-8').split("\n")
            out_bytes = [info_line for info_line in out_bytes if 'LaTeX Error' in info_line or 'undefined' in info_line]
            out_bytes = "\n".join(out_bytes[:10])
        except:
            out_bytes = "UTF-8 decode failed"
        msg = "[ERROR]CallError ：" + out_bytes
        res_code = e.returncode
    except subprocess.TimeoutExpired as e:
        res_code = 100
        msg = "[ERROR]Timeout : " + str(e)
    except Exception as e:
        res_code = 200
        msg = "[ERROR]Unknown Error : " + str(e)
    return res_code, msg

def compile_pdf(tex_file): # compile twice
    # Trying to compile tex code into pdf file, 1st
    res_code, msg = run_cmd(['pdflatex', '-synctex=1', '-interaction=nonstopmode', tex_file])
    if res_code == 0:
        logger.info("Compile tex to pdf first time {} !".format(tex_file))
    else:
        logger.error("Compile failed {} !".format(tex_file))
        logger.error("Error message : {}".format(msg))
    # Trying to compile tex code into pdf file, 2nd
    res_code, msg = run_cmd(['pdflatex', '-synctex=1', '-interaction=nonstopmode', tex_file])
    if res_code == 0:
        logger.info("Compile tex to pdf second time {} !".format(tex_file))
    else:
        logger.error("Compile failed {} !".format(tex_file))
        logger.error("Error message : {}".format(msg))

def shell_script(pdf_id2tex_file_name):
    success_compiled = {}
    pbar = tqdm.tqdm(pdf_id2tex_file_name.keys())
    for folder_path in pbar:
        try:
            pbar.set_description("Processing {}".format(folder_path))
            tex_file = pdf_id2tex_file_name[folder_path]
            # Entering target folder
            os.chdir(folder_path)
            logger.info("Entered {} !".format(folder_path))
            # Trying to compile tex code into pdf file
            shutil.copy(tex_file+".black", tex_file)
            compile_pdf(tex_file)
            if os.path.exists(tex_file[:-4]+".pdf"):
                shutil.move(tex_file[:-4]+".pdf", tex_file[:-4]+".black.pdf")
                success_compiled[folder_path] = [("black", tex_file[:-4]+".black.pdf")]
            shutil.copy(tex_file+".color", tex_file)
            compile_pdf(tex_file)
            if os.path.exists(tex_file[:-4]+".pdf"):
                shutil.move(tex_file[:-4]+".pdf", tex_file[:-4]+".color.pdf")
                if isinstance(success_compiled[folder_path], list):
                    success_compiled[folder_path].append(("color", tex_file.replace(".tex", ".color.pdf")))
                else:
                    success_compiled[folder_path] = [("color", tex_file.replace(".tex", ".color.pdf"))]
        except:
            continue
    return success_compiled

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex_base_folder', type=str, \
        help='The parent folder saving all tex files.')
    parser.add_argument('--pdf_id2tex_file_name', type=str, \
        help='Saving pdf ids mapping to corresponding tex files.')
    parser.add_argument('--success_compiled', type=str, \
        help='Saving susscessfully compiled pdf folder.')
    parser.add_argument('--renewed_compiled', type=str, \
        help='Saving renewed compiled pdf folder.')
    args = parser.parse_args()
    return args

def compile_all():
    args = parse_args()
    pdf_id2tex_file_name = {}
    all_list = os.listdir(args.tex_base_folder)
    all_list.sort()
    pbar = tqdm.tqdm(all_list)
    for base_folder in pbar:
        pbar.set_description("Processing {}".format(base_folder))
        folder = os.path.join(args.tex_base_folder, base_folder)
        try:
            main_tex_file = get_main_tex_path(folder)
            if main_tex_file == "":
                continue
            if not os.path.exists(main_tex_file+".raw"): # If not exists .raw file, copy the origin tex file as raw file, saving the original tex files without color
                shutil.copy(main_tex_file, main_tex_file+".raw")
            renewed_latex_contents = convert_latex(main_tex_file+".raw", Insert_Colors)
            open(main_tex_file+".color", "w").writelines([line+'\n' for line in renewed_latex_contents])
            renewed_latex_contents = convert_latex(main_tex_file+".raw", Insert_Black)
            open(main_tex_file+".black", "w").writelines([line+'\n' for line in renewed_latex_contents])
            pdf_id2tex_file_name[folder] = main_tex_file
        except:
            continue
    json.dump(pdf_id2tex_file_name, open(args.pdf_id2tex_file_name, "w"), ensure_ascii=False)
    success_compiled = shell_script(pdf_id2tex_file_name)
    json.dump(success_compiled, open(args.success_compiled, "w"), ensure_ascii=False)

def compile_renewed():
    args = parse_args()
    if not os.path.exists(args.pdf_id2tex_file_name):
        logger.error("Should provide {} before calling compile_renewed !".format(args.pdf_id2tex_file_name))
    jd = json.load(open(args.pdf_id2tex_file_name))
    pbar = tqdm.tqdm(list(jd.keys()))
    renewed_compiled = {}
    for folder in pbar:
        pbar.set_description("Processing {}".format(os.path.basename(folder)))
        tex_file = jd[folder]
        raw_file, color_file, black_file = tex_file+".raw", tex_file+".color", tex_file+".black"
        lack_file = False
        for f in [raw_file, color_file, black_file]:
            if not os.path.exists(f):
                lack_file = True
                logger.warning("Lacking file {}! Skip this tex folder...".format(f))
        if lack_file: continue
        old_color_tex_contents = open_tex_file(color_file)
        new_color_tex_contents = convert_latex(raw_file, Insert_Colors)
        old_black_tex_contents = open_tex_file(black_file)
        new_black_tex_contents = convert_latex(raw_file, Insert_Black)
        if old_color_tex_contents != new_color_tex_contents or old_black_tex_contents != new_black_tex_contents: # This version convert tex file differently, renew tex and pdf file.
            logger.info("Found version compiled different: {} !".format(folder))
            try:
                # Renew color/black file
                open(color_file, "w").writelines([line+'\n' for line in new_color_tex_contents])
                open(black_file, "w").writelines([line+'\n' for line in new_black_tex_contents])
                # Entering target folder
                os.chdir(folder)
                logger.info("Entered {} !".format(folder))
                # Trying to compile tex code into pdf file, black
                shutil.copy(black_file, tex_file)
                compile_pdf(tex_file)
                if os.path.exists(tex_file[:-4]+".pdf"):
                    shutil.move(tex_file[:-4]+".pdf", tex_file[:-4]+".black.pdf")
                    renewed_compiled[folder] = [("black", tex_file[:-4]+".black.pdf")]
                # Trying to compile tex code into pdf file, color
                shutil.copy(color_file, tex_file)
                compile_pdf(tex_file)
                if os.path.exists(tex_file[:-4]+".pdf"):
                    shutil.move(tex_file[:-4]+".pdf", tex_file[:-4]+".color.pdf")
                    renewed_compiled[folder] = [("color", tex_file[:-4]+".color.pdf")]
            except:
                continue
    json.dump(renewed_compiled, open(args.renewed_compiled, "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    compile_all()
    compile_renewed()