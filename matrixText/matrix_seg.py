# -*- coding:utf-8 -*-
import codecs

import os
import time
import sys
import re
import urllib
import urllib2

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

sys.path.append('../..')

"""
segment
input anyou-file
1. anyou root-node file
2. anyou every file
"""

patt_time = re.compile(r'^(((\s?\d\s?){4}年(\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){4}年(\s?\d\s?){1,2}月)|'
                       r'((\s?\d\s?){4}年)|((\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){1,2}日))'
                       r'(底|初|末|以来|年底|年初|年末|月底|月初|月末|开始|期间|份)?$')

patt_num = re.compile(r'\d+\s*\，\s*\d+')

patt_sym = re.compile(r'\，\s*\w+\s*[\。\，]')

patt_seg = re.compile(r'\。|\？|\！|\，')


def list_file(path):
    """
    :param path:
    :return:
    """
    file_list = []
    files = os.listdir(path)
    for f in files:
        if f[0] == '.':
            pass
        else:
            file_list.append(f)
    return file_list


def seg_cont_ansj(to_seg_content, ansj_serve_url=None, out_filter=0):
    """
    :param to_seg_content:
    :param ansj_serve_url:
    :param out_filter: 0 - seg_vec, 1 - seg_filter, * - seg_vec, seg_filter
    :return:
    """
    if not to_seg_content or not ansj_serve_url:
        return None

    seg_ontent = dict()
    seg_ontent['segTent'] = to_seg_content
    data_urlencode = urllib.urlencode(seg_ontent)
    req = urllib2.Request(ansj_serve_url, data=data_urlencode)
    response = urllib2.urlopen(req)
    time.sleep(0.001)
    seg_res = response.read()

    seg_vec_tfidf = seg_res.split('-SEGMENT-')

    if out_filter == 0:
        # return segseg-words full
        return seg_vec_tfidf[0]
    elif out_filter == 1:
        # return seg-words filter
        return seg_vec_tfidf[1]
    else:
        # return seg-words full+filter
        return seg_vec_tfidf[0], seg_vec_tfidf[1]

    # seg_vec = seg_vec_tfidf[0].split()
    # seg_filter = seg_vec_tfidf[1].split()

    # if out_filter == 0:
    #     # return segseg-words full
    #     return seg_vec
    # elif out_filter == 1:
    #     # return seg-words filter
    #     return seg_filter
    # else:
    #     # return seg-words full+filter
    #     return seg_vec, seg_filter


def seg_sent_base(sentence):
    """
    :param sentence:
    :return:
    """
    match = re.compile(r'\。|\.|\，|\,')
    res_segment = match.split(sentence)

    return res_segment


def seg_sentence(sentence, min_words_num=5):
    """ Segment sentence

    Args:
        sentence: sentence
        min_words_num: a sentence contains min_words_num words
    Returns:
        segment sentence
    """
    if not sentence:
        return None

    if not isinstance(sentence, str):
        sentence = str(sentence)

    num_seg_all = patt_num.findall(sentence)
    while num_seg_all:
        for num_one in num_seg_all:
            sentence = sentence.replace(num_one, num_one.replace('，', ','))
        num_seg_all = patt_num.findall(sentence)

    sym_seg_all = patt_sym.findall(sentence)
    while sym_seg_all:
        for sym_one in sym_seg_all:
            sentence = sentence.replace(sym_one, sym_one.replace('，', ',', 1))
        sym_seg_all = patt_sym.findall(sentence)

    res_segment = patt_seg.split(sentence)
    res_seg_len = len(res_segment)

    min_sent_len = min_words_num * len('。')
    for i in range(res_seg_len):
        if i >= res_seg_len:
            break
        if patt_time.match(res_segment[i].replace(' ', '')) or len(res_segment[i].replace(' ', '')) < min_sent_len:
            if len(res_segment) == 1:
                break
            if i < len(res_segment) - 1:
                res_segment[i] += ',' + res_segment[i + 1]
                res_segment.remove(res_segment[i + 1])
            else:
                res_segment[i - 1] += ',' + res_segment[i]
                res_segment.remove(res_segment[i])

            res_seg_len = len(res_segment)

    return res_segment

