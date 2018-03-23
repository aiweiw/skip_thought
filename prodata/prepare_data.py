#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import

import codecs
import os
import re
import time
import sys

re_file_name = re.compile(r'^90\d+')

anyou_file_path = os.path.expanduser('~/fastText/anyou/minshi/src')
anyou_train_data = '../data/aytrain.txt'

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


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


def main():
    obj_src = None
    obj_tgt = None
    try:
        obj_tgt = codecs.open(anyou_train_data, 'w', 'utf-8')
        seq_file = list_file(anyou_file_path)
        for file_i in seq_file:
            if not re_file_name.match(file_i):
                continue

            file_src = anyou_file_path + '/' + file_i
            obj_src = codecs.open(file_src, 'r', 'utf-8')
            line = obj_src.readline()
            while True:
                if line:
                    break
                line = obj_src.readline()
            obj_tgt.write((line.strip(' \r\n') + '\n').encode('utf-8'))
            obj_tgt.flush()
            obj_src.close()
            time.sleep(0)

        obj_tgt.close()
    except Exception, e:
        print Exception, e
    finally:
        if obj_src and not obj_src.closed:
            obj_src.close()
        if obj_tgt and not obj_tgt.closed:
            obj_tgt.close()


if __name__ == '__main__':
    main()
    print('ok')