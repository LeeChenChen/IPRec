# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 16:27
# @Author : lee
# @FileName : Logging.py

import sys


class Logging():
    def __init__(self, filename):
        self.filename = filename

    def print(self, str_log):
        filename = self.filename
        print(str_log)
        sys.stdout.flush()
        # with open(filename, 'a') as f:
        #     f.write("%s\r\n" % str_log)
        #     f.flush()