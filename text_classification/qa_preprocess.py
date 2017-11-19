#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import os
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

with open(root + "datasets/text/wangke/train.txt", "w") as ftrain:
    with open(root + "datasets/text/wangke/test.txt", "w") as ftest:
        index = 0
        for line in open(root + "datasets/text/wangke/qa_set.txt", "r"):
            line_split = line.strip().split()
            tag = line_split[0][:line_split[0].index(":")]
            sentence = " ".join(line_split[1:])
            print sentence
            print tag
            print line.strip()

            if index < 900:
                ftrain.write(tag + " __content__ " + sentence + "\n")
            else:
                ftest.write(tag + " __content__ " + sentence + "\n")
            index += 1
