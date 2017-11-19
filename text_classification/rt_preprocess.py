#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import os
import sys
from collections import OrderedDict

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

result = {}
for line in open(root + "datasets/text/wangke/rt-polarity.pos"):
    result[line.strip()] = "1"
for line in open(root + "datasets/text/wangke/rt-polarity.neg"):
    result[line.strip()] = "0"

nb_samples = len(result)
print "nb_samples: ", nb_samples

index = 0
with open(root + "datasets/text/wangke/train.txt", "w") as ftrain:
    with open(root + "datasets/text/wangke/test.txt", "w") as ftest:
        for sentence, tag in result.items():
            if index < nb_samples - nb_samples / 9:
                ftrain.write(tag + " __content__ " + sentence + "\n")
            else:
                ftest.write(tag + " __content__ " + sentence + "\n")
            index += 1
