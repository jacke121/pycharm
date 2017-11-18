#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import os
import sys
from collections import OrderedDict

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

import pandas as pd
import jieba
import re


def clean_str_new(string):
    string = re.sub(r"^[\\/\"]", "", string)
    string = re.sub(r"^（[0-9]）", "", string)
    string = re.sub(r"^\([0-9]\)", "", string)
    string = re.sub(r"^[0-9]）", "", string)
    string = re.sub(r"^[0-9]\.", "", string)
    string = re.sub(r"^[0-9],", "", string)
    string = re.sub(r"^[0-9]、", "", string)
    string = re.sub(r"^[0-9]，", "", string)
    string = re.sub(r"^[第]?[0-9一二三四五][，：。:]", "", string)
    string = re.sub(r"^[0-9]\.", "", string)
    string = re.sub(r"^①", "", string)
    string = re.sub(r"^③", "", string)
    string = re.sub(r"^⑤", "", string)
    string = re.sub(r"^⑥", "", string)
    string = re.sub(r"^⑦", "", string)
    string = re.sub(r"^一，", "", string)
    string = re.sub(r"^[0-9]丶", "", string)
    string = re.sub(r"^4丶", "", string)
    string = re.sub(r"^(第)?(一|二|三|四)(，|：)", "", string)
    string = re.sub(r"^(0-9)[^0-9]", "", string)
    string = re.sub(r"^我：", "", string)
    string = re.sub(r"^其(一|二)，", "", string)
    string = re.sub(r"^，", "", string)
    string = re.sub(r"^；5，，", "", string)

    string = string.replace("\n", " ")

    return string


train_data = pd.read_csv(root + 'datasets/text/wangke/wangke_train.csv')
train_data.columns = ['context', '1Title', '2Title', '3Title', '1', '2', '3']
train_data = train_data.iloc[:, :4]
train_data.dropna(inplace=True)

sentences = [" ".join(jieba.cut(clean_str_new(line.strip()))).strip()
             for line in train_data['context']]
tags = [tag.strip() for tag in train_data['1Title'].values]
result = OrderedDict()
for sentence, tag in zip(sentences, tags):
    if sentence in result and tag not in result[sentence]:
        result[sentence] = result[sentence] + " " + tag
    else:
        result[sentence] = tag

index = 0
with open(root + "datasets/text/wangke/train.txt", "w") as ftrain:
    with open(root + "datasets/text/wangke/test.txt", "w") as ftest:
        for sentence, tag in result.items():
            if index < 8000:
                ftrain.write(tag + " __content__ " + sentence + "\n")
            else:
                ftest.write(tag + " __content__ " + sentence + "\n")
            index += 1
