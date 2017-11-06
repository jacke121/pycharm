#!/usr/bin python
# -*- coding: utf-8 -*-
# Created by Weihang Huang on 2017/9/10
import sys
import os
import surprise
import pickle

reload(sys)
sys.setdefaultencoding('utf-8')

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
result_path = root + "/datasets/recommend/result/audiofm.bin"

if __name__ == '__main__':
    dict_model=  pickle.load(open(result_path, 'rb'))
    algo = dict_model['algo']
    index = 0
    count = 100
    for line in open("abc.txt"):
        if index>=count: break
        index = index +1
        uid, iid, rate = map(lambda x:x.strip(),line.split(","))[0:3]

        # get a prediction for specific users and items.
        pred = algo.predict(uid, iid, r_ui=float(rate), verbose=True)

    pred = algo.predict("1000002", "1000675", r_ui=float(rate), verbose=True)
    # pred = algo.predict("10000000000000000002", "1000675", r_ui=float(rate), verbose=True)
    # pred = algo.predict("10000000000000000003", "1000675", r_ui=float(rate), verbose=True)
    # pred = algo.predict("10000000000000000005", "1000675", r_ui=float(rate), verbose=True)
    pred = algo.predict("1000002", "11110006750000000000", r_ui=float(rate), verbose=True)
    pred = algo.predict("1000002", "11110006750000000010", r_ui=float(rate), verbose=True)
