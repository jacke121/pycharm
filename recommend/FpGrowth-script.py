#!/usr/bin python
# -*- coding: utf-8 -*-
# Created by Weihang Huang on 2017/10/29
import sys

import pyfpgrowth
from collections import defaultdict
from surprise import Reader, Dataset

reload(sys)
sys.setdefaultencoding('utf-8')

transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]

dict_value = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
transactions = [map(lambda x: dict_value[x], liste) for liste in transactions]
print (transactions)
patterns = pyfpgrowth.find_frequent_patterns(transactions, 4)
print (patterns)
rules = pyfpgrowth.generate_association_rules(patterns, 0.2)
print (rules)

reader = Reader(line_format='user item rating', rating_scale=(0, 10), sep=',')

data = Dataset.load_from_file("abc.txt", reader=reader)

trainset = data.build_full_trainset()

# user raw id, item raw id, translated rating, time stamp
currentuid = None
fpdata = defaultdict(list)
for uid, iid, _ in trainset.all_ratings():
    fpdata[uid].append(iid)

transactions = []
for _, v in fpdata.items():
    transactions.append(v[:20])
print (transactions)

patterns = pyfpgrowth.find_frequent_patterns(transactions, 40)
print (patterns)
rules = pyfpgrowth.generate_association_rules(patterns, 0.2)
print (rules)
