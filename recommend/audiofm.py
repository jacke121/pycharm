#!/usr/bin python
# -*- coding: utf-8 -*-
# Created by Weihang Huang on 2017/9/2

import pandas as pd
import numpy as np
import pickle
import os

import time
from surprise import SVD, Reader, accuracy, KNNBasic,SVDpp
from surprise import Dataset,evaluate
from surprise import evaluate, print_perf

# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
# data = Dataset.load_builtin('ml-100k')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
file_path = root + "/datasets/recommend/audiofm/user_artist_data.txt"
result_path = root + "/datasets/recommend/result/audiofm.bin"
users = []
items = []
rates = []
index = 0
for line in open(file_path):
    if index >= 100000:
        break
    index = index + 1

    user, item, rate = line.split()[0:3]
    if int(rate) > 10:
        rate = '10'
    users.append(int(user))
    items.append(int(item))
    rates.append(int(rate))
data = {"users": users, "items": items, "rates": rates}
data = pd.DataFrame(data)
data.info()
print(data.describe())

data.to_csv("abc.txt", index=None, header=None, columns=["users", "items", "rates"])

reader = Reader(line_format='user item rating', rating_scale=(0, 10), sep=',')

data = Dataset.load_from_file("abc.txt", reader=reader)

data.split(n_folds=10)
# sim_options = {'name': 'cosine',
#                'user_based': False  # compute  similarities between items
#                }
# algo = KNNBasic(sim_options=sim_options)
# We'll use the famous SVD algorithm.
algo = SVD(verbose=True)
for _ in range(10):
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

dump_obj = {'predictions': perf,
            'algo': algo
            }
pickle.dump(dump_obj, open(result_path, 'wb'))
exit()


start_time = time.time()
for trainset, testset in data.folds():
    # train and test algorithm.
    algo.train(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
# Evaluate performances of our algorithm on the dataset.


dump_obj = {'predictions': predictions,
            'algo': algo
            }
pickle.dump(dump_obj, open(result_path, 'wb'))
end_time = time.time()
print(trainset.all_users())
print(trainset.all_items())
print(trainset.all_ratings())

uid = str(1000002)  # raw user id (as in the ratings file). They are **strings**!
iid = str(100006)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=0, verbose=True)
print("cost time(s)", end_time - start_time)

