from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import os
import pandas as pd
from surprise import Dataset, Reader, evaluate, print_perf
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp
from sklearn.metrics import mean_squared_error

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"


# train = [
#     {"user": "1", "item": "5", "age": 19},
#     {"user": "2", "item": "43", "age": 33},
#     {"user": "3", "item": "20", "age": 55},
#     {"user": "4", "item": "10", "age": 20},
# ]
# v = DictVectorizer()
# X = v.fit_transform(train)
# print(X.toarray())
# y = np.repeat(1.0, X.shape[0])
# fm = pylibfm.FM()
# fm.fit(X, y)
# fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))


# ml-100k
# Read in data
def loadData(filename, path="ml-100k/"):
    data = []
    y = []
    users = set()
    items = set()

    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({"user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return data, np.array(y), users, items


def convert_data(data, y):
    users = []
    items = []
    rates = []
    for i in range(len(y)):
        users.append(data[i]["user_id"])
        items.append(data[i]["movie_id"])
        rates.append(int(y[i]))
    return users, items, rates


def build_dataset(data_x, data_y):
    users, items, rates = convert_data(data_x, data_y)
    data = {"users": users, "items": items, "rates": rates}
    data = pd.DataFrame(data, columns=["users", "items", "rates"])
    reader = Reader(rating_scale=(0, 5))
    train_data = Dataset.load_from_df(df=data, reader=reader)
    result = train_data.build_full_trainset()
    # train_data.split(10)
    return result


def get_predictions(algo, data_x, data_y):
    users, items, rates = convert_data(data_x, data_y)
    preds = []
    for i in range(len(users)):
        pred = algo.predict(users[i], items[i])
        preds.append(pred.est)
    targets = np.asarray(data_y)
    preds = np.asarray(preds)
    for i in range(10):
        print "svd: ", i, targets[i], preds[i]
    return mean_squared_error(targets, preds)


(train_data, y_train, train_users, train_items) = loadData("ua.base", root + "datasets/recommend/ml-100k/")
(test_data, y_test, test_users, test_items) = loadData("ua.test", root + "datasets/recommend/ml-100k/")

trainset = build_dataset(train_data, y_train)
verbose = 2
algo = SVD()
algo.train(trainset)
print get_predictions(algo, test_data, y_test)

v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=10, num_iter=20, verbose=False, task="regression", initial_learning_rate=0.001,
                learning_rate_schedule="optimal")

fm.fit(X_train, y_train)

# Evaluate
preds = fm.predict(X_test)
for i in range(10):
    print "fm: ", i, y_test[i], preds[i]

print("FM MSE: %.4f" % mean_squared_error(y_test, preds))
