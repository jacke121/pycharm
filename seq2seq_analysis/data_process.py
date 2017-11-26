#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-12
import sys
import os
import numpy as np
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"


# zh-en translation generate word dictionary and training data
def create_dataset(in_file):
    zh_line, en_line = [], []
    for i, line in enumerate(open(in_file, 'r')):
        if i % 2 == 0:
            zh_line.append(line[:-1])
        else:
            en_line.append(line[:-1])
    en_vocab_dict = Counter(word.strip(',." ;:)(][!-') for sentence in en_line for word in sentence.split())
    zh_vocab_dict = Counter(word.strip(',." ;:)(][!-') for sentence in zh_line for word in sentence.split())

    en_vocab = map(lambda x: x[0], sorted(en_vocab_dict.items(), key=lambda x: -x[1]))
    zh_vocab = map(lambda x: x[0], sorted(zh_vocab_dict.items(), key=lambda x: -x[1]))

    en_vocab = en_vocab[:3000]
    zh_vocab = zh_vocab[:3000]

    start_idx = 2
    zh_word2idx = dict([(word, idx + start_idx) for idx, word in enumerate(zh_vocab)])
    zh_word2idx['<pad>'] = 0
    zh_word2idx['<ukn>'] = 1

    zh_idx2word = dict([(idx, word) for word, idx in zh_word2idx.iteritems()])

    start_idx = 4
    en_word2idx = dict([(word, idx + start_idx) for idx, word in enumerate(en_vocab)])
    en_word2idx['<pad>'] = 0
    en_word2idx['<ukn>'] = 1
    en_word2idx['<go>'] = 2
    en_word2idx['<eos>'] = 3

    en_idx2word = dict([(idx, word) for word, idx in en_word2idx.iteritems()])

    x = [[zh_word2idx.get(word.strip(',." ;:)(][!'), 0) for word in sentence.split()] for sentence in zh_line]
    y = [[en_word2idx.get(word.strip(',." ;:)(][!'), 0) for word in sentence.split()] for sentence in en_line]

    X = []
    Y = []
    for i in range(len(x)):
        n1 = len(x[i])
        n2 = len(y[i])
        n = n1 if n1 < n2 else n2
        if abs(n1 - n2) <= 0.3 * n:
            if n1 <= 15 and n2 <= 15:
                X.append(x[i])
                Y.append(y[i])
                # print " ".join([zh_idx2word[index] for index in x[i]])
                # print " ".join([en_idx2word[index] for index in y[i]])

    return X, Y, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab


# load data with word dictionary
def load_data(in_file, zh_word2idx, en_word2idx):
    zh_line, en_line = [], []
    for i, line in enumerate(open(in_file, 'r')):
        if i % 2 == 0:
            each = []
            for word in line[:-1].split():
                each.append(zh_word2idx.get(word.strip(',." ;:)(][!-'), 0))
            zh_line.append(each)
            # print " ".join([zh_idx2word[index] for index in each])
        else:
            # en_line.append([en_word2idx[word.strip(',." ;:)(][?!-')] for word in line[:-1].split()])
            each = []
            for word in line[:-1].split():
                each.append(en_word2idx.get(word.strip(',." ;:)(][!-'), 0))
            en_line.append(each)
            # print " ".join([en_idx2word[index] for index in each])

    return zh_line, en_line


def data_padding(x, y, zh_word2idx, en_word2idx, length=15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [zh_word2idx['<pad>']]
        y[i] = [en_word2idx['<go>']] + y[i] + [en_word2idx['<eos>']] + (length - len(y[i])) * [en_word2idx['<pad>']]


if __name__ == '__main__':
    length = 15
    train_file = root + "datasets/text/spoken_data/spoken.train"
    test_file = root + "datasets/text/spoken_data/spoken.valid"
    X_train, Y_train, zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab = create_dataset(
        train_file)
    X_test, Y_test = load_data(test_file, zh_word2idx, en_word2idx)

    data_padding(X_train, Y_train, zh_word2idx, en_word2idx, length=length)
    data_padding(X_test, Y_test, zh_word2idx, en_word2idx, length=length)

    X_train = np.asarray(X_train, dtype=np.int)
    Y_train = np.asarray(Y_train, dtype=np.int)
    X_test = np.asarray(X_test, dtype=np.int)
    Y_test = np.asarray(Y_test, dtype=np.int)

    np.save(root + "datasets/text/spoken_data/pack.npz.npy", np.array([X_train, Y_train, X_test, Y_test,
                              zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab]))

    # for i in range(len(X_train)):
    #     print " ".join([zh_idx2word[index] for index in X_train[i]])
    #     print " ".join([en_idx2word[index] for index in Y_train[i]])
    # exit()

    # input_seq_len = length
    # output_seq_len = length + 2
    #
    # zh_vocab_size = len(zh_vocab) + 2  # + <pad>, <ukn>
    # en_vocab_size = len(en_vocab) + 4  # + <pad>, <ukn>, <eos>, <go>
