#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import sys
import os
import numpy as np
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"


def create_dataset(in_file, max_size):
    sent_line, tag_line = [], []
    for line in open(in_file, 'r'):
        sent_tag = line.split("__content__")
        sent_line.append(sent_tag[1].strip())
        tag_line.append(sent_tag[0].strip())

    sent_vocab_dict = Counter(word for sentence in sent_line for word in sentence.split())
    tag_vocab_dict = Counter(word for sentence in tag_line for word in sentence.split())

    sent_vocab_cnt = map(lambda x: (x[0], x[1]), sorted(sent_vocab_dict.items(), key=lambda x: -x[1]))
    sent_vocab_cnt = filter(lambda x: x[1] >= max_size, sent_vocab_cnt)
    tag_vocab_cnt = map(lambda x: (x[0], x[1]), sorted(tag_vocab_dict.items(), key=lambda x: -x[1]))

    sent_vocab = map(lambda x: x[0], sent_vocab_cnt)
    tag_vocab = map(lambda x: x[0], tag_vocab_cnt)

    start_idx = 2
    sent_word2idx = dict([(word, idx + start_idx) for idx, word in enumerate(sent_vocab)])
    sent_word2idx['<ukn>'] = 0
    sent_word2idx['<pad>'] = 1
    sent_idx2word = dict([(idx, word) for word, idx in sent_word2idx.iteritems()])

    tag_word2idx = dict([(word, idx) for idx, word in enumerate(tag_vocab)])
    tag_idx2word = dict([(idx, word) for word, idx in tag_word2idx.iteritems()])

    x = [[sent_word2idx.get(word, 0) for word in sentence.split()] for sentence in sent_line]
    y = [[tag_word2idx.get(word) for word in sentence.split()] for sentence in tag_line]

    # for sent,tag in zip(x,y):
    #     print " ".join([sent_idx2word.get(word) for word in sent])
    #     print " ".join([tag_idx2word.get(tta) for tta in tag])
    # for k,v in sent_word2idx.items():
    #     print k,v

    return x, y, sent_word2idx, sent_idx2word, sent_vocab, tag_word2idx, tag_idx2word, tag_vocab


# load data with word dictionary
def load_data(in_file, zh_word2idx, en_word2idx):
    sent_line, tag_line = [], []

    for line in open(in_file, 'r'):
        sent_tag = line.split("__content__")
        each = []
        for word in sent_tag[1].strip().split():
            each.append(sent_word2idx.get(word, 0))

        sent_line.append(each)

        each = []
        for tta in sent_tag[0].strip().split():
            each.append(tag_word2idx[tta])

        tag_line.append(each)

    return sent_line, tag_line


def data_padding(x, y, sent_word2idx, tag_word2idx, length=15):
    classes = len(tag_word2idx)
    for i in range(len(x)):
        if len(x[i]) >= length:
            x[i] = x[i][:length]
        else:
            x[i] = x[i] + (length - len(x[i])) * [sent_word2idx['<pad>']]
        tag_vec = [0] * classes
        for tag in y[i]:
            tag_vec[tag] = 1
        y[i] = tag_vec


if __name__ == '__main__':
    length = 20
    max_size = 0
    train_file = root + "datasets/text/wangke/train.txt"
    test_file = root + "datasets/text/wangke/test.txt"
    X_train, Y_train, sent_word2idx, sent_idx2word, sent_vocab, tag_word2idx, tag_idx2word, tag_vocab = create_dataset(
        train_file,max_size=max_size)
    X_test, Y_test = load_data(test_file, sent_word2idx, tag_word2idx)

    data_padding(X_train, Y_train, sent_word2idx, tag_word2idx, length=length)
    data_padding(X_test, Y_test, sent_word2idx, tag_word2idx, length=length)

    for sent, tag in zip(X_train, Y_train):
        print " ".join([sent_idx2word[word] for word in sent])
        ts = []
        for i, tta in enumerate(tag):
            if tta:
                ts.append(i)
        print " ".join([tag_idx2word[ttaS] for ttaS in ts])
    print "nb_trains: ",len(X_train)
    print "nb_tests: ",len(X_test)
    print "nb_classes: ",len(tag_vocab)
    # for key,tag in tag_idx2word.items():
    #     print key,tag
    # for x in X_train:
    #     print len(x)

    X_train = np.asarray(X_train, dtype=np.int)
    Y_train = np.asarray(Y_train, dtype=np.int)
    X_test = np.asarray(X_test, dtype=np.int)
    Y_test = np.asarray(Y_test, dtype=np.int)

    np.save(root + 'datasets/text/wangke/pack.npz', np.array([X_train, Y_train, X_test, Y_test,
                                                              sent_word2idx, sent_idx2word, sent_vocab, tag_word2idx,
                                                              tag_idx2word, tag_vocab]))
