#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-12
import os
import sys

import numpy as np
from keras.models import load_model

reload(sys)
sys.setdefaultencoding('utf8')

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"

if __name__ == '__main__':
    [X_train, Y_train, X_test, Y_test, zh_word2idx, zh_idx2word,
     zh_vocab, en_word2idx, en_idx2word, en_vocab] = np.load('pack.npz.npy')

    model = load_model('seq2seq_ks.hdf5')

    Y_pred = model.predict_classes(X_test, verbose=0)
    Y_pred = np.squeeze(Y_pred)
    Y_test = np.squeeze(Y_test)

    for i in range(len(X_test)):
        print "*" * 15
        print " ".join([zh_idx2word[index] for index in X_test[i]])
        print " ".join([en_idx2word[index] for index in Y_pred[i]])
        print " ".join([en_idx2word[index] for index in Y_test[i]])
