#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-12
import sys
import os

import keras
import numpy as np
from keras import Sequential
from keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Activation, Embedding

reload(sys)
sys.setdefaultencoding('utf8')

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/"


def build_model(input_length, input_vocab_size, output_length, output_vocab_size,
                embedding_size=128, RNN=LSTM, HIDDEN_SIZE=128, LAYERS=2):
    print 'Build model...'

    model = Sequential()
    model.add(Embedding(embedding_size, input_vocab_size, input_length=input_length))
    model.add(RNN(HIDDEN_SIZE, input_shape=(input_length, input_vocab_size)))
    model.add(RepeatVector(output_length))

    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(TimeDistributed(Dense(output_vocab_size)))
    model.add(Activation('softmax'))
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['acc'])
    model.summary()
    return model


if __name__ == '__main__':
    [X_train, Y_train, X_test, Y_test, zh_word2idx, zh_idx2word,
     zh_vocab, en_word2idx, en_idx2word, en_vocab] = np.load('pack.npz.npy')
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)
    # print Y_train.shape
    # exit()

    # for i in range(len(X_train)):
    #     print " ".join([zh_idx2word[index] for index in X_train[i]])
    #     print " ".join([en_idx2word[index] for index in Y_train[i]])

    # for i in range(len(X_test)):
    #     print " ".join([zh_idx2word[index] for index in X_test[i]])
    #     print " ".join([en_idx2word[index] for index in Y_test[i]])

    input_seq_len = X_train.shape[1]
    output_seq_len = Y_train.shape[1]

    zh_vocab_size = len(zh_vocab) + 2  # + <pad>, <ukn>
    en_vocab_size = len(en_vocab) + 4  # + <pad>, <ukn>, <eos>, <go>
    model = build_model(input_seq_len, zh_vocab_size, output_seq_len, en_vocab_size)
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), verbose=1)
    model.save('seq2seq_ks.hdf5')
