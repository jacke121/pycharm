#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import copy
import sys
import os
import tensorflow as tf
from tensorflow.contrib import rnn

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'


def fastText(input_x, vocab_size, embed_size, num_classes, initializer=tf.random_normal_initializer(stddev=0.1)):
    Embedding = tf.get_variable("Embedding", shape=[vocab_size, embed_size],
                                initializer=initializer)  # [vocab_size,embed_size]
    sentence_embeddings = tf.nn.embedding_lookup(Embedding, input_x)  # [None,self.sentence_len,self.embed_size]

    sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [None,self.embed_size]

    W_projection = tf.get_variable("W_projection", shape=[embed_size, num_classes],
                                   initializer=initializer)  # [embed_size,label_size]
    b_projection = tf.get_variable("b_projection", shape=[num_classes])  # [label_size]

    logits = tf.matmul(sentence_embeddings, W_projection) + b_projection  # [None, self.label_size]
    return logits


def textCNN(input_x, dropout_keep_prob, sequence_length, vocab_size, embed_size, num_classes, filter_sizes, num_filters,
            initializer=tf.random_normal_initializer(stddev=0.1)):
    num_filters_total = num_filters * len(filter_sizes)
    # tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
    Embedding = tf.get_variable("Embedding", shape=[vocab_size, embed_size],
                                initializer=initializer)  # [vocab_size,embed_size]

    embedded_words = tf.nn.embedding_lookup(Embedding, input_x)  # [None,sentence_length,embed_size]
    sentence_embeddings_expanded = tf.expand_dims(embedded_words, -1)  # [None,sentence_length,embed_size,1).

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("convolution-pooling-%s" % filter_size):
            filter = tf.get_variable("filter-%s" % filter_size, [filter_size, embed_size, 1, num_filters],
                                     initializer=initializer)
            conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
            b = tf.get_variable("b-%s" % filter_size, [num_filters])  # ADD 2017-06-09
            h = tf.nn.relu(tf.nn.bias_add(conv, b),
                           "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")  # shape:[batch_size, 1, 1, num_filters]
            pooled_outputs.append(pooled)
    h_pool = tf.concat(pooled_outputs, 3)  # shape:[batch_size, 1, 1, num_filters_total]
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # shape should be:[None,num_filters_total]

    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, keep_prob=dropout_keep_prob)  # [None,num_filters_total]

    # 5. logits(use linear layer)and predictions(argmax)
    with tf.name_scope("output"):
        W_projection = tf.get_variable("W_projection", shape=[num_filters_total, num_classes],
                                       initializer=initializer)  # [embed_size,label_size]
        b_projection = tf.get_variable("b_projection", shape=[num_classes])  # [label_size]
        logits = tf.matmul(h_drop, W_projection) + b_projection  # shape:[None, num_classes]
    return logits


def textRNN(input_x, dropout_keep_prob, vocab_size, embed_size, num_classes, hidden_size,
            initializer=tf.random_normal_initializer(stddev=0.1)):
    Embedding = tf.get_variable("Embedding", shape=[vocab_size, embed_size],
                                initializer=initializer)  # [vocab_size,embed_size]
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x)  # shape:[None,sentence_length,embed_size]
    lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # forward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # backward direction cell
    if dropout_keep_prob is not None:
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words,
                                                 dtype=tf.float32)  # [batch_size,sequence_length,hidden_size]
    print("outputs:===>", outputs)
    output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
    output_rnn_last = tf.reduce_mean(output_rnn, axis=1)  # [batch_size,hidden_size*2]
    print("output_rnn_last:", output_rnn_last)
    with tf.name_scope("output"):
        W_projection = tf.get_variable("W_projection", shape=[hidden_size * 2, num_classes],
                                       initializer=initializer)  # [embed_size,label_size]
        b_projection = tf.get_variable("b_projection", shape=[num_classes])  # [label_size]
        logits = tf.matmul(output_rnn_last, W_projection) + b_projection  # shape:[None, num_classes]
    return logits


def textRCNN(input_x, dropout_keep_prob, batch_size, sequence_length, vocab_size, embed_size, num_classes, hidden_size,
             initializer=tf.random_normal_initializer(stddev=0.1)):
    # tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
    Embedding = tf.get_variable("Embedding", shape=[vocab_size, embed_size],
                                initializer=initializer)  # [vocab_size,embed_size]
    left_side_first_word = tf.get_variable("left_side_first_word", shape=[batch_size, embed_size],
                                           initializer=initializer)
    right_side_last_word = tf.get_variable("right_side_last_word", shape=[batch_size, embed_size],
                                           initializer=initializer)

    W_l = tf.get_variable("W_l", shape=[embed_size, embed_size], initializer=initializer)
    W_r = tf.get_variable("W_r", shape=[embed_size, embed_size], initializer=initializer)
    W_sl = tf.get_variable("W_sl", shape=[embed_size, embed_size], initializer=initializer)
    W_sr = tf.get_variable("W_sr", shape=[embed_size, embed_size], initializer=initializer)
    embedded_words = tf.nn.embedding_lookup(Embedding, input_x)  # [None,sentence_length,embed_size]

    embedded_words_split = tf.split(embedded_words, sequence_length,
                                    axis=1)  # sentence_length个[None,1,embed_size]
    embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]  # sentence_length个[None,embed_size]
    embedding_previous = left_side_first_word
    context_left_previous = tf.zeros((batch_size, embed_size))

    context_left_list = []
    for i, current_embedding_word in enumerate(embedded_words_squeezed):  # sentence_length个[None,embed_size]
        left_c = tf.matmul(context_left_previous, W_l)
        left_e = tf.matmul(embedding_previous, W_sl)  # embedding_previous;[batch_size,embed_size]
        left_h = left_c + left_e
        context_left = tf.nn.tanh(left_h)
        context_left_list.append(context_left)  # append result to list
        embedding_previous = current_embedding_word  # assign embedding_previous
        context_left_previous = context_left  # assign context_left_previous

    embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
    embedded_words_squeezed2.reverse()
    embedding_afterward = right_side_last_word
    context_right_afterward = tf.zeros((batch_size, embed_size))
    context_right_list = []
    for j, current_embedding_word in enumerate(embedded_words_squeezed2):
        right_c = tf.matmul(context_right_afterward, W_r)
        right_e = tf.matmul(embedding_afterward, W_sr)
        right_h = right_c + right_e
        context_right = tf.nn.tanh(right_h)
        context_right_list.append(context_right)
        embedding_afterward = current_embedding_word
        context_right_afterward = context_right
    # 4.ensemble left,embedding,right to output
    output_list = []
    for index, current_embedding_word in enumerate(embedded_words_squeezed):
        representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                   axis=1)
        # print(i,"representation:",representation)
        output_list.append(representation)  # shape:sentence_length个[None,embed_size*3]
    # 5. stack list to a tensor
    # print("output_list:",output_list) #(3, 5, 8, 100)
    output_conv = tf.stack(output_list, axis=1)  # shape:[None,sentence_length,embed_size*3]

    output_pooling = tf.reduce_max(output_conv, axis=1)  # shape:[None,embed_size*3]
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(output_pooling, keep_prob=dropout_keep_prob)  # [None,num_filters_total]

    with tf.name_scope("output"):
        W_projection = tf.get_variable("W_projection", shape=[hidden_size * 3, num_classes],
                                       initializer=initializer)  # [embed_size * 3,label_size]
        b_projection = tf.get_variable("b_projection", shape=[num_classes])  # [label_size]
        logits = tf.matmul(h_drop, W_projection) + b_projection  # [batch_size,num_classes]
    return logits
