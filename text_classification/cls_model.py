#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import sys
import os
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'


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
