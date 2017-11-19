#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import sys
import os
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'


def loss(input_y, logits, l2_lambda=0.0001):  # 0.001
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
        loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
    return loss


def loss_multilabel(input_y, logits, l2_lambda=0.00001):
    with tf.name_scope("loss"):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y, logits=logits)
        print("sigmoid_cross_entropy_with_logits.losses:", losses)
        losses = tf.reduce_sum(losses, axis=1)
        loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
    return loss

