#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-22
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg


def fcn_model(inputs, num_classes=21, is_training=True, dropout_keep_prob=0.8, reuse=None):
    if not is_training:
        dropout_keep_prob = 1.0

    with tf.variable_scope('vgg_16', reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.selu,
                            weights_initializer=tf.glorot_normal_initializer()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = tf.contrib.nn.alpha_dropout(net, dropout_keep_prob)

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            net = slim.conv2d_transpose(net, 256, kernel_size=(3, 3), stride=(2, 2), scope="deconv1")
            net = tf.contrib.nn.alpha_dropout(net, dropout_keep_prob)
            # net = slim.batch_norm(net, 8, is_training=is_training)
            # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout')

            net = slim.conv2d_transpose(net, 128, kernel_size=(3, 3), stride=(2, 2), scope="deconv2")
            # net = slim.batch_norm(net, 8, is_training=is_training)

            net = slim.conv2d_transpose(net, 64, kernel_size=(3, 3), stride=(4, 4), scope="deconv3")
            net = tf.contrib.nn.alpha_dropout(net, dropout_keep_prob)
            # net = slim.batch_norm(net, 8, is_training=is_training)

            net = slim.conv2d_transpose(net, 32, kernel_size=(3, 3), stride=(2, 2), scope="deconv4")
            # preds = slim.batch_norm(net, 8, is_training=is_training)

            preds = slim.conv2d(net, num_classes, [2, 2], scope="conv6")

            return preds


def fcn_model_vgg(inputs, num_classes=21, is_training=True, dropout_keep_prob=0.8, scope='vgg_16', reuse=None):
    if not is_training:
        dropout_keep_prob = 1.0

    with tf.variable_scope(scope, reuse=reuse):
        net = slim.repeat(
            inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = slim.conv2d_transpose(net, 256, kernel_size=(3, 3), stride=(2, 2), scope="deconv1")
        net = slim.batch_norm(net, is_training=is_training)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')

        net = slim.conv2d_transpose(net, 128, kernel_size=(3, 3), stride=(2, 2), scope="deconv2")
        net = slim.batch_norm(net, is_training=is_training)

        net = slim.conv2d_transpose(net, 64, kernel_size=(3, 3), stride=(4, 4), scope="deconv3")
        net = slim.batch_norm(net, is_training=is_training)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')

        net = slim.conv2d_transpose(net, 32, kernel_size=(3, 3), stride=(2, 2), scope="deconv4")
        preds = slim.conv2d(net, num_classes, [2, 2], activation_fn=None, scope="conv6")

        return preds


def create_discriminator(discrim_inputs, discrim_targets, ndf=32, scope='discriminator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        net = tf.concat([discrim_inputs, discrim_targets], axis=3)

        net = slim.conv2d(net, ndf, [3, 3], stride=(2, 2), activation_fn=tf.nn.leaky_relu, scope='layer1')
        net = slim.conv2d(net, ndf, [3, 3], stride=(2, 2), activation_fn=tf.nn.leaky_relu, scope='layer2')
        net = slim.batch_norm(net)
        net = slim.conv2d(net, ndf, [3, 3], stride=(1, 1), activation_fn=tf.nn.leaky_relu, scope='layer3')
        net = slim.batch_norm(net)
        net = slim.conv2d(net, 1, [3, 3], stride=(2, 2), activation_fn=tf.nn.sigmoid, scope='layer4')

    return net
