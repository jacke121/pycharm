#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-12
import tensorflow as tf


# pool max_pool avg_pool
# tf.nn.max_pool( value=input, ksize=[1, n_size, n_size, 1],
#                 strides=[1, stride, stride, 1], padding=padding)
# LRN
# tf.nn.local_response_normalization(
#                 pool, depth_radius=7, alpha=0.001, beta=0.75)

# 权重矩阵
# self.weight = tf.Variable(
#     initial_value=tf.truncated_normal(
#         shape=[n_size, n_size, self.input_shape[3], self.n_filter],
#         mean=0.0, stddev=numpy.sqrt(
#             2.0 / (self.input_shape[1] * self.input_shape[2] * self.input_shape[3]))),
#     name='W_%s' % (name))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)
