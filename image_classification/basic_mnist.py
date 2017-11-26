#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-12
import sys
import os
import tensorflow as tf
import numpy as np
import math
from progressbar import ProgressBar
from tensorflow.examples.tutorials.mnist import input_data

from basic_utils import weight_variable, bias_variable, max_pool_2x2, conv2d, batchnorm

reload(sys)
sys.setdefaultencoding('utf8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

# load mnist data
mnist = input_data.read_data_sets("data/", one_hot=True, validation_size=0)
# batch_x, batch_y = mnist.train.next_batch(64)
batch_size = 32
nb_epochs = 3
model_path = "data/model.ckpt"

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# ********************** conv1 *********************************
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

# ********************** conv2 *********************************
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
# Y3bn, update_ema3 = batchnorm(h_conv2, tst, iter, B3, convolutional=True)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

# ********************* func1 layer *********************************
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ********************* func2 layer *********************************
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
# use Gradientdescentoptimizer

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# calculate the accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# init session
sess = tf.Session()
# saver.restore(sess,model_path)
# preds = sess.run(prediction, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
# print np.argmax(preds, axis=1), np.argmax(mnist.test.labels, axis=1)
# exit()

sess.run(tf.global_variables_initializer())

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0

# Training cycle
for epoch in range(nb_epochs):
    avg_cost = 0.
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed)
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    progress = ProgressBar()
    for i in progress(range(total_batch)):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, lr: learning_rate})
        cost = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        avg_cost += cost / total_batch
    # Compute average loss
    # Display logs per epoch step
    if epoch % 1 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

preds = sess.run(prediction, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print np.argmax(preds, axis=1), np.argmax(mnist.test.labels, axis=1)
# Save model weights to disk
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)
