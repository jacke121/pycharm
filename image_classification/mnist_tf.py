#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-26
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

from keras.datasets import mnist
import keras

batch_size = 32
is_training = 1
num_classes = 10
epochs = 12
model_dir = "data"
tf.logging.set_verbosity(tf.logging.INFO)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def input_fn(is_training, data_x, data_y, batch_size=32, num_epochs=1):
    assert len(data_x) == len(data_y), "image and target must be the same length"
    input_x = tf.data.Dataset.from_tensor_slices(data_x)
    input_y = tf.data.Dataset.from_tensor_slices(data_y)

    data_xy = tf.data.Dataset.zip((input_x, input_y))
    batched_dataset = data_xy.repeat(num_epochs)

    if is_training:
        batched_dataset = batched_dataset.shuffle(len(data_x))

    batched_dataset = batched_dataset.batch(batch_size).prefetch(batch_size * 10)

    batched_iter = batched_dataset.make_one_shot_iterator()
    x, y = batched_iter.get_next()
    return x, y


def mnist_model(inputs, mode):
    """Takes the MNIST inputs and mode and outputs a tensor of logits."""
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    dense1 = tf.layers.dense(inputs=dropout, units=200)
    logits = tf.layers.dense(inputs=dense1, units=10)
    return logits


def mnist_model_fn(features, labels, mode):
    """Model function for MNIST."""
    logits = mnist_model(features, mode)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.summary.scalar("avg_loss", loss)
    tf.summary.scalar("avg_accuracy", accuracy[1])
    tf.summary.image("mnist_image", features, max_outputs=5)

    # Create a tensor named train_accuracy for logging purposes
    # tf.identity(accuracy[1], name='train_accuracy')
    # tf.identity(loss, name='loss')
    # tf.summary.scalar('train_accuracy', accuracy[1])
    # tf.summary.scalar('loss', loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


# x, y = input_fn(is_training, x_train, y_train, batch_size=batch_size, num_epochs=10)
# exit()

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=mnist_model_fn, model_dir=model_dir)

# Set up training hook that logs the training accuracy every 100 steps.
# tensors_to_log = {
#     'train_accuracy': 'train_accuracy',
#     'train_loss': 'loss',
# }
# logging_hook = tf.train.LoggingTensorHook(
#     tensors=tensors_to_log, every_n_iter=1)
# Train the model
# mnist_classifier.train(
#     input_fn=lambda: input_fn(is_training, x_train, y_train, batch_size=batch_size, num_epochs=10),
#     hooks=None, steps=1000)

# Evaluate the model and print results
# eval_results = mnist_classifier.evaluate(
#     input_fn=lambda: input_fn(is_training, x_train, y_train, batch_size=batch_size))
# print()
# print('Evaluation results:\n\t%s' % eval_results)
