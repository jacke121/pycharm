#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-21
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

from dataprovider.coco import CocoDataset
from dataprovider.colormap import colorize
from dataprovider.dataloader import SegDataIterator
from dataprovider.voc import VOCDataset
from image_segmentation.slim_fcn import utils
from image_segmentation.slim_fcn import models
from image_segmentation.slim_fcn.voc_generator import PascalVocGenerator, ImageSetLoader


def deprocessing(x):
    return (x + 1) / 2.


dataset_dir = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/coco"
subset = "train"
year = '2017'
dataset_train = CocoDataset()
dataset_train.load_coco(dataset_dir, subset, year=year)

# subset = "val"
# dataset_val = CocoDataset()
# dataset_val.load_coco(dataset_dir, subset, year=year)

# dataset_dir = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC"
# subset = "trainval"
# year = "2012"
# dataset_train = VOCDataset(dataset_dir, year=year, split=subset)
#
# subset = "val"
# dataset_val = VOCDataset(dataset_dir, year=year, split=subset)

target_size = (224, 224)
batch_size = 32
nb_per_batches = 100
# nb_per_batches = len(dataset_train) // batch_size
# nb_per_val_batches = len(dataset_val) // batch_size
num_classes = len(dataset_train.get_classes())
print('total length: ', len(dataset_train))
print('total classes: ', num_classes)

nb_epochs = 1000
save_freqs = 5
is_training = True
dropout_keep_prob = 0.7
output_dir = "vgg_16_coco"
checkpoint_dir = output_dir + "/checkpoints"

segData_train = SegDataIterator(dataset_train, image_size=target_size, batch_size=batch_size, shuffle=True,
                                max_queue_size=4, deprocess_X=lambda x: 2 * x / 255. - 1)

# segData_val = SegDataIterator(dataset_val, image_size=target_size, batch_size=batch_size, shuffle=False,
#                               max_queue_size=2, deprocess_X=lambda x: 2 * x / 255. - 1)

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(None, 224, 224, num_classes))

preds = models.fcn_model_vgg(inputs, num_classes=num_classes, is_training=is_training,
                             dropout_keep_prob=dropout_keep_prob, reuse=None)

if not is_training:
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("not find model")
    x, y = segData_val.data_gen()
    preds_ = sess.run(preds, feed_dict={inputs: x})
    x = deprocessing(x)
    filename = output_dir + "/validation"
    if not os.path.exists(filename):
        os.mkdir(filename)

    for i in range(10):
        plt.imsave(filename + "/" + str(i) + "-x.jpg", x[i])
        plt.imsave(filename + "/" + str(i) + "-y.jpg", utils.colorize(np.argmax(y[i], axis=-1)) / 255)

        plt.imsave(filename + "/" + str(i) + "-p.jpg", utils.colorize(np.argmax(preds_[i], axis=-1)) / 255)

    exit()


def fcn_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                                       head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def fcn_acc(preds, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=-1),
                                           tf.argmax(labels, axis=-1)), tf.float32))


loss = fcn_loss(preds, labels, num_classes=21)
acc = fcn_acc(preds, labels)

train_op = tf.train.AdamOptimizer().minimize(loss)

init_op = tf.global_variables_initializer()
loss_sm = tf.summary.scalar("loss", loss)
acc_sm = tf.summary.scalar("acc", acc)
summary_op = tf.summary.merge([loss_sm, acc_sm])

# input_image = tf.py_func(deprocessing, [tf.identity(inputs)], tf.float32)
# pred_image = tf.py_func(colorize, [preds], tf.float32)
# real_image = tf.py_func(colorize, [tf.identity(labels)], tf.float32)
# val_input_image_sm = tf.summary.image("real_image", input_image, 6)
# val_pred_image_sm = tf.summary.image("pred_image", pred_image, 6)
# val_real_image_sm = tf.summary.image("real_image", real_image, 6)

val_loss_sm = tf.summary.scalar("val_loss", loss)
val_acc_sm = tf.summary.scalar("val_acc", acc)
val_summary_op = tf.summary.merge([val_loss_sm, val_acc_sm])
# val_summary_op = tf.summary.merge([val_loss_sm, val_acc_sm, val_input_image_sm, val_real_image_sm, val_pred_image_sm])

sess = tf.Session()
sess.run(init_op)

# x, y = segData_train.data_gen()
# input_image = sess.run(real_image, feed_dict={inputs: x,labels:y})
# import matplotlib.pyplot as plt
# print(x.shape)
# print(np.max(x))
# print(np.min(x))
# print(input_image.shape)
# print(np.max(input_image))
# print(np.min(input_image))
# for i in range(5):
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow((x[i] + 1) / 2)
#     plt.subplot(1, 2, 2)
#     plt.imshow(input_image[i])
# plt.show()
#
# exit()

summary_writer = tf.summary.FileWriter(output_dir + '/logs', sess.graph)
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Load weights!")

for epoch in range(1, nb_epochs + 1):
    avg_loss = 0.
    avg_acc = 0.
    for step in tqdm(range(nb_per_batches)):
        x, y = next(segData_train)
        # x, y = segData_train.data_gen()
        preds_, loss_, acc_, _, summary_ = sess.run([preds, loss, acc, train_op, summary_op],
                                                    feed_dict={inputs: x, labels: y})
        summary_writer.add_summary(summary_, global_step=(epoch - 1) * nb_per_batches + step + 1)

        avg_loss += loss_
        avg_acc += acc_
    avg_loss /= nb_per_batches
    avg_acc /= nb_per_batches

    if epoch % save_freqs != 0:
        print("epoch: {0}-{1} loss:{2:.2f} acc:{3:.2f}".format(epoch, nb_epochs, avg_loss, avg_acc))

    else:
        # avg_val_loss = 0.
        # avg_val_acc = 0.
        # for step in range(nb_per_val_batches):
        #     # val_x, val_y = next(segData_val)
        #     val_x, val_y = segData_val.data_gen()
        #     val_preds_, val_loss_, val_acc_, val_summary_ = sess.run([preds, loss, acc, val_summary_op],
        #                                                              feed_dict={inputs: val_x, labels: val_y})
        #     summary_writer.add_summary(val_summary_, global_step=(epoch - 1) * nb_per_val_batches + step + 1)
        #     avg_val_loss += val_loss_
        #     avg_val_acc += val_acc_
        # avg_val_loss /= nb_per_val_batches
        # avg_val_acc /= nb_per_val_batches
        #
        # last_x = deprocessing(val_x)
        # last_y = val_y
        # last_p = val_preds_
        # print("epoch: {0}-{1} loss:{2:.2f} acc:{3:.2f} val_loss:{4:.2f} val_acc:{5:.2f}".
        #       format(epoch, nb_epochs, avg_loss, avg_acc, avg_val_loss, avg_val_acc))
        last_x = deprocessing(x)
        last_y = y
        last_p = preds_
        print("epoch: {0}-{1} loss:{2:.2f} acc:{3:.2f}".format(epoch, nb_epochs, avg_loss, avg_acc))

        saver.save(sess, checkpoint_dir + '/model-{0:.2f}.ckpt'.format(avg_loss),
                   global_step=epoch)

        filename = output_dir + "/image {0}".format(epoch)
        if not os.path.exists(filename):
            os.mkdir(filename)

        for i in range(min(10, len(last_x))):
            plt.imsave(filename + "/" + str(i) + "-x.jpg", last_x[i])
            plt.imsave(filename + "/" + str(i) + "-y.jpg", utils.colorize(np.argmax(last_y[i], axis=-1)) / 255)
            plt.imsave(filename + "/" + str(i) + "-p.jpg", utils.colorize(np.argmax(last_p[i], axis=-1)) / 255)
