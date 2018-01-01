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
from image_segmentation.slim_fcn import utils
from image_segmentation.slim_fcn import models
from image_segmentation.slim_fcn.voc_generator import PascalVocGenerator, ImageSetLoader


def deprocessing(x):
    return (x + 1) / 2.


datagen = PascalVocGenerator(image_shape=[224, 224, 3],
                             image_resample=True,
                             pixelwise_center=False, preprocessing_function=lambda x: 2 * x / 255. - 1
                             )

base_root = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC2012/"
image_set = base_root + "ImageSets/Segmentation/trainval.txt"
image_dir = base_root + "JPEGImages"
label_dir = base_root + "SegmentationClass"
target_size = (224, 224)
num_classes = 21

train_loader = ImageSetLoader(image_set, image_dir=image_dir, label_dir=label_dir, target_size=(224, 224))

batch_size = 32
nb_epochs = 1000
save_freqs = 2
is_training = True
dropout_keep_prob = 0.7
output_dir = "vgg_16_fn"
checkpoint_dir = output_dir + "/checkpoints"
nb_per_batches = len(train_loader.filenames) // batch_size

# ori_img = train_loader.load_seg(train_loader.filenames[0])
# img = utils.colorize(ori_img,num_classes)
# plt.imshow(img)
# plt.imsave("abc.jpg",img/255)
# plt.show()
# print(img.shape)
# print(np.min(img))
# print(np.max(img))
# exit()
# plt.figure()
# plt.imshow(ori_img/255)
datagen = datagen.flow_from_imageset(
    class_mode='categorical',
    classes=21,
    batch_size=32,
    shuffle=True,
    image_set_loader=train_loader)

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(None, 224, 224, 21))

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
    x, y = next(datagen)
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
tf.summary.scalar("loss", loss)
tf.summary.scalar("acc", acc)
summary_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(init_op)

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
        x, y = next(datagen)
        preds_, loss_, acc_, _, summary_ = sess.run([preds, loss, acc, train_op, summary_op],
                                                    feed_dict={inputs: x, labels: y})
        summary_writer.add_summary(summary_, global_step=(epoch - 1) * nb_per_batches + step + 1)
        avg_loss += loss_
        avg_acc += acc_
    avg_loss /= nb_per_batches
    avg_acc /= nb_per_batches

    last_x = deprocessing(x)
    last_y = y
    last_p = preds_

    print("epoch: {0}-{1} loss:{2:.2f} acc:{3:.2f}".format(epoch, nb_epochs, avg_loss, avg_acc))
    if epoch % save_freqs == 0:
        saver.save(sess, checkpoint_dir + '/model-{0:.2f}.ckpt'.format(avg_loss),
                   global_step=epoch)

        filename = output_dir + "/image {0}".format(epoch)
        if not os.path.exists(filename):
            os.mkdir(filename)
        for i in range(10):
            plt.imsave(filename + "/" + str(i) + "-x.jpg", last_x[i])
            plt.imsave(filename + "/" + str(i) + "-y.jpg", utils.colorize(np.argmax(last_y[i], axis=-1)) / 255)

            plt.imsave(filename + "/" + str(i) + "-p.jpg", utils.colorize(np.argmax(last_p[i], axis=-1)) / 255)
