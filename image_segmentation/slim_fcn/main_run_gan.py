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
val_loader = ImageSetLoader(image_set, image_dir=image_dir, label_dir=label_dir, target_size=(224, 224))

batch_size = 32
nb_epochs = 1000
save_freqs = 2
is_training = True
dropout_keep_prob = 0.7
output_dir = "vgg_gan"
checkpoint_dir = output_dir + "/checkpoints"
nb_per_batches = len(train_loader.filenames) // batch_size
gan_weight = 0.1
l1_weight = 100
EPS = 1e-12

datagen_iter = datagen.flow_from_imageset(
    class_mode='categorical',
    classes=21,
    batch_size=32,
    shuffle=True,
    image_set_loader=train_loader)

datagen_val_iter = datagen.flow_from_imageset(
    class_mode='categorical',
    classes=21,
    batch_size=32,
    shuffle=True,
    image_set_loader=val_loader)

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(None, 224, 224, 21))

preds = models.fcn_model_vgg(inputs, num_classes=num_classes, is_training=is_training,
                             dropout_keep_prob=dropout_keep_prob, scope="generator", reuse=None)
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


predict_fake = models.create_discriminator(preds, inputs, scope="discriminator")
gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
gen_loss_L1 = fcn_loss(preds, labels, num_classes=21)
gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
gen_acc = fcn_acc(preds, labels)

gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
gen_optim = tf.train.AdamOptimizer()
gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
gen_train_op = gen_optim.apply_gradients(gen_grads_and_vars)

true_inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
true_labels = tf.placeholder(tf.float32, shape=(None, 224, 224, 21))

predict_real = models.create_discriminator(true_labels, true_inputs, scope="discriminator", reuse=True)
discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
discrim_loss /= 2.0

discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
discrim_optim = tf.train.AdamOptimizer()
discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
discrim_train_op = discrim_optim.apply_gradients(discrim_grads_and_vars)

gen_loss_GAN_sm = tf.summary.scalar("gen_loss_GAN", gen_loss_GAN)
gen_loss_L1_sm = tf.summary.scalar("gen_loss_L1", gen_loss_L1)
gen_loss_sm = tf.summary.scalar("gen_loss", gen_loss)
gen_acc_sm = tf.summary.scalar("gen_acc", gen_acc)
gen_total_sm = tf.summary.merge([gen_loss_GAN_sm, gen_loss_L1_sm, gen_loss_sm, gen_acc_sm])

discrim_loss_sm = tf.summary.scalar("discrim_loss", discrim_loss)
discrim_total_sm = tf.summary.merge([discrim_loss_sm])

init_op = tf.global_variables_initializer()

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
    avg_dis_loss = 0.
    for step in tqdm(range(nb_per_batches)):
        # train
        x, y = next(datagen_iter)
        preds_, gen_loss_GAN_, gen_loss_L1_, gen_loss_, gen_acc_, _, gen_total_sm_ = \
            sess.run([preds, gen_loss_GAN, gen_loss_L1,
                      gen_loss, gen_acc, gen_train_op, gen_total_sm],
                     feed_dict={inputs: x, labels: y})
        summary_writer.add_summary(gen_total_sm_, global_step=(epoch - 1) * nb_per_batches + step + 1)

        # gan
        true_x, true_y = next(datagen_val_iter)
        dis_loss_, _, discrim_total_sm_ = sess.run([discrim_loss, discrim_train_op, discrim_total_sm],
                                                   feed_dict={inputs: x, labels: y, true_inputs: true_x,
                                                              true_labels: true_y})
        summary_writer.add_summary(discrim_total_sm_, global_step=(epoch - 1) * nb_per_batches + step + 1)

        avg_loss += gen_loss_L1_
        avg_acc += gen_acc_
        avg_dis_loss += dis_loss_

    avg_loss /= nb_per_batches
    avg_acc /= nb_per_batches

    last_x = deprocessing(x)
    last_y = y
    last_p = preds_

    print("epoch: {0}-{1} loss:{2:.2f} acc:{3:.2f} dis_loss:{4:.2f}".format(epoch, nb_epochs, avg_loss, avg_acc,
                                                                            avg_dis_loss))
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
