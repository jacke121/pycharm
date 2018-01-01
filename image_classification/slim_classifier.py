#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-16
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import urllib2

import pickle
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.contrib.slim.nets import vgg, resnet_v1
from keras.preprocessing import image as image_pil
from PIL import Image as pil_image

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
checkpoints_dir = "/home/andrew/PycharmProjects/algorithm-project/slim_models"
print(root)

image_path = "/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC2012/JPEGImages/2007_000033.jpg"
# img = image_pil.load_img(image_path, target_size=(224, 224))
img = pil_image.open(image_path)
img = img.resize((224, 224), pil_image.BILINEAR)
img.show()
image = np.asarray(img,np.float32)
# image = image_pil.img_to_array(img)
# _R_MEAN = 123.68
# _G_MEAN = 116.78
# _B_MEAN = 103.94
# for vgg and resnet
image[:, :, 0] = image[:, :, 0] - 123.680
image[:, :, 1] = image[:, :, 1] - 116.779
image[:, :, 2] = image[:, :, 2] - 103.939

image = np.expand_dims(image, axis=0)

print(image.shape)

# pil_image.fromarray(x.astype('uint8'), 'RGB')

input_image = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
# 创建模型，使用默认的arg scope参数
# arg_scope是slim library的一个常用参数
# 可以设置它指定网络层的参数，比如stride, padding 等等。
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(input_image,
                           # logits, _ = resnet_v1.resnet_v1_50(input_image,
                           num_classes=1000,
                           is_training=False)

# 我们在输出层使用softmax函数，使输出项是概率值
probabilities = tf.nn.softmax(logits)

# 创建一个函数，从checkpoint读入网络权值
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
    slim.get_model_variables('vgg_16'))

with tf.Session() as sess:
    # 加载权值
    init_fn(sess)

    probabilities = sess.run(probabilities, feed_dict={input_image: image})
    print(probabilities.shape)
    print('Predicted:', decode_predictions(probabilities, top=5)[0])

