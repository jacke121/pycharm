#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-18
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

model = VGG16(weights='imagenet', include_top=True)

img_path = '/home/andrew/PycharmProjects/algorithm-project/datasets/image/VOC2012/JPEGImages/2007_000033.jpg'
import cv2
from PIL import Image
import matplotlib.pyplot as plt
cv_img = cv2.imread(img_path)
img = plt.imread(img_path)
plt.figure()
plt.imshow(cv_img[:,:,::-1])
plt.figure()
plt.imshow(img)
plt.show()
print(cv_img[:,:,::-1]-img)
# # print(img)
exit()

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
org_x = np.copy(x)

x = preprocess_input(x)

features = model.predict(x)
print(features.shape)
print('Predicted:', decode_predictions(features, top=5)[0])

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg

input_image = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
# 创建模型，使用默认的arg scope参数
# arg_scope是slim library的一个常用参数
# 可以设置它指定网络层的参数，比如stride, padding 等等。
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(input_image,
                           # logits, _ = resnet_v1.resnet_v1_50(input_image,
                           num_classes=1000,
                           is_training=False)

# 创建一个函数，从checkpoint读入网络权值
checkpoints_dir = "/home/andrew/PycharmProjects/algorithm-project/slim_models"
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
    slim.get_model_variables('vgg_16'))

# 我们在输出层使用softmax函数，使输出项是概率值
probabilities = tf.nn.softmax(logits)
with tf.Session() as sess:
    # 加载权值
    init_fn(sess)
    org_x[0, :, :, 0] = org_x[0, :, :, 0] - 123.680
    org_x[0, :, :, 1] = org_x[0, :, :, 1] - 116.779
    org_x[0, :, :, 2] = org_x[0, :, :, 2] - 103.939

    print(org_x[0, 1, 1, 0])
    print(x[0, 1, 1, 0])

    probabilities = sess.run(probabilities, feed_dict={input_image: org_x})
    print(probabilities.shape)
    print('Predicted:', decode_predictions(probabilities, top=5)[0])
