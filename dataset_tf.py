#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-19
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

# 支持from_generator textLine tfrecord
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def gen():
    for i in range(10):
        yield i


# dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = tf.data.Dataset.from_generator(gen, tf.int32)
dataset = dataset.repeat().shuffle(len(data))
dataset = dataset.map(lambda x: x + 1)
# dataset = tf.data.Dataset()

dataset = dataset.batch(32).prefetch(len(data) * 5)
# 不需要初始化
iterator = dataset.make_one_shot_iterator()
# 初始初始化,可以传入placeholder
# iterator = dataset.make_initializable_iterator()
# sess.run(iterator.iterator.initializer)
value = iterator.get_next()

sess = tf.Session()
for i in range(40):
    print(sess.run(value))
