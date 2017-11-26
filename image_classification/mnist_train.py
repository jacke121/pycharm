# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Here: ChannelLast
# show channelfirst
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

import math

# print("Tensorflow version " + tf.__version__)
logging.set_verbosity(logging.INFO)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=False, reshape=True, validation_size=0)


def train_data_input_fn():
    return tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],
                                  batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)


# Eval data is an in-memory constant here.
def eval_data_input_fn():
    return tf.constant(mnist.test.images), tf.constant(mnist.test.labels)


# Model loss (not needed in INFER mode)
def conv_model_loss(Ylogits, Y_, mode):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_, 10), Ylogits)) * 100 \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model optimiser (only needed in TRAIN mode)
def conv_model_train_op(loss, mode):
    # Compatibility warning: optimize_loss is still in contrib. This will change in Tensorflow 1.2
    return tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=0.003, optimizer="Adam",
                                           # to remove learning rate decay, comment the next line
                                           learning_rate_decay_fn=lambda lr, step: 0.0001 + tf.train.exponential_decay(
                                               lr, step, -2000, math.e)
                                           ) if mode == tf.estimator.ModeKeys.TRAIN else None


# Model evaluation metric (not needed in INFER mode)
def conv_model_eval_metrics(classes, Y_, mode):
    # You can name the fields of your metrics dictionary as you like.
    return {'accuracy': tf.metrics.accuracy(classes, Y_)} \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model
def conv_model(features, labels, mode):
    X = features
    Y_ = labels
    XX = tf.reshape(X, [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    Y1 = tf.layers.conv2d(XX, filters=6, kernel_size=[6, 6], padding="same", activation=tf.nn.relu,
                          bias_initializer=biasInit)
    Y2 = tf.layers.conv2d(Y1, filters=12, kernel_size=[5, 5], padding="same", strides=2, activation=tf.nn.relu,
                          bias_initializer=biasInit)
    Y3 = tf.layers.conv2d(Y2, filters=24, kernel_size=[4, 4], padding="same", strides=2, activation=tf.nn.relu,
                          bias_initializer=biasInit)
    Y4 = tf.reshape(Y3, [-1, 24 * 7 * 7])
    Y5 = tf.layers.dense(Y4, 200, activation=tf.nn.relu, bias_initializer=biasInit)
    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    Y5d = tf.layers.dropout(Y5, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    Ylogits = tf.layers.dense(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics
    )


estimator = tf.estimator.Estimator(model_fn=conv_model, model_dir="checkpoints")
estimator.train(input_fn=train_data_input_fn, steps=10000)
estimator.evaluate(input_fn=eval_data_input_fn, steps=1)
