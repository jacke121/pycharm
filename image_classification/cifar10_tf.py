# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
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
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import argparse
import tensorflow as tf

from image_classification import cifar_process

reload(sys)
sys.setdefaultencoding('utf-8')
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--model_dir', type=str, default='cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='The number of images per batch.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

(x_train, y_train), (x_test, y_test) = cifar_process.load_data()


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def input_fn(is_training, data_x, data_y, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
    Returns:
      A tuple of images and labels.
    """
    assert len(data_x) == len(data_y), "image and target must be the same length"
    input_x = tf.data.Dataset.from_tensor_slices(data_x)
    input_y = tf.data.Dataset.from_tensor_slices(data_y)
    dataset = tf.data.Dataset.zip((input_x, input_y))

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(
        lambda image, label: (preprocess_image(image, is_training), tf.one_hot(tf.squeeze(label), _NUM_CLASSES)))

    dataset = dataset.prefetch(2 * batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def cifar10_model(inputs, mode):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    bn2 = tf.layers.batch_normalization(
        inputs=pool1, training=(mode == tf.estimator.ModeKeys.TRAIN), fused=True)
    bn2 = tf.nn.relu(bn2)

    conv2 = tf.layers.conv2d(
        inputs=bn2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    dense1 = tf.layers.dense(inputs=dropout, units=200, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense1, units=10)
    return logits


def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    tf.summary.image('images', features, max_outputs=6)

    logits = cifar10_model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.
        initial_learning_rate = 0.1 * params['batch_size'] / 128
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'batch_size': FLAGS.batch_size,
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
                True, x_train, y_train, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=lambda: input_fn(False, x_test, y_test, FLAGS.batch_size))
        print()
        print('Evaluation results:\n\t%s' % eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
