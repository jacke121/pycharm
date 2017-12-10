#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-11-18
import sys
import os
import tensorflow as tf
import numpy as np
import utils
import cls_model
import random

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

[X_train, Y_train, X_test, Y_test, sent_word2idx, sent_idx2word,
 sent_vocab, tag_word2idx, tag_idx2word, tag_vocab] = np.load(root + 'datasets/text/wangke/pack.npz.npy')

nb_epoches = 1000
batch_size = 32
checkpoint_dir = root + 'datasets/text/wangke/checkpoint/'
checkpoint_steps = 100
result_path = root + 'datasets/text/wangke/result.txt'

nb_samples = len(X_train)
steps = len(X_train) / batch_size
sequence_length = len(X_train[0])
num_classes = len(tag_vocab)
vocab_size = len(sent_vocab)

print("nb_samples: ", nb_samples)
print("num_classes: ", num_classes)

is_training = False
multi_label_flag = False

embed_size = 128
learning_rate = 0.05
decay_steps = 100
decay_rate = 0.96
dropout_keep_prob_value = 0.6

clip_gradients = 5.0
decay_rate_big = 0.50
l2_lambda = 0.001

is_shuffle = True

if is_shuffle:
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

# add placeholder (X,label)
input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X
input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")  # y:[None,num_classes]
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
# global_step = tf.Variable(0, trainable=False, name="Global_Step")

# *********************************************************************************************************
# fastText
# logits = cls_model.fastText(input_x, vocab_size, embed_size, num_classes)
# *********************************************************************************************************
# textCNN
# filter_sizes = [3, 4, 5]
# num_filters = 128
# logits = cls_model.textCNN(input_x, dropout_keep_prob, sequence_length, vocab_size,
#                            embed_size, num_classes, filter_sizes, num_filters)
# *********************************************************************************************************
# textRNN
hidden_size = 128
logits = cls_model.textRNN(input_x, dropout_keep_prob, vocab_size, embed_size, num_classes, hidden_size)
# *********************************************************************************************************
# textRCNN
# hidden_size = 128
# logits = cls_model.textRCNN(input_x, dropout_keep_prob, batch_size, sequence_length, vocab_size, embed_size,
#                             num_classes, hidden_size)
# *********************************************************************************************************


if multi_label_flag:
    print("going to use multi label loss.")
    loss_val = utils.loss_multilabel(tf.cast(input_y, tf.float32), logits, l2_lambda=l2_lambda)
else:
    print("going to use single label loss.")
    loss_val = utils.loss(tf.cast(input_y, tf.float32), logits, l2_lambda=l2_lambda)

predictions = tf.argmax(logits, 1, name="predictions")  # shape:[None,]
labels = tf.argmax(input_y, 1, output_type=tf.int32, name="labels")
correct_prediction = tf.equal(tf.cast(predictions, tf.int32), labels)  # tf.argmax(logits, 1)-->[batch_size]

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_or_create_global_step(), decay_steps, decay_rate,
                                           staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(loss_val, var_list=tf.trainable_variables())

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if is_training:
    for itr in range(nb_epoches):
        avg_loss = 0.
        avg_acc = 0.

        # train
        for step in range(steps):
            feed_input_x = X_train[batch_size * step:batch_size * (step + 1)]
            feed_input_y = Y_train[batch_size * step:batch_size * (step + 1)]

            loss, acc, predict, lab, _ = sess.run([loss_val, accuracy, predictions, labels, train_op],
                                                  feed_dict={input_x: feed_input_x, input_y: feed_input_y,
                                                             dropout_keep_prob: dropout_keep_prob_value})
            avg_loss += loss
            avg_acc += acc
        avg_loss /= steps
        avg_acc /= steps
        if (itr + 1) % checkpoint_steps == 0:
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=itr + 1)
            # test
        loss, acc, predict, lab, lr = sess.run([loss_val, accuracy, predictions, labels, learning_rate],
                                               feed_dict={input_x: X_test, input_y: Y_test,
                                                          dropout_keep_prob: 1.0})

        print("itr: ", itr, "loss:", avg_loss, "acc:", avg_acc, "val_loss: ", loss, "val_acc", acc)
        print("learning_rate: ", lr)
        print("labels: ", lab[:30])
        print("predis: ", predict[:30])

else:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        pass
    loss, acc, predict, lab, = sess.run([loss_val, accuracy, predictions, labels],
                                        feed_dict={input_x: X_test, input_y: Y_test,
                                                   dropout_keep_prob: 1.0})
    print("val_loss: ", loss, "val_acc", acc)
    print("labels: ", lab)
    print("predis: ", predict)
    with open(result_path, 'w') as f_w:
        for sent, tag, pred in zip(X_test, Y_test, predict):
            sent_line = " ".join([sent_idx2word[word] for word in sent])
            tag_t = []
            for i, tta in enumerate(tag):
                if tta:
                    tag_t.append(i)

            tag_line = " ".join([tag_idx2word[tta] for tta in tag_t])
            pred_line = tag_idx2word[pred]
            f_w.write(tag_line + " __pred__ " + pred_line + " __content__ " + sent_line + "\n")
