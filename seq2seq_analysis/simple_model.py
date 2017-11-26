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

[X_train, Y_train, X_test, Y_test,
 zh_word2idx, zh_idx2word, zh_vocab, en_word2idx, en_idx2word, en_vocab] = np.load(
    root + 'datasets/text/spoken_data/pack.npz.npy')

batch_size = 32
src_vocab_size = len(zh_word2idx)
source_sequence_length = len(X_train[0])
tgt_vocab_size = len(en_word2idx)
decoder_length = len(Y_train[0])
num_units = 128
src_embed_size = num_units
tgt_embed_size = num_units
learning_rate = 0.001
max_gradient_norm = 100.
dtype = tf.float32
maximum_iterations = int(decoder_length * 1.5)
beam_width = 10
num_layers = 1

tile_Y_train = Y_train[:, 1:decoder_length]
tile_Y_train = np.insert(tile_Y_train, -1, 0, axis=1)


def build_encoder(source):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("embeddings") as scope:
            embedding_encoder = tf.get_variable(
                "encoder_embeddings", [src_vocab_size + 10, src_embed_size], scope.dtype)
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)

        # Run Dynamic RNN
        #   encoder_outpus: [batch_size, src_length,num_units]
        #   encoder_state: [batch_size, num_units]
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, dtype=tf.float32)

        # Construct forward and backward cells
        # forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        # backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        #
        # bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
        #                                                             encoder_emb_inp, dtype=tf.float32)
        # encoder_outputs = tf.concat(bi_outputs, -1)
        # encoder_state = encoder_state[0]

        # multi-layer RNN
        # cell_list = []
        # for i in range(num_layers):
        #     single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        #     cell_list.append(single_cell)
        #
        # encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, dtype=tf.float32)
        # encoder_state = encoder_state[-1]

        # multi-layer bi-RNN
        # cell_list = []
        # for i in range(num_layers):
        #     single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        #     cell_list.append(single_cell)
        #
        # fw_encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        #
        # cell_list = []
        # for i in range(num_layers):
        #     single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        #     cell_list.append(single_cell)
        #
        # bk_encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        #
        # bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_encoder_cell, bk_encoder_cell,
        #                                                                encoder_emb_inp, dtype=tf.float32)
        #
        # encoder_outputs = tf.concat(bi_outputs, -1)
        # encoder_state = bi_encoder_state[-1]
        #

        print(tf.shape(encoder_outputs))
        print(tf.shape(encoder_state))

        return encoder_outputs, encoder_state


def build_decoder(target_input, encoder_state, batch_size=32):
    with tf.variable_scope("decoder"):
        embedding_decoder = tf.get_variable(
            "decoder_embedding", [tgt_vocab_size, tgt_embed_size], dtype)
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_input)
        # basic decoder
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        # bi decoder
        # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units * 2)
        # multi-layer RNN
        # cell_list = []
        # for i in range(num_layers):
        #     single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        #     cell_list.append(single_cell)
        #
        # decoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

        # Helper
        decoder_lengths = [decoder_length] * batch_size
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
        # Decoder
        projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
                                                  output_layer=projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

    with tf.variable_scope("inference"):
        # # Helper
        # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     embedding_decoder,
        #     tf.fill([batch_size], 2), 3)
        #
        # # Decoder
        # infer_decoder = tf.contrib.seq2seq.BasicDecoder(
        #     decoder_cell, helper, encoder_state,
        #     output_layer=projection_layer)
        # # Dynamic decoding
        # infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        #     infer_decoder, maximum_iterations=maximum_iterations)
        # translations = outputs.predicted_ids
        # translations = infer_outputs.sample_id


        #  beam search
        start_tokens = tf.fill([batch_size], 2)

        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)

        # Define a beam-search decoder
        infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_decoder,
            start_tokens=start_tokens,
            end_token=3,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=projection_layer,
            length_penalty_weight=0.0)

        infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            infer_decoder, maximum_iterations=maximum_iterations)
        translations = infer_outputs.predicted_ids

    return logits, translations


if __name__ == '__main__':

    sess = tf.Session()

    input_x = tf.data.Dataset.from_tensor_slices(X_train)
    input_y = tf.data.Dataset.from_tensor_slices(Y_train)
    input_tile_y = tf.data.Dataset.from_tensor_slices(tile_Y_train)

    data_xy = tf.data.Dataset.zip((input_x, input_y, input_tile_y))

    batched_dataset = data_xy.repeat().shuffle(len(X_train)).batch(batch_size)
    # abc = tf.data.Dataset()
    batched_iter = batched_dataset.make_one_shot_iterator()
    x, y, tile_y = batched_iter.get_next()
    encoder_outputs, encoder_state = build_encoder(x)
    logits, translations = build_decoder(y, encoder_state)

    preds = tf.argmax(logits, axis=-1)
    # 需要tile_y
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tile_y, logits=logits)
    train_loss = tf.reduce_mean(tf.reduce_sum(crossent, axis=1))

    # Calculate and clip gradients
    params = tf.trainable_variables()
    # gradients = tf.gradients(train_loss, params)
    # clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    # Optimization
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(train_loss, var_list=tf.trainable_variables())
    # train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        print(str(i) + "*************")
        np_x, np_y, np_pred, np_tile_y, np_loss, _ = sess.run(
            [x, y, preds, tile_y, train_loss, train_op])  # (1, array([1]))
        # for j in range(1):
        #     print('--------------')
        #     print("".join([zh_idx2word[index] for index in np_x[j]]).replace("<pad>", "").strip())
        #     print(" ".join([en_idx2word[index] for index in np_y[j]]).replace("<pad>", "").strip())
        #     print(" ".join([en_idx2word[index] for index in np_tile_y[j]]).replace("<pad>", "").strip())
        #     print(" ".join([en_idx2word[index] for index in np_pred[j]]).replace("<pad>", "").strip())
        print(np_loss)
        np_x, np_y, np_t = sess.run([x, y, translations])
        for j in range(3):
            print('--------------')
            print("".join([zh_idx2word[index] for index in np_x[j]]).replace("<pad>", "").strip())
            print(" ".join([en_idx2word[index] for index in np_y[j]]).replace("<pad>", "").strip())
            # print(" ".join([en_idx2word[index] for index in np_t[j]]).replace("<pad>", "").strip())
            np_t_width = np.transpose(np_t[j])
            print(np_t_width.shape)
            for k in range(len(np_t_width)):
                print(
                    " ".join([en_idx2word.get(index, "<nn>") for index in np_t_width[k]]).replace("<pad>", "").strip())
            print('--------------')
