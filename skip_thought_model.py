#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import tensorflow as tf
from tensorflow.python.layers.core import Dense


class SkipThoughtModel(object):
    """
    Model skip-thought
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, start_vocab, max_target_len,
                 rnn_size, num_layers, dropout, embedding_size, learning_rate, num_keep_ckpts):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.start_vocab = start_vocab  # start_vocab = ['<pad>', '<go>', '<eos>', '<unk>']
        self.max_target_len = max_target_len

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

        self.curr_encoder_output = None
        self.curr_encoder_state = None
        self.prev_train_logits = None
        self.prev_predict_logits = None
        self.next_train_logits = None
        self.next_predict_logits = None
        self.cost = None
        self.train_op = None

        self.encoder_output = None
        self.encoder_state = None
        self.prev_train_decoder_output = None
        self.prev_predict_decoder_output = None
        self.next_train_decoder_output = None
        self.next_predict_decoder_output = None

        self._logger = logging.getLogger(__name__)
        self._build_placeholder()
        self._logger.info('Build placeholder.')
        self._build_model()
        self._logger.info('Build model.')
        self._build_train()
        self._logger.info('Build train.')
        self._build_predict()
        self._logger.info('Build predict.')

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=num_keep_ckpts)

    def _build_placeholder(self):
        with tf.variable_scope('placeholders'):
            # curr input
            self.input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.input_seq_len = tf.placeholder(tf.int32, [None], name='inputs_seq_len')
            self.batch_size = tf.size(self.input_seq_len)
            # prev target
            self.prev_target_data_input = tf.placeholder(tf.int32, [None, None], name='prev_targets_input')
            self.prev_target_data_output = tf.placeholder(tf.int32, [None, None], name='prev_targets_output')
            self.prev_target_mask = tf.placeholder(tf.float32, [None, None], name='prev_targets_mask')
            self.prev_target_seq_len = tf.placeholder(tf.int32, [None], name='prev_targets_seq_len')
            # next target
            self.next_target_data_input = tf.placeholder(tf.int32, [None, None], name='next_targets_input')
            self.next_target_data_output = tf.placeholder(tf.int32, [None, None], name='next_targets_output')
            self.next_target_mask = tf.placeholder(tf.float32, [None, None], name='next_targets_mask')
            self.next_target_seq_len = tf.placeholder(tf.int32, [None], name='next_targets_seq_len')

    @staticmethod
    def get_lstm_cell(rnn_size, dropout):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
        #                                           input_keep_prob=(1.0 - dropout), output_keep_prob=1.0)
        return lstm_cell

    @staticmethod
    def get_gru_cell(rnn_size, dropout):
        gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
        # gru_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_cell,
        #                                          input_keep_prob=(1.0 - dropout), output_keep_prob=1.0)
        return gru_cell

    def _build_encoder(self, enc_scope_name):
        with tf.name_scope(enc_scope_name):
            # Tensor, [batch_size, max_time, embed_size]
            encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_data, self.src_vocab_size,
                                                                   self.embedding_size)
            cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.rnn_size, self.dropout)
                                                for _ in range(self.num_layers)])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                              sequence_length=self.input_seq_len, dtype=tf.float32)
        return encoder_output, encoder_state

    def _build_decoder(self, dec_scope_name, encoder_output, encoder_state, target_data, target_seq_len):
        with tf.name_scope(dec_scope_name):
            decoder_embeddings = tf.Variable(tf.random_uniform([self.tgt_vocab_size, self.embedding_size]))
            # cell
            cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.rnn_size, self.dropout)
                                                for _ in range(self.num_layers)])
            # attention-model
            cell, decoder_initial_state = self._build_attention(encoder_output, encoder_state, cell)
            # output_layer
            output_layer = Dense(self.tgt_vocab_size, use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            with tf.variable_scope(dec_scope_name + '_train'):
                # Data format of target_data: <GO>...<PAD>
                # Tensor: [batch_size, max_time, embed_size], type: float32.
                decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, target_data)
                train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                 sequence_length=target_seq_len, time_major=False)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper,
                                                                decoder_initial_state, output_layer)
                train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True,
                                                                               maximum_iterations=self.max_target_len)

            with tf.variable_scope(dec_scope_name + '_predict', reuse=True):
                # start_tokens = tf.tile(tf.constant([self.start_vocab.index('<go>')], dtype=tf.int32),
                #                        [self.batch_size], name='start_tokens')
                predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    decoder_embeddings,
                    tf.fill([self.batch_size], self.start_vocab.index('<go>')),
                    self.start_vocab.index('<eos>'))
                predict_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predict_helper,
                                                                  decoder_initial_state, output_layer)
                predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, impute_finished=True,
                                                                                 maximum_iterations=self.max_target_len)

        return train_decoder_output, predict_decoder_output

    def _build_attention(self, encoder_output, encoder_state, cell):
        # attention_states: [batch_size, max_time, num_units]
        # attention_states = tf.transpose(encoder_output, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.rnn_size, encoder_output,
                                                                memory_sequence_length=self.input_seq_len)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                   attention_layer_size=self.rnn_size)
        decoder_initial_state = cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state

    def _build_model(self):
        self.encoder_output, self.encoder_state = self._build_encoder('encoder')

        self.prev_train_decoder_output, self.prev_predict_decoder_output = \
            self._build_decoder('prev_decoder', self.encoder_output, self.encoder_state,
                                self.prev_target_data_input, self.prev_target_seq_len)

        self.next_train_decoder_output, self.next_predict_decoder_output = \
            self._build_decoder('next_decoder', self.encoder_output, self.encoder_state,
                                self.next_target_data_input, self.next_target_seq_len)

    @staticmethod
    def _cmpt_loss(scope_name, train_decoder_output, target_data_output, target_mask):
        with tf.variable_scope(scope_name):
            train_logits = tf.identity(train_decoder_output.rnn_output, name='logits')
            # Data format of target_data_output: ...<EOS><PAD>
            cost = tf.contrib.seq2seq.sequence_loss(train_logits, tf.convert_to_tensor(target_data_output),
                                                    target_mask)
        return cost

    def _build_train(self):
        prev_cost = self._cmpt_loss('prev', self.prev_train_decoder_output, self.prev_target_data_output,
                                    self.prev_target_mask)

        next_cost = self._cmpt_loss('next', self.next_train_decoder_output, self.next_target_data_output,
                                    self.next_target_mask)

        self.cost = prev_cost + next_cost
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def _build_predict(self):
        with tf.variable_scope('prev'):
            self.prev_train_logits = tf.identity(self.prev_train_decoder_output.rnn_output, name='logits')
            self.prev_predict_logits = tf.identity(self.prev_predict_decoder_output.sample_id, name='predictions')

        with tf.variable_scope('next'):
            self.next_train_logits = tf.identity(self.next_train_decoder_output.rnn_output, name='logits')
            self.next_predict_logits = tf.identity(self.next_predict_decoder_output.sample_id, name='predictions')

        with tf.variable_scope('curr'):
            self.curr_encoder_output = tf.identity(self.encoder_output, name='output')
            self.curr_encoder_state = tf.identity(self.encoder_state, name='state')
