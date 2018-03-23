#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import time
import tensorflow as tf

from skip_thought_model import SkipThoughtModel
from prodata.data_utils import TextData

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch Size.')
tf.app.flags.DEFINE_integer('eval_batch_size', 1, 'Eval batch Size.')
tf.app.flags.DEFINE_integer('rnn_size', 200, 'RNN Size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')
tf.app.flags.DEFINE_float('dropout', 0.2, 'Dropout rate (not keep_prob)')
tf.app.flags.DEFINE_integer('epochs', 300, 'Maximum number of epochs in training.')
tf.app.flags.DEFINE_integer('eval_per_epoch', 50, 'Eval per epoch.')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Encoder & Decoder embedding size.')
tf.app.flags.DEFINE_integer('num_keep_ckpts', 5, 'Max number of checkpoints to keep.')
tf.app.flags.DEFINE_integer('target_max_len', 50, 'Target sentence max length.')
tf.app.flags.DEFINE_integer('max_vocab_size', 20000, 'Max vocab size.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'model/', 'Path to model.')
tf.app.flags.DEFINE_string('train_data_path', 'data/aytrain.txt.a.word', 'Path to file with train data.')
tf.app.flags.DEFINE_string('eval_data_path', 'data/aytrain.test', 'Path to file of eval data.')

FLAGS = tf.app.flags.FLAGS


def train_feed_dict(model, batch, tgt_max_len):
    assert model and batch
    train_feed_data = {
        # curr
        model.input_data: batch[0].data,
        model.input_seq_len: batch[0].seq_lengths,
        # prev
        model.prev_target_data_input: batch[1].data,
        model.prev_target_data_output: batch[2].data,
        model.prev_target_mask: batch[2].weights,
        model.prev_target_seq_len:
            [tgt_max_len] * batch[1].seq_lengths.size,
        # next
        model.next_target_data_input: batch[3].data,
        model.next_target_data_output: batch[4].data,
        model.next_target_mask: batch[4].weights,
        model.next_target_seq_len:
            [tgt_max_len] * batch[3].seq_lengths.size
    }
    return train_feed_data


def eval_feed_dict(model, batch):
    assert model and batch
    eval_feed_data = {
        # curr
        model.input_data: batch.data,
        model.input_seq_len: batch.seq_lengths
    }

    return eval_feed_data


def main(_):
    start_time = time.time()

    train_gragh = tf.Graph()

    with train_gragh.as_default():
        print('init data...')
        text_data = TextData(
            FLAGS.train_data_path, max_vocab_size=FLAGS.max_vocab_size, max_len=FLAGS.target_max_len
        )
        if text_data.max_len > FLAGS.target_max_len:
            FLAGS.target_max_len = text_data.max_len
        print('init model...')
        skip_thought_model = SkipThoughtModel(
            len(text_data.vocab), len(text_data.vocab), text_data.vocab.START_VOCAB, FLAGS.target_max_len,
            FLAGS.rnn_size, FLAGS.num_layers, FLAGS.dropout,
            FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.num_keep_ckpts
        )

        with tf.Session() as sess:
            print('init train...')
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(1, FLAGS.epochs + 1):
                # train
                train_data = text_data.pro_triples_data(FLAGS.batch_size)
                for j, train_batch in enumerate(train_data):
                    # print(j, batch)
                    batch_loss, _ = sess.run(
                        [skip_thought_model.cost, skip_thought_model.train_op],
                        feed_dict=train_feed_dict(skip_thought_model, train_batch, FLAGS.target_max_len)
                    )
                    print(i, j, batch_loss)
                # if eval?
                if i % FLAGS.eval_per_epoch == 0:
                    # eval
                    eval_data = text_data.pro_tuple_data(FLAGS.eval_data_path, FLAGS.eval_batch_size)
                    for l, pred_batch in enumerate(eval_data):
                        if pred_batch == text_data.ONE_LINE_TOKEN:
                            continue
                        prev_predict, next_predict = sess.run(
                            [skip_thought_model.prev_predict_logits, skip_thought_model.next_predict_logits],
                            feed_dict=eval_feed_dict(skip_thought_model, pred_batch)
                        )

                        print(l, '------')
                        for pred_i in prev_predict:
                            for pred_j in pred_i:
                                print(text_data.vocab.index2word[pred_j], end=',')
                            print()

                        for next_i in next_predict:
                            for next_j in next_i:
                                print(text_data.vocab.index2word[next_j], end=',')
                            print()

                    # save session
                    skip_thought_model.saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=i)

    print('Elapse time: ' + str((time.time() - start_time)))


if __name__ == '__main__':
    print('start')
    print(time.strftime("%Y/%m/%d %H:%M:%S"))
    tf.app.run()
    print('ok')
    print(time.strftime("%Y/%m/%d %H:%M:%S"))
