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
tf.app.flags.DEFINE_integer('pred_batch_size', 1, 'Predict batch Size.')
tf.app.flags.DEFINE_integer('rnn_size', 200, 'RNN Size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')
tf.app.flags.DEFINE_float('dropout', 0.2, 'Dropout rate (not keep_prob)')
tf.app.flags.DEFINE_integer('epochs', 200, 'Maximum number of epochs in training.')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Encoder & Decoder embedding size.')
tf.app.flags.DEFINE_integer('num_keep_ckpts', 5, 'Max number of checkpoints to keep.')
tf.app.flags.DEFINE_integer('target_max_len', 50, 'Target sentence max length.')
tf.app.flags.DEFINE_integer('max_vocab_size', 20000, 'Max vocab size.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'model/', 'Path to model.')
tf.app.flags.DEFINE_string('train_data_path', 'data/aytrain.txt.a.word', 'Path to file with train data.')
tf.app.flags.DEFINE_string('pred_src_path', 'data/aytrain.txt.a.word', 'Path to file of predict source data.')
tf.app.flags.DEFINE_string('pred_tgt_path', 'data/aytrain.txt.matrix', 'Path to file of predict matrix data.')

FLAGS = tf.app.flags.FLAGS


def pred_feed_dict(model, batch):
    assert model and batch
    pred_feed_data = {
        # curr
        model.input_data: batch.data,
        model.input_seq_len: batch.seq_lengths
    }

    return pred_feed_data


def main(_):
    start_time = time.time()

    train_gragh = tf.Graph()

    with train_gragh.as_default():
        print('init data...')
        text_data = TextData(
            FLAGS.train_data_path, max_vocab_size=FLAGS.max_vocab_size, max_len=FLAGS.target_max_len)
        if text_data.max_len > FLAGS.target_max_len:
            FLAGS.target_max_len = text_data.max_len
        print('init model...')
        skip_thought_model = SkipThoughtModel(
            len(text_data.vocab), len(text_data.vocab), text_data.vocab.START_VOCAB, FLAGS.target_max_len,
            FLAGS.rnn_size, FLAGS.num_layers, FLAGS.dropout,
            FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.num_keep_ckpts)

        with tf.Session() as sess:
            print('init predict...')
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                skip_thought_model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass

            with open(FLAGS.pred_tgt_path, 'w') as f:
                pred_data = text_data.pro_tuple_data(FLAGS.pred_src_path, batch_size=FLAGS.pred_batch_size)
                for j, batch in enumerate(pred_data):
                    if batch == text_data.ONE_LINE_TOKEN:
                        f.write('\n'.encode('utf-8'))
                        continue
                    prev_predict, curr_state = sess.run(
                        [skip_thought_model.prev_predict_logits, skip_thought_model.curr_encoder_state],
                        feed_dict=pred_feed_dict(skip_thought_model, batch)
                    )

                    print(j, '------')
                    for pred_i in prev_predict:
                        for pred_j in pred_i:
                            print(text_data.vocab.index2word[pred_j], end=',')
                        print()

                    # print(curr_state)
                    f.write((' '.join(map(str, curr_state[-1][-1])) + ' ').encode('utf-8'))

    print('Elapse time: ' + str((time.time() - start_time)))


if __name__ == '__main__':
    print('start')
    tf.app.run()
    print('ok')
