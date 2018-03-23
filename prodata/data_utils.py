# -*- coding: utf-8 -*-

import logging
import random

import numpy as np
import re
import sys
import utils
import os

from collections import defaultdict
from matrixText.matrix_seg import seg_sentence

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

# patt_time = re.compile(r'^(((\s?\d\s?){4}年(\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){4}年(\s?\d\s?){1,2}月)|'
#                        r'((\s?\d\s?){4}年)|((\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){1,2}日))'
#                        r'(底|初|末|以来|年底|年初|年末|月底|月初|月末|开始|期间|份)?$')
#
# patt_num = re.compile(r'\d+\s+\，\s+\d+')
#
# patt_sym = re.compile(r'\，\s+\w+\s+[\。\，]')
#
# patt_seg = re.compile(r'\。|\？|\！|\，')

patt_han = re.compile(u'([\u4E00-\u9FD5]+)')


class Batch:
    def __init__(self, data, pad_value, go_value, eos_value):
        """Class which creates batch from data and could be passed into
        SkipthoughtModel._fill_feed_dict_* methods.

        For encoder batches, `seq_lengths` field is used in order to fill feed_dict.
        For decoder batches, `weights` field is used in order to fill feed_dict.
        (See SkipthoughtModel code)

        Args:
            data (np.array): Encoded and padded batch.
            pad_value (int): <pad> token index.
            go_value (int): <go> token index.
            eos_value (int): <eos> token index.
        """
        self.data = data
        self.pad_value = pad_value
        self.go_value = go_value
        self.eos_value = eos_value

        self._weights = utils.seq_loss_weights(self.data, self.pad_value)
        self._seq_lengths = utils.sequence_lengths(self.data, self.pad_value)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return self.data.__repr__()

    @property
    def weights(self):
        return self._weights

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def shape(self):
        return self.data.shape


class Vocab:
    GO_TOKEN = '<go>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    START_VOCAB = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.add_words(Vocab.START_VOCAB)

    @property
    def eos_value(self):
        return self.encode_word(Vocab.EOS_TOKEN)

    @property
    def go_value(self):
        return self.encode_word(Vocab.GO_TOKEN)

    @property
    def pad_value(self):
        return self.encode_word(Vocab.PAD_TOKEN)

    @property
    def unk_value(self):
        return self.encode_word(Vocab.UNK_TOKEN)

    def cut_by_freq(self, max_vocab_size):
        """Removes all words except `max_vocab_size` most frequent ones.

        Args:
            max_vocab_size (int): Target vocabulary size.
        """
        for token in Vocab.START_VOCAB:
            self.word_freq.pop(token, None)
        self.word_freq = sorted(self.word_freq.items(), key=lambda x: x[1],
                                reverse=True)[:max_vocab_size - len(Vocab.START_VOCAB)]
        self.word_freq = dict(self.word_freq)
        self._id_words_update()
        # for token in Vocab.START_VOCAB:
        #     self.word_freq[token] = 1
        # self._id_word_mappings_from_word_freq()

    def add_word(self, word, count=1):
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
        self.word_freq[word] += count

    def add_words(self, words):
        if not words:
            return
        for word in words:
            if patt_han.match(word.decode('utf-8')) or word in Vocab.START_VOCAB:
                self.add_word(word)

    def encode_word(self, word):
        if word not in self.word2index:
            return self.word2index[Vocab.UNK_TOKEN]
        else:
            return self.word2index[word]

    def encode_words(self, words, with_eos=False, with_go=False):
        encoded = []
        if with_go:
            encoded.append(self.go_value)
        encoded.extend([self.encode_word(w) for w in words])
        if with_eos:
            encoded.append(self.eos_value)
        return encoded

    def decode_idx(self, index):
        return self.index2word[index]

    def decode_idxs(self, indices):
        return [self.decode_idx(idx) for idx in indices]

    def _id_word_mappings_from_word_freq(self):
        words = self.word_freq.keys()
        self.index2word = dict(enumerate(words))
        self.word2index = {v: k for k, v in self.index2word.items()}

    def _id_words_update(self):
        vocab = Vocab.START_VOCAB + sorted(self.word_freq, key=self.word_freq.get, reverse=True)
        self.index2word = dict(enumerate(vocab))
        self.word2index = {v: k for k, v in self.index2word.items()}

    def __len__(self):
        return len(self.word2index)  # len(self.word_freq)

    def __contains__(self, item):
        return item in self.word2index


class TextData:

    NUM_LINE_TRIPLES = 3
    ONE_LINE_TOKEN = '<one>'

    def __init__(self, fname, line_min_words=5, line_process_fn=lambda x: x.strip(),
                 max_vocab_size=100000, max_len=100, verbose=10000):
        """Class for reading text data and making batches.

        Args:
            fname (str): File with data.
            line_process_fn (callable): Line processing function (str -> str). Use it if you want
                to do lemmatization or remove stopwords or smth. Default lambda x: x.strip()
            max_vocab_size (int): Maximum vocabulary size. Most frequent words are used.
            verbose (int): Verbosity level on reading data.
        """
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)
        self.fname = fname
        self.line_min_words = line_min_words
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size
        self.line_process_fn = line_process_fn

        self._check_args()

        self.vocab = Vocab()
        self.total_lines = None
        self.prev_sent = None
        self.curr_sent = None
        self.next_sent = None

        self._build_vocabulary_and_stats()

    def _check_args(self):
        import os
        assert self.max_vocab_size > 0
        assert os.path.isfile(self.fname)

    def _build_vocabulary_and_stats(self):
        """Builds vocabulary, calculates maximum length and total number of
        lines in file.
        """
        with open(self.fname) as f:
            # self.vocab = Vocab()
            self.total_lines = 0
            for line in f:
                tokens = self._tok_line(line)
                tmp_max_len = max(map(len, map(self._tok_line, seg_sentence(line)))) + 2  # 2 = len([<go>, <eos>])
                if tmp_max_len > self.max_len:
                    self.max_len = tmp_max_len
                if not tokens:
                    continue
                self.vocab.add_words(tokens)

                self.total_lines += 1
                if self.total_lines % self.verbose == 0:
                    self._logger.info('Read\t{0} lines.'.format(
                        self.total_lines))
        self.vocab.cut_by_freq(self.max_vocab_size)
        self._logger.info('Done building vocab and stats.')

    def pro_triples_data(self, batch_size):
        """Generate triples data, reads lines from file and encodes words.
        """
        self.prev_sent = list()
        self.curr_sent = list()
        self.next_sent = list()
        with open(self.fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seg_line = seg_sentence(line, self.line_min_words)
                if not seg_line:
                    continue

                triples = self.make_lines_triples(seg_line)
                self.prev_sent.extend(triples[0])
                self.curr_sent.extend(triples[1])
                self.next_sent.extend(triples[2])
                if len(self.curr_sent) < batch_size:
                    continue
                # TODO: Optimization batch-data initialization.
                for data_iter in self.triples_data_iterator(self.prev_sent, self.curr_sent, self.next_sent,
                                                            max_len=self.max_len, batch_size=batch_size):
                    yield data_iter

                self.prev_sent = list()
                self.curr_sent = list()
                self.next_sent = list()

    def pro_tuple_data(self, out_file_name, batch_size=1):
        """Generate one tuple data, reads lines from file and encodes words.
        """
        if not out_file_name or not os.path.exists(out_file_name):
            return
        curr_sent = []
        with open(out_file_name) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seg_line = seg_sentence(line, self.line_min_words)
                if not seg_line:
                    continue

                curr_sent.extend(filter(lambda x: x, seg_line))
                for data_iter in self.lines_curr_iterator(curr_sent, batch_size=batch_size):
                    yield data_iter
                yield TextData.ONE_LINE_TOKEN
                curr_sent = []

    def _tok_line(self, line):
        """Tokenizes raw line.

        Args:
            line (str): Raw line.
        Returns:
            tokens (list of str): List of tokens.
        """
        if not line or not isinstance(line, str):
            return
        return self.line_process_fn(line).split()

    def encode_line(self, line, with_eos=False, with_go=False):
        """Encodes raw line to list of word indices. Applies ``line_process_fn`` before encoding.

        Args:
            line (str): Raw lines.
            with_eos (bool): Whether to append eos_value at the end or not.
            with_go (bool): Whether to append go_token in the beginning of line or not.
        Returns:
             encoded (list of ints): Encoded line.
        """
        tokens = self._tok_line(line)
        encoded = self.vocab.encode_words(tokens, with_eos, with_go)
        return encoded

    def encode_lines(self, lines, with_eos=False, with_go=False):
        """Encodes raw lines to list of word indices. Applies ``line_process_fn`` for each line.

        Args:
            lines (list of str): List of raw lines.
            with_eos (bool): Whether to append eos_value at the end of each line or not.
            with_go (bool): Whether to append go_token in the beginning of each line or not.
        Returns:
             encoded (list of list of ints): List of encoded lines.
        """
        encoded = [self.encode_line(line, with_eos, with_go) for line in lines]
        return encoded

    def decode_line(self, encoded_line):
        return self.vocab.decode_idxs(encoded_line)

    def make_batch(self, encoded_lines, max_len=None):
        """Makes `Batch` instance based on `encoded_lines`.

        Args:
            encoded_lines (list of list of int): List of encoded lines. Encoded lines
            can be obtained via ``encode_lines`` or ``encode_line`` methods.
            max_len (int): If not None, lines will be padded up to max_len with vocab.pad_value.
                Otherwise, lines will be padded using maximum length of line in ``encoded_lines``.
        Returns:
            batch (Batch): Batch instance.
        """
        if not max_len:
            max_len = min(max(map(len, encoded_lines)), self.max_len)
        encoded_lines = [line[:max_len-1] for line in encoded_lines]
        padded_lines = utils.pad_sequences(encoded_lines, max_len, self.vocab.pad_value)
        batch = Batch(padded_lines, self.vocab.pad_value, self.vocab.go_value, self.vocab.eos_value)
        return batch

    @staticmethod
    def _make_triples_for_paragraph(paragraph):
        """Generate prev, curr, next lists based on paragraph.
        """
        if len(paragraph) < TextData.NUM_LINE_TRIPLES:
            return [], [], []
        prev = paragraph[:-2]
        curr = paragraph[1:-1]
        next = paragraph[2:]
        return prev, curr, next

    def make_triples(self, lines):
        """Returns prev, curr, next lists based on lines.

        Context is not shared between different paragraphs in text. So, last line in one paragraph
        will not be in context with first line in the next paragraph.
        Paragraphs must be separated by '\n\n'

        There will be asymmetric context for first and last lines.

        Args:
            lines (list of str): List of lines.
        Returns:
            prev, curr, next (tuple of list of str):
        """
        idxs = [-1]+list(filter(None, [i if len(lines[i]) == 0 else None for i in range(len(lines))]))+[len(lines)]
        all_prev, all_curr, all_next = [], [], []
        for start, end in zip(idxs[:-1], idxs[1:]):
            tmp_prev, tmp_curr, tmp_next = self._make_triples_for_paragraph(lines[start+1:end])
            if tmp_prev == [] or tmp_curr == [] or tmp_next == []:
                continue
            all_prev.extend(tmp_prev)
            all_curr.extend(tmp_curr)
            all_next.extend(tmp_next)
        return all_prev, all_curr, all_next

    @staticmethod
    def make_lines_triples(lines):
        """Returns prev, curr, next lists based on lines.
        """
        lines = filter(lambda x: x, lines)
        if len(lines) < TextData.NUM_LINE_TRIPLES:
            return [], [], []
        # prev, curr, next
        return lines[:-2], lines[1:-1], lines[2:]

    def triples_data_iterator(self, prev_data, curr_data, next_data,
                              max_len, batch_size=64, shuffle=False):
        """Creates iterator for (current sentence, prev sentence, next sentence)
        data. Is is useful for training skip-thought vectors.

        Args:
            curr_data (list of lists of ints): List with raw lines which corresponds to current sentences.
                Lines can be with different lengths. They will be encoder inputs.
            prev_data (list of lists of ints): List with raw previous
                lines. Lines can be with different lengths.
            next_data (list of lists of ints): List with raw next lines.
                Lines can be with different lengths.
            max_len (int): Maximum length for padding previous and next sentences.
            batch_size (int): Size of batch.
            shuffle (bool): Whether to shuffle data or not.

        Yields:
            enc_inp, prev_inp, prev_targ, next_inp, next_targ (Batch)

        """
        if shuffle:
            indices = np.random.permutation(len(curr_data))
            curr_data = [curr_data[i] for i in indices]
            prev_data = [prev_data[i] for i in indices]
            next_data = [next_data[i] for i in indices]

        total_processed_examples = 0
        total_steps = int(np.ceil(len(curr_data) / float(batch_size)))
        for step in range(total_steps):
            batch_start = step * batch_size

            curr = curr_data[batch_start:batch_start + batch_size]
            prev = prev_data[batch_start:batch_start + batch_size]
            next = next_data[batch_start:batch_start + batch_size]

            if batch_start + batch_size > len(curr_data):
                num_index = min(batch_size, batch_start + batch_size - len(curr_data))
                data_index = random.sample(range(len(curr_data)), num_index)
                for i_index in data_index:
                    curr.append(curr_data[i_index])
                    prev.append(prev_data[i_index])
                    next.append(next_data[i_index])

            enc_inp = self.make_batch(self.encode_lines(curr))

            prev_inp = self.make_batch(self.encode_lines(prev, with_go=True), max_len)
            prev_targ = self.make_batch(self.encode_lines(prev, with_eos=True), max_len)

            next_inp = self.make_batch(self.encode_lines(next, with_go=True), max_len)
            next_targ = self.make_batch(self.encode_lines(next, with_eos=True), max_len)
            assert prev_inp.shape == prev_targ.shape == next_inp.shape == next_targ.shape, (prev, curr, next)

            yield enc_inp, prev_inp, prev_targ, next_inp, next_targ

            total_processed_examples += len(curr)

            if total_processed_examples >= len(curr_data):
                break

        assert total_processed_examples >= len(curr_data), \
            'Expected {} and processed {}'.format(len(curr_data),
                                                  total_processed_examples)

    def lines_curr_iterator(self, curr_data, batch_size=1, shuffle=False):
        """Creates iterator for current sentence data. Is is useful for predicting | encoding skip-thought vectors.
        """

        if shuffle:
            indices = np.random.permutation(len(curr_data))
            curr_data = [curr_data[i] for i in indices]

        total_processed_examples = 0
        total_steps = int(np.ceil(len(curr_data) / float(batch_size)))
        for step in range(total_steps):
            batch_start = step * batch_size

            curr = curr_data[batch_start:batch_start + batch_size]

            if batch_start + batch_size > len(curr_data):
                num_index = min(batch_size, batch_start + batch_size - len(curr_data))
                data_index = random.sample(range(len(curr_data)), num_index)
                for i_index in data_index:
                    curr.append(curr_data[i_index])

            enc_inp = self.make_batch(self.encode_lines(curr))
            assert curr

            yield enc_inp

            total_processed_examples += len(curr)
            if total_processed_examples >= len(curr_data):
                break
        assert total_processed_examples >= len(curr_data), \
            'Expected {} and processed {}'.format(len(curr_data),
                                                  total_processed_examples)

    # @staticmethod
    # def seg_sentence(sentence, min_words_num=5):
    #     """ Segment sentence
    #
    #     Args:
    #         sentence: sentence
    #         min_words_num: a sentence contains min_words_num words
    #     Returns:
    #         segment sentence
    #     """
    #     if not sentence:
    #         return None
    #
    #     num_seg_all = patt_num.findall(sentence)
    #     while num_seg_all:
    #         for num_one in num_seg_all:
    #             sentence = sentence.replace(num_one, num_one.replace('，', ','))
    #         num_seg_all = patt_num.findall(sentence)
    #
    #     sym_seg_all = patt_sym.findall(sentence)
    #     while sym_seg_all:
    #         for sym_one in sym_seg_all:
    #             sentence = sentence.replace(sym_one, sym_one.replace('，', ',', 1))
    #         sym_seg_all = patt_sym.findall(sentence)
    #
    #     res_segment = patt_seg.split(sentence)
    #     res_seg_len = len(res_segment)
    #
    #     min_sent_len = min_words_num * len('。')
    #     for i in range(res_seg_len):
    #         if i >= res_seg_len:
    #             break
    #         if patt_time.match(res_segment[i].replace(' ', '')) \
    #                 or len(res_segment[i].replace(' ', '')) < min_sent_len:
    #             if len(res_segment) == 1:
    #                 break
    #             if i < len(res_segment) - 1:
    #                 res_segment[i] += ',' + res_segment[i + 1]
    #                 res_segment.remove(res_segment[i + 1])
    #             else:
    #                 res_segment[i - 1] += ',' + res_segment[i]
    #                 res_segment.remove(res_segment[i])
    #
    #             res_seg_len = len(res_segment)
    #
    #     return res_segment
