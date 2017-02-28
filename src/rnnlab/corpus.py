import os
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer

from utils import to_mb_name


class Corpus(object):
    """
    Creates train and test corpus to use for rnn training
    """

    def __init__(self, corpus_name, block_order, num_epochs, mb_size, bptt_steps,
                 num_mbs_in_doc, vocab_file_name, freq_cutoff, probes_name):
        ##########################################################################
        # define data dir
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        ##########################################################################
        # assign instance variables
        self.corpus_name = corpus_name
        self.block_order = block_order
        self.num_epochs = num_epochs
        self.vocab_file_name = vocab_file_name # if specified, this is the list of tokens to include in vocab
        self.freq_cutoff = freq_cutoff # use this if no vocab file specified, None includes all tokens
        self.probes_name = probes_name # list of tokens to use for analysis after training
        self.num_mbs_in_doc = num_mbs_in_doc  # number of minibatches contained in one document
        ##########################################################################
        # make instance variables
        self.train_doc_token_lists, self.test_doc_token_lists = self.make_doc_token_lists(mb_size, bptt_steps)
        self.num_train_doc_ids = len(self.train_doc_token_lists)
        self.num_test_doc_ids = len(self.test_doc_token_lists)
        self.num_blocks = self.num_train_doc_ids * num_epochs
        print 'Num train docs: {} |Num test docs: {}'.format(self.num_train_doc_ids, self.num_test_doc_ids)
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict, \
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict = self.make_token_data()

    def gen_doc_token_lists(self, corpus_token_list, doc_size):
        ##########################################################################
        for i in range(0, len(corpus_token_list), doc_size):
            yield corpus_token_list[i:i + doc_size]

    def make_doc_token_lists(self, mb_size, bptt_steps, num_test_docs=100):
        ##########################################################################
        # load corpus
        with open(os.path.join(self.data_dir, self.corpus_name, 'corpus.txt'), 'r') as f:
            corpus_doc_list = list(f.readlines())
        num_docs_in_corpus = len(corpus_doc_list)
        ##########################################################################
        # make test ids
        np.random.seed(42)  # this makes sure test doc ids are always the same
        test_doc_ids = np.random.choice(range(num_docs_in_corpus), num_test_docs, replace=False)
        ##########################################################################
        # make train and test corpus token lists
        train_corpus_token_list, test_corpus_token_list = [], []
        for doc_id, doc in enumerate(corpus_doc_list):
            doc_token_list = doc.split()
            doc_token_list_cleaned = [token.strip() for token in doc_token_list]
            if doc_id not in test_doc_ids:
                train_corpus_token_list += doc_token_list_cleaned
            else:
                test_corpus_token_list += doc_token_list_cleaned
        ##########################################################################
        # make train and test doc token lists
        token_lists = []
        num_tokens_in_doc = mb_size * self.num_mbs_in_doc + (bptt_steps)
        for token_list in [train_corpus_token_list, test_corpus_token_list]:
            # resize
            num_train_docs = len(token_list) / num_tokens_in_doc
            num_tokens_in_corpus = num_train_docs * num_tokens_in_doc
            ##########################################################################
            # split into docs
            token_list_truncated = token_list[:num_tokens_in_corpus]
            doc_token_lists = list(self.gen_doc_token_lists(token_list_truncated, num_tokens_in_doc))
            token_lists.append(doc_token_lists)
        train_doc_token_lists, test_doc_token_lists = token_lists
        ##########################################################################
        return train_doc_token_lists, test_doc_token_lists

    def gen_train_doc_id(self, num_epochs=None, block_order=None):
        ##########################################################################
        # inits
        if num_epochs is None: num_epochs = self.num_epochs
        if block_order is None: block_order = self.block_order
        ##########################################################################
        # arrange train_doc_ids
        train_doc_ids = range(self.num_train_doc_ids)
        if block_order == 'shuffled':
            np.random.shuffle(train_doc_ids)
            print 'Shuffled training doc ids'
        elif block_order == 'reversed':
            train_doc_ids = train_doc_ids[::-1]
            print 'Reversed training doc ids'
        elif block_order == 'chronological':
            pass
        else:
            raise AttributeError('rnnlab: block_order value not recognized.'
                                 ' Please use: "chronological", "shuffled", or "reversed"')
        ##########################################################################
        # generator
        doc_ids = train_doc_ids * num_epochs
        for doc_id in doc_ids:
            yield doc_id


    def gen_batch(self, mb_size, bptt_steps, doc_id):
        ##########################################################################
        # get doc token ids for either training or test doc
        if doc_id == 'test':
            doc_token_list = []
            for i in range(self.num_test_doc_ids): doc_token_list += self.test_doc_token_lists[i]  # merge test docs
        else:
            doc_token_list = self.train_doc_token_lists[doc_id]
        ##########################################################################
        # make batches_mat
        num_rows = self.num_mbs_in_doc * mb_size
        batches_mat_X = np.zeros((num_rows, bptt_steps), dtype=np.int)
        batches_mat_Y = np.zeros((num_rows, 1), dtype=np.int)
        for row_id in range(num_rows):
            token_window_with_y = doc_token_list[row_id:row_id + bptt_steps + 1]
            token_window = token_window_with_y[:-1]
            last_token = token_window_with_y[-1]
            batches_mat_X[row_id, :] = [self.token_id_dict[token] for token in token_window]
            batches_mat_Y[row_id] = self.token_id_dict[last_token]
        ##########################################################################
        # split_array generator
        for X, Y in zip(np.vsplit(batches_mat_X, self.num_mbs_in_doc),
                        np.vsplit(batches_mat_Y, self.num_mbs_in_doc)):
            ##########################################################################
            yield (X, Y.flatten())


    def make_token_data(self):
        ##########################################################################
        # inits
        vocab_list = []
        token_list = []
        probe_list = []
        probe_cat_dict = {}
        ##########################################################################
        # make vocab from freq_cutoff
        if not self.vocab_file_name:
            cv = CountVectorizer()
            vectorized_corpus = cv.fit_transform(self.train_doc_token_lists).toarray()
            types = cv.get_feature_names()
            type_freqs = np.asarray(vectorized_corpus.sum(axis=0)).ravel()
            assert len(type_freqs) == len(types)
            ids = np.where(type_freqs > self.freq_cutoff)
            vocab_list = [types[i] for i in ids]
            assert len(vocab_list) != 0
        ##########################################################################
        # make vocab from vocab file
        else:
            path = os.path.join(self.data_dir, self.corpus_name)
            file_name = '{}.txt'.format(self.vocab_file_name)
            with open(os.path.join(path, file_name), 'r') as f:
                for line in f.readlines():
                    token = line.strip().strip('\n')
                    vocab_list.append(token)
        ##########################################################################
        # make token list and dict from vocab
        vocab_list.append("UNKNOWN")
        for token in vocab_list:
            if not token in token_list:
                token_list.append(token)
        ##########################################################################
        # add probes to token_list, if not already in token_list
        path = os.path.join(self.data_dir, 'probes')
        probe_file_name = '{}.txt'.format(self.probes_name)
        with open(os.path.join(path, probe_file_name), 'r') as f:
            for line in f:
                data = (line.strip().strip('\n').strip()).split()
                cat = data[0]
                probe = data[1]
                if probe not in token_list:
                    if '' == raw_input(
                            'Probe "{}" not in token list. Increase vocab thr or press ENTER to continue'.format(
                                probe)):
                        token_list.append(probe)
                probe_list.append(probe)
                probe_cat_dict[probe] = cat
        ##########################################################################
        # sort and make id dicts
        token_list = sorted(token_list)
        token_id_dict = {token: token_list.index(token) for token in token_list}
        probe_list = sorted(probe_list)
        probe_id_dict = {probe: probe_list.index(probe) for probe in probe_list}
        cat_list = list(sorted(set(probe_cat_dict.values())))
        cat_probe_list_dict = {cat: [probe for probe in probe_list if probe_cat_dict[probe] == cat]
                               for cat in cat_list}
        ##########################################################################
        # replace out of vocab words with UNKNOWN token
        for i in range(self.num_train_doc_ids):
            self.train_doc_token_lists[i] = [token if token in token_id_dict else "UNKNOWN"
                                             for token in self.train_doc_token_lists[i]]
        for i in range(self.num_test_doc_ids):
            self.test_doc_token_lists[i] = [token if token in token_id_dict else "UNKNOWN"
                                            for token in self.test_doc_token_lists[i]]
        ##########################################################################
        num_tokens, num_probes = len(token_list), len(probe_list)
        print 'Vocabulary size: {} | Num probes in vocab : {}'.format(num_tokens, num_probes)
        ##########################################################################
        return token_list, token_id_dict, probe_list, probe_id_dict, probe_cat_dict, cat_list, cat_probe_list_dict
