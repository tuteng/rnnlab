import os
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer


from utils import to_block_name


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
        ##########################################################################
        # make instance variables
        self.num_mbs_in_doc = num_mbs_in_doc
        self.doc_token_lists, self.num_total_docs = self.make_doc_token_lists(mb_size, bptt_steps)
        self.test_doc_ids, self.train_doc_ids = self.split_corpus()
        self.num_train_doc_ids, self.num_test_doc_ids = len(self.train_doc_ids), len(self.test_doc_ids)
        self.num_total_train_docs = self.num_train_doc_ids * self.num_epochs
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict, \
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict = self.make_token_data()

    def gen_doc_token_list(self, corpus_token_list, doc_size):
        ##########################################################################
        for i in range(0, len(corpus_token_list), doc_size):
            yield corpus_token_list[i:i + doc_size]

    def make_doc_token_lists(self, mb_size, bptt_steps):
        ##########################################################################
        # load corpus
        corpus_token_list = []
        with open(os.path.join(self.data_dir, self.corpus_name, 'corpus.txt'),'r') as f:
            for doc in f.readlines():
                doc_token_list = doc.split()
                doc_token_list_cleaned = [token.strip() for token in doc_token_list]
                corpus_token_list += doc_token_list_cleaned
        ##########################################################################
        # resize
        num_tokens_in_doc = mb_size * self.num_mbs_in_doc + (bptt_steps)
        num_docs = len(corpus_token_list) / num_tokens_in_doc
        included_num_tokens = num_docs * num_tokens_in_doc
        corpus_token_list_resized = corpus_token_list[:included_num_tokens]
        doc_token_lists = list(self.gen_doc_token_list(corpus_token_list_resized, num_tokens_in_doc))
        num_total_docs = len(doc_token_lists)
        ##########################################################################
        print 'Num total docs: {}'.format(num_total_docs)
        ##########################################################################
        return doc_token_lists, num_total_docs

    def split_corpus(self):
        ##########################################################################
        # split corpus into train and test
        num_train_docs = None
        for divisor in [100, 10]:
            num_train_docs = self.num_total_docs - (self.num_total_docs % divisor)
            if num_train_docs > 1:
                break
        num_test_docs = self.num_total_docs - num_train_docs
        ##########################################################################
        print 'Split corpus into {} train docs and {} test docs'.format(num_train_docs, num_test_docs)
        ##########################################################################
        # get randomly sampled doc names for both test and train doc names
        np.random.seed(42) # this makes sure test doc ids are always the same
        test_doc_ids = np.random.choice(range(self.num_total_docs), num_test_docs, replace=False)
        train_doc_ids = [i for i in range(self.num_total_docs) if i not in test_doc_ids]
        assert len(train_doc_ids) == num_train_docs
        ##########################################################################
        num_tokens_in_train_docs = sum(len(doc_token_list) for doc_token_list
                                       in np.asarray(self.doc_token_lists)[train_doc_ids])
        num_tokens_per_doc = float(num_tokens_in_train_docs) / num_train_docs
        print 'Num tokens in training docs : {} | Num tokens per doc: {}'.format(
            num_tokens_in_train_docs, num_tokens_per_doc)
        ##########################################################################
        return test_doc_ids, train_doc_ids


    def gen_train_block_name_and_id(self, num_epochs=None, block_order=None):
        ##########################################################################
        # inits
        if num_epochs is None:
            num_epochs = self.num_epochs
            num_generator_ids = self.num_total_train_docs
        else:
            num_generator_ids = self.num_train_doc_ids
        if block_order is None: block_order = self.block_order
        ##########################################################################
        if block_order == 'shuffled':
            np.random.shuffle(self.train_doc_ids)
            print 'Shuffled training doc ids'
        elif block_order == 'reversed':
            self.train_doc_ids = self.train_doc_ids[::-1]
            print 'Reversed training doc ids'
        elif block_order == 'chronological':
            pass
        else:
            raise AttributeError('rnnlab: block_order value not recognized.'
                                 ' Please use: "chronological", "shuffled", or "reversed"')
        ##########################################################################
        # generator
        train_doc_ids_across_epochs = self.train_doc_ids * num_epochs
        for n in range(num_generator_ids):
            block_id = train_doc_ids_across_epochs[n]
            block_name = to_block_name(n+1)
            yield (block_name, block_id)


    def gen_batch(self, mb_size, bptt_steps, doc_id):
        ##########################################################################
        # get doc token ids for either training or test doc
        if doc_id == 'test':
            doc_token_list = []
            for i in self.test_doc_ids: doc_token_list += self.doc_token_lists[i]
        else:
            doc_token_list = self.doc_token_lists[doc_id]
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
            vectorized_corpus = cv.fit_transform(self.doc_token_lists).toarray()
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
        for i in range(self.num_total_docs):
            self.doc_token_lists[i] = [token if token in token_id_dict else "UNKNOWN"
                                       for token in self.doc_token_lists[i]]
        ##########################################################################
        num_tokens, num_probes = len(token_list), len(probe_list)
        print 'Vocabulary size: {} | Num probes in vocab : {}'.format(num_tokens, num_probes)
        ##########################################################################
        return token_list, token_id_dict, probe_list, probe_id_dict, probe_cat_dict, cat_list, cat_probe_list_dict
