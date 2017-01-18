import os
import numpy as np
import scipy.stats
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Corpus(object):
    """
    Creates train and test corpus to use for rnn training
    """

    def __init__(self, corpus_name, vocab_file_name=None, freq_cutoff=None, probes_name=None,
                 exclude_tokens=('PERIOD', 'QUESTION', 'EXCLAIM')):
        ##########################################################################
        # define data dir
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        ##########################################################################
        # assign instance variables
        self.corpus_name = corpus_name
        self.vocab_file_name = vocab_file_name # if specified, this is the list of tokens to include in vocab
        self.freq_cutoff = freq_cutoff # use this if no vocab file specified, None includes all tokens
        self.probes_name = probes_name # list of tokens to use for analysis after training
        self.exclude_tokens = exclude_tokens
        self.corpus_content, self.num_total_docs = self.get_corpus_content()
        ##########################################################################
        # make instance variables
        self.test_doc_ids, self.train_doc_ids = self.split_corpus()
        self.num_train_docs, self.num_test_docs = len(self.train_doc_ids), len(self.test_doc_ids)
        self.token_list, self.token_id_dict, self.probe_list,\
        self.probe_id_dict, self.probe_cat_dict, self.cat_list = self.make_token_data()
        self.probe_cf_traj_dict = self.make_probe_cf_traj_dict()


    def get_corpus_content(self):
        ##########################################################################
        print 'Loading corpus...'
        if self.exclude_tokens: print 'Excluding tokens: {}'.format(', '.join(self.exclude_tokens))
        ##########################################################################
        # load corpus
        corpus_content = []
        with open(os.path.join(self.data_dir, self.corpus_name, 'corpus.txt'),'r') as f:
            for doc in f.readlines():
                if self.exclude_tokens:
                    doc_token_list = filter(lambda a: a not in self.exclude_tokens, doc.split())
                else:
                    doc_token_list = doc.split()
                doc_token_list_cleaned = [token.strip() for token in doc_token_list]
                corpus_content.append(doc_token_list_cleaned)
        num_total_docs = len(corpus_content)
        ##########################################################################
        return corpus_content, num_total_docs


    def split_corpus(self, min=70, save_ev=10): #TODO get save_ev from config
        ##########################################################################
        # split corpus into train and test based on save_ev
        if self.num_total_docs < 2 * min:
            sys.exit('rnnlab: Not enough docs in corpus: {}.'.format(self.num_total_docs))
        if save_ev > self.num_total_docs:
            sys.exit('rnnlab: save_ev is larger than number of total documents.')
        while True:
            num_train_docs = (self.num_total_docs - min)
            if num_train_docs % save_ev == 0:
                break
            else:
                min += 1
        ##########################################################################
        num_test_docs = self.num_total_docs - num_train_docs
        print 'Split corpus into {} train docs and {} test docs'.format(num_train_docs, num_test_docs)
        ##########################################################################
        # get randomly sampled doc names for both test and train doc names
        np.random.seed(42) # this makes sure test doc ids are always the same
        test_doc_ids = list(np.random.random_integers(0, self.num_total_docs, num_test_docs))
        train_doc_ids = [i for i in range(self.num_total_docs) if i not in test_doc_ids]
        for i in train_doc_ids: assert i not in test_doc_ids
        assert len(test_doc_ids) != 0
        ##########################################################################
        return test_doc_ids, train_doc_ids


    def gen_train_block_name_and_id(self, epochs, shuffle=False): # TODO make sure shuffling works
        ##########################################################################
        max_num_train_blocks = self.num_train_docs * epochs
        if shuffle: np.random.shuffle(self.train_doc_ids)
        train_doc_ids = self.train_doc_ids * epochs
        for n in range(max_num_train_blocks):
            block_id = train_doc_ids[n]
            block_name = self.to_block_name(n+1) # +1 assures names start at 1
            yield (block_name, block_id)


    def to_block_name(self, block):
        ##########################################################################
        return ('0000' + str(block))[-4:]


    def gen_batch(self, mb_size, bptt_steps, doc_id):
        ##########################################################################
        # get doc token ids for either training or test doc
        if doc_id == 'test':
            doc_token_list = []
            for i in self.test_doc_ids: doc_token_list += self.corpus_content[i]
        else:
            doc_token_list = self.corpus_content[doc_id]
        doc_token_ids = []
        for token in doc_token_list:
            if not token in self.token_id_dict:
                token = "UNKNOWN"
            doc_token_ids.append(self.token_id_dict[token])
        ##########################################################################
        num_token_ids = len(doc_token_ids)
        num_batches = num_token_ids // mb_size
        ##########################################################################
        # reduce num_batches if necessary
        while num_batches * mb_size > num_token_ids - mb_size - bptt_steps - 1:
            num_batches -= 1
        ##########################################################################
        # moving window generator
        x = np.zeros([mb_size, bptt_steps + 1], dtype=np.int32)  # + 1 is so that y can be extracted
        for batch_number in range(num_batches):
            batch_start = batch_number * mb_size
            for i in range(mb_size):
                x[i, :] = doc_token_ids[(batch_start + i): (batch_start + i) + bptt_steps + 1]
            y = x[:, -1]
            x_shortened = x[:, :-1]
            ##########################################################################
            yield (x_shortened, y)


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
            cv = CountVectorizer() # TODO make sure this works
            vectorized_corpus = cv.fit_transform(self.corpus_content).toarray()
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
        # exclude tokens, if specified
        if self.exclude_tokens is not None:
            for token in self.exclude_tokens:
                if token in vocab_list: vocab_list.remove(token)
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
                data = (line.strip().strip('\n').strip()).split() # TODO probe list is not made in alphabetical order, problem?
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
        ##########################################################################
        num_tokens, num_probes = len(token_list), len(probe_list)
        print 'Vocabulary size: {} | Num probes in vocab : {}'.format(num_tokens, num_probes)
        ##########################################################################
        return token_list, token_id_dict, probe_list, probe_id_dict, probe_cat_dict, cat_list


    def make_tf_idf_corr_curve(self, save_ev): # TODO make this method work
        ##########################################################################
        # get docs
        path = os.path.join(self.data_dir, self.corpus_name)
        file_name = 'corpus.txt'
        with open(os.path.join(path, file_name), 'r') as f:
            docs = f.readlines()
        ##########################################################################
        # get term freq TODO make term freq for vocab words only
        cv = CountVectorizer()
        tf = cv.fit_transform(docs).toarray()
        ##########################################################################
        # get doc_tfidfs
        np.set_printoptions(threshold=1)
        tfidf = TfidfTransformer(use_idf=True, smooth_idf=False, norm=None)
        doc_tfidfs = tfidf.fit_transform(tf).toarray()
        ##########################################################################
        # make average tfidf for every weights_interval docs
        def groupedAvg(myArray, N):
            result = np.cumsum(myArray, 0)[N - 1::N] / float(N)
            result[1:] = result[1:] - result[:-1]
            return result

        avg_doc_tfidfs = groupedAvg(doc_tfidfs, save_ev)
        num_avg_doc_tfidfs = len(avg_doc_tfidfs)
        print 'Num tf-idf vectors generated: {}'.format(num_avg_doc_tfidfs)
        ##########################################################################
        # get curve of correlations of tfidf
        tf_idf_corr_curve = []
        for i in xrange(num_avg_doc_tfidfs):
            if i == 0:
                tf_idf_corr_curve.append(1)
            else:
                corr = scipy.stats.pearsonr(avg_doc_tfidfs[i - 1], doc_tfidfs[i * save_ev])[0]
                tf_idf_corr_curve.append(corr)
        ##########################################################################
        # adjust length to weights_interval
        num_data_points = len(tf_idf_corr_curve)
        max_data_points = num_data_points - num_data_points % save_ev
        print 'Returning first {}'.format(len(tf_idf_corr_curve[:max_data_points]))
        # TODO make sure the data points line up exactly with the documents
        ##########################################################################
        return tf_idf_corr_curve[:max_data_points]


    def make_probe_cf_traj_dict(self): # dict key is probe and value is numpy array with cf traj
        ##########################################################################
        print 'Making probe cumfreq trajectory dict...'
        ##########################################################################
        # make dict
        probe_cf_traj_dict = {probe: np.zeros(self.num_train_docs) for probe in self.probe_list}
        ##########################################################################
        # collect probe frequency
        for block_name, doc_id in self.gen_train_block_name_and_id(epochs=1, shuffle=False):
            doc = self.corpus_content[doc_id]
            traj_id = int(block_name) - 1
            for probe in doc:
                if probe in self.probe_id_dict:  probe_cf_traj_dict[probe][traj_id] += 1
        ##########################################################################
        # calc cumulative sum
        for probe, probe_freq_traj in probe_cf_traj_dict.iteritems():
            probe_cf_traj_dict[probe] = np.cumsum(probe_freq_traj)
        ##########################################################################
        return probe_cf_traj_dict




