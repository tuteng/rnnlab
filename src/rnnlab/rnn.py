import csv
import datetime
import os
import shutil
import sys
import time
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import pyprind
import tensorflow as tf

from corpus import Corpus
from database import DataBase

from utils import create_rnn_graph
from utils import check_disk_space
from utils import get_log_entries_list
from utils import is_training_completed
from utils import load_rnnlabrc
from utils import make_lex_div_traj
from utils import make_probe_cf_traj_dict
from utils import make_rnnlab_alias
from utils import make_tf_idf_mat
from utils import remove_log_entry
from utils import to_block_name
from utils import calc_token_ba_list



np.set_printoptions(suppress=True)

class RNN():
    """
    Instantiates an RNN model, includes methods to train and save training data
    """
    def __init__(self, flavor, user_configs):
        ##########################################################################
        # define directories
        self.runs_dir = load_rnnlabrc('runs_dir')
        self.log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
        ##########################################################################
        # make configs dict
        self.configs_dict = self.make_configs_dict(user_configs, flavor)
        ##########################################################################
        # calc num_epochs and save_ev_block
        self.num_reps = int(self.configs_dict['num_reps'])
        self.num_iterations = int(self.configs_dict['num_iterations'])
        self.num_epochs = int(self.num_reps / self.num_iterations)
        self.save_ev_block = int(self.configs_dict['save_ev']) * self.num_epochs
        if not self.num_reps % self.num_iterations == 0:
            sys.exit('rnnlab: "num_reps" must be divisible by "num_iterations"')
        print 'Num reps: {} / Num iterations: {} --> Num epochs : {}'.format(
            self.num_reps, self.num_iterations, self.num_epochs)
        print 'Num epochs: {} x save_ev: {} --> Saving every {}th block'.format(
            self.num_epochs,int(self.configs_dict['save_ev']), self.save_ev_block)
        ##########################################################################
        # make corpus
        corpus_kwargs = {key : self.configs_dict[key]
                       for key in ['corpus_name', 'vocab_file_name', 'freq_cutoff',
                                   'probes_name', 'save_ev','block_order']}
        corpus_kwargs['num_epochs'] = self.num_epochs
        self.corpus = Corpus(**corpus_kwargs)
        ##########################################################################
        # create rnn_graph
        num_input_units = len(self.corpus.token_list)
        self.rnn_graph = create_rnn_graph(num_input_units, self.configs_dict)
        ##########################################################################
        # assign instance variables
        self.model_name = self.configs_dict['model_name']
        self.block_order = str(self.configs_dict['block_order'])
        self.mb_size = int(self.configs_dict['mb_size'])
        self.num_hidden_units = int(self.configs_dict['num_hidden_units'])
        self.bptt_steps = int(self.configs_dict['bptt_steps'])
        self.stop_block_name = to_block_name(self.corpus.num_train_doc_ids * self.num_epochs)
        self.avg_token_ba_list = []


    def train(self):
        ##########################################################################
        self.prepare_training()
        ##########################################################################
        # inits
        elapsed_start = time.time()
        num_mbs_trained, total_num_examples_seen, completed = 0, 0, False
        ##########################################################################
        # save to data base before training
        print 'Saving data from untrained model...'
        self.data_step(to_block_name(0))
        ##########################################################################
        # training blocks
        for block_name, block_id in self.corpus.gen_train_block_name_and_id():
            self.print_train_stats(elapsed_start, block_name, block_id, num_mbs_trained)
            ##########################################################################
            # training iterations
            for iteration_counter in xrange(self.num_iterations):
                ##########################################################################
                # batch
                for (X, Y) in self.corpus.gen_batch(self.mb_size, self.bptt_steps, block_id):
                    num_mbs_trained += 1
                    total_num_examples_seen += self.mb_size
                    self.rnn_graph.sess.run(self.rnn_graph.train_step,
                                            feed_dict={self.rnn_graph.x: X, self.rnn_graph.y: Y})
            ##########################################################################
            # data_step
            if int(block_name) % self.save_ev_block == 0:
                self.data_step(block_name)
        ##########################################################################
        # at end of training, close session and upload data
        self.complete_training()

    def data_step(self, block_name):
        ##########################################################################
        start = time.time()
        ##########################################################################
        # save checkpoint
        self.save_ckpt(block_name)
        ##########################################################################
        # extract acts
        df = self.make_df()
        ##########################################################################
        # analyze acts
        database = DataBase(self.configs_dict, df, block_name)
        test_pp, avg_token_ba, token_ba_list = self.analyze(database)
        ##########################################################################
        # save to disk
        database.save_to_disk(block_name, test_pp, avg_token_ba, token_ba_list)
        #########################################################################
        # print to console
        print 'Test perplexity : {} |Avg token ba : {} |Database ops completed in {} secs'.format(
            test_pp, avg_token_ba, int(abs(time.time() - start)))
        self.avg_token_ba_list.append(avg_token_ba)
        #########################################################################
        # update log with best_avg_token_ba and completed
        best_avg_token_ba = max(self.avg_token_ba_list)
        completed = 1 if self.stop_block_name == block_name else 0
        self.update_log(best_avg_token_ba, completed)

    def analyze(self, database):
        ##########################################################################
        # calc token_ba_list
        num_ba_samples = int(load_rnnlabrc('num_ba_samples'))
        token_ba_list = calc_token_ba_list(database, num_samples=num_ba_samples)
        avg_token_ba = np.mean(token_ba_list)
        ##########################################################################
        # calc test perplexity
        test_pp = self.calc_test_pp()
        ##########################################################################
        return test_pp, avg_token_ba, token_ba_list



    def save_ckpt(self, block_name):
        ##########################################################################
        ckpt_outfile = "weights_at_block_{}.ckpt".format(block_name)
        path = os.path.join(self.runs_dir, self.model_name, 'Weights')
        self.rnn_graph.saver.save(self.rnn_graph.sess, os.path.join(path, ckpt_outfile))


    def prepare_training(self):
        ##########################################################################
        # check disk space
        check_disk_space()
        ##########################################################################
        # remove low priority data from previous rnns
        self.remove_old_data()
        ##########################################################################
        # make log entry
        self.make_log_entry()
        ##########################################################################
        # make alias so user can call browser app from bash with 'rnnlab'
        app_dirname = os.path.dirname(__file__)
        make_rnnlab_alias(app_dirname)
        ##########################################################################
        # create data dirs
        for dir in ['Configs', 'Weights', 'Balanced_Accuracy', 'Data_Frame',
                    'Token_Data', 'Corpus_Data', 'Classifier', 'Figures']:
            path = os.path.join(self.runs_dir, self.model_name, dir)
            if not os.path.isdir(path):
                os.makedirs(path)
        ##########################################################################
        #  save token data
        path = os.path.join(self.runs_dir, self.model_name, 'Token_Data')
        file_name = 'token_data.npz'.format(self.model_name)
        np.savez(os.path.join(path, file_name),
                 token_list=self.corpus.token_list,
                 token_id_dict=self.corpus.token_id_dict,
                 probe_id_dict=self.corpus.probe_id_dict,
                 probe_list=self.corpus.probe_list,
                 probe_cat_dict=self.corpus.probe_cat_dict,
                 cat_list=self.corpus.cat_list,
                 cat_probe_list_dict=self.corpus.cat_probe_list_dict)
        ##########################################################################
        # make corpus data
        probe_cf_traj_dict = make_probe_cf_traj_dict(self.corpus, self.save_ev_block)
        tf_idf_mat = make_tf_idf_mat(self.corpus, self.save_ev_block)
        lex_div_traj = make_lex_div_traj(self.corpus, self.save_ev_block)
        num_input_units = len(self.corpus.token_list)
        ##########################################################################
        # save corpus data
        path = os.path.join(self.runs_dir, self.model_name, 'Corpus_Data')
        file_name = 'corpus_data.npz'.format(self.model_name)
        np.savez(os.path.join(path, file_name),
                 probe_cf_traj_dict=probe_cf_traj_dict,
                 num_train_doc_ids=self.corpus.num_train_doc_ids,
                 tf_idf_mat=tf_idf_mat,
                 lex_div_traj=lex_div_traj,
                 num_input_units=num_input_units)
        ##########################################################################
        #  save configs_dict to npy
        path = os.path.join(self.runs_dir, self.model_name, 'Configs')
        file_name = 'configs_dict.npy'
        np.save(os.path.join(path, file_name), self.configs_dict)
        ##########################################################################
        print 'Saved token_data, corpus_data, and configs_dict'


    def remove_old_data(self, num_more_recent=5):
        ##########################################################################
        del_candidates_list = []
        model_names_deleted = []
        ##########################################################################
        # get runs log
        if os.path.isfile(self.log_path):
            log_entries_list, headers = get_log_entries_list()
            ##########################################################################
            # make del_candidates_list
            if log_entries_list:
                for log_entry in log_entries_list:
                    model_name, best_avg_token_ba = log_entry[0], log_entry[-1]
                    flavor = model_name.split('_')[-1]
                    del_candidates_list.append((model_name, best_avg_token_ba, flavor))
            ##########################################################################
            # sort del_candidates_list
            sorted_del_candidates_list = sorted(del_candidates_list, key=itemgetter(2, 1))
            ##########################################################################
            # group sorted_del_candidates_list
            for flavor, group in groupby(sorted_del_candidates_list, itemgetter(2)):
                ##########################################################################
                # delete model data until length of group is num_more_recent
                group_list = list(group)
                while len(group_list) > num_more_recent:
                    model_name = group_list[0][0]
                    model_names_deleted.append(model_name)
                    if os.path.isdir(os.path.join(self.runs_dir, model_name)):
                        shutil.rmtree(os.path.join(self.runs_dir, model_name))
                        print('Deleted {} from runs dir'.format(model_name))
                    group_list.pop(0)
            ##########################################################################
            # remove log entries corresponding with models deleted above and if not completed (completed data is still informative)
            for model_name in model_names_deleted:
                if not is_training_completed(model_name):
                    remove_log_entry(model_name)
        ##########################################################################
        else:
            print 'rnnlab WARNING: Could not find {}'.format(self.log_path)


    def make_log_entry(self):
        ##########################################################################
        # clean up dict for writing
        configs_dict = self.configs_dict.copy()
        for entry_to_pop in ['flavor', 'model_name', 'corpus_name']:
            configs_dict.pop(entry_to_pop)
        ##########################################################################
        # write log header
        if not os.path.isfile(self.log_path):
            header = [str(key) for key in configs_dict.keys()]
            header.insert(0, 'model_name')
            header.append('completed')
            header.append('best_token_ba')
            writer = csv.writer(open(self.log_path, 'w'))
            writer.writerow(header)
            print 'Creating rnnlab_log.csv'
            time.sleep(1)
        ##########################################################################
        # add new entry
        writer = csv.writer(open(self.log_path, 'a'))
        all_params_list = [str(configs_dict[key]) for key in configs_dict.keys()]
        all_params_list.insert(0, self.model_name)
        all_params_list.append('0')  # completed
        all_params_list.append('0')  # best_token_ba
        writer.writerow(all_params_list)


    def update_log(self, best_avg_token_ba, completed):
        ##########################################################################
        log_content = csv.reader(open(self.log_path, 'r'))
        log_content_new = []
        for row in log_content:
            if row[0] == self.model_name:
                row[-1] = format(best_avg_token_ba, '.3f')
                row[-2] = int(completed)
            log_content_new.append(row)
        with open(self.log_path, 'w') as f:
            writer = csv.writer(f)
            for row in log_content_new:
                writer.writerow(row)


    def print_train_stats(self, elapsed_start, block_name, block_id, num_mbs_trained):
        ##########################################################################
        secs = int(abs(elapsed_start - time.time()))
        hours = int(float(secs) / 3600)
        print '{} |Block Name: {}/{} Id: {:>4} |Batch: {:>10} |Elapsed: {:>2} hrs'.format(
            self.model_name, block_name, self.corpus.num_total_train_docs, block_id, num_mbs_trained, hours)


    def send_data_to_web_app(self):
        ##########################################################################
        pass


    def complete_training(self, remove_dfs=False):
        ##########################################################################
        # send data to web app
        self.send_data_to_web_app()
        ##########################################################################
        # delete dfs to save space, if specified
        if remove_dfs:
            path = os.path.join(self.runs_dir, self.model_name, 'Data_Frame')
            for file_name in sorted(os.listdir(path))[:-1]:
                os.remove(os.path.join(path, file_name))
        ##########################################################################
        self.rnn_graph.sess.close()
        tf.reset_default_graph()
        print '{} Training Session Closed and Graph reset\n\n'.format(self.model_name)


    def make_df(self, fast=True, mbsize=128):
        ##########################################################################
        print 'Extracting activations for probes...'
        ##########################################################################
        # inits
        pbar = pyprind.ProgBar(self.corpus.num_train_doc_ids)
        acts_mat_list = []
        tokens_in_mb = []
        mb_data_dict_keys = ['X', 'doc_name', 'probe', 'probe_id', 'Y', 'cat', 'token_pp']
        mb_data_dict = {key: [] for key in mb_data_dict_keys}
        probe_X = np.zeros((mbsize, self.bptt_steps), dtype=int)
        probe_Y = np.zeros(mbsize, dtype=int)
        num_tokens_in_batch = 0
        probe_freq_dict = {probe: 0 for probe in self.corpus.probe_list}
        ##########################################################################
        # get mb_data_dict from each block_name and add to df
        for block_name, block_id in self.corpus.gen_train_block_name_and_id(1, 'chronological'):
            pbar.update()
            for (X, Y) in self.corpus.gen_batch(mbsize, self.bptt_steps, block_id):
                tokens = [self.corpus.token_list[token_id] for token_id in X[:, -1]]
                for n, token in enumerate(tokens):
                    ##########################################################################
                    # if token is probe, add data to mb_data_dict
                    if token in self.corpus.probe_id_dict and token not in tokens_in_mb:
                        if fast:
                            tokens_in_mb.append(token)
                        mb_data_dict['X'].append(X[n]) # this is a list of token_ids in bptt window
                        mb_data_dict['doc_name'].append(int(block_name))
                        mb_data_dict['probe'].append(token)
                        mb_data_dict['probe_id'].append(self.corpus.probe_id_dict[token])
                        mb_data_dict['Y'].append(Y[n])
                        mb_data_dict['cat'].append(self.corpus.probe_cat_dict[token])
                        ##########################################################################
                        # build batch
                        probe_X[num_tokens_in_batch] = X[n]
                        probe_Y[num_tokens_in_batch] = Y[n]
                        num_tokens_in_batch += 1
                        probe_freq_dict[token] += 1.0
                    ##########################################################################
                    # when batch ready, calculate acts_mat and pp_vec
                    if num_tokens_in_batch == mbsize:
                        [acts_mat, pp_vec] = self.rnn_graph.sess.run(
                            [self.rnn_graph.last_hidden_state, self.rnn_graph.pp_vec],
                            feed_dict={self.rnn_graph.x: probe_X, self.rnn_graph.y: probe_Y})
                        acts_mat_list.append(acts_mat)
                        mb_data_dict['token_pp'] += pp_vec.tolist()
                        ##########################################################################
                        # reset batch
                        tokens_in_mb = []
                        num_tokens_in_batch = 0
                        probe_X = np.zeros((mbsize, self.bptt_steps), dtype=int)
                        probe_Y = np.zeros(mbsize, dtype=int)
        ##########################################################################
        # make df
        acts_for_df = np.vstack((mat for mat in acts_mat_list))
        acts_for_df_labels = ['H{}'.format(i) for i in range(self.num_hidden_units)]
        df = pd.DataFrame(acts_for_df, columns=acts_for_df_labels)
        num_data = len(mb_data_dict['token_pp'])
        for df_column_label in mb_data_dict_keys:
            if 'X' == df_column_label:
               for n, row in enumerate(np.asarray(mb_data_dict['X'][:num_data]).T):
                   df['X{}'.format(n)] = row
            else:
                df[df_column_label] = mb_data_dict[df_column_label][:num_data]
        ##########################################################################
        return df



    def calc_test_pp(self, mbsize=128):
        ##########################################################################
        print 'Calculating test perplexity...'
        ##########################################################################
        test_pp_sum, num_batches, test_pp = 0, 0, 0
        for (X, Y) in self.corpus.gen_batch(mbsize, self.bptt_steps, 'test'):
            test_pp_batch = self.rnn_graph.sess.run(self.rnn_graph.mean_pp,
                                                    feed_dict={self.rnn_graph.x: X, self.rnn_graph.y: Y})
            test_pp_sum += test_pp_batch
            num_batches += 1
        test_pp = int(test_pp_sum / num_batches)
        ##########################################################################
        return test_pp

    def make_model_name(self, flavor):
        ##########################################################################
        # make model_name (flavor in name is required for data removal to work)
        time_of_init = datetime.datetime.now().strftime('%m-%d-%H-%M')
        model_name = '{}_{}'.format(time_of_init, flavor)
        ##########################################################################
        # exit if model name dir already exists
        if os.path.isdir(os.path.join(self.runs_dir, model_name)):
            sys.exit('rnnlab : Model name already exists. This is typically caused by starting two models '
                     'within the same minute. Please try again.')
        ##########################################################################
        return model_name

    def make_configs_dict(self, user_configs, flavor):
        ##########################################################################
        # default configs
        configs_dict = {
            'bptt_steps': 7,
            'num_hidden_units': 512,
            'mb_size': 64,
            'learning_rate': 0.001,
            'weight_init': 'uus',
            'act_function': 'sigmoid',
            'bias': 1,
            'leakage': 0.95,
            'num_iterations': 20,
            'num_reps': 20,
            'block_order': 'chronological',
            'optimizer': 'adagrad',
            'save_ev': 100,
            'model_name': self.make_model_name(flavor),
            'flavor': flavor,
            'corpus_name': None,
            'vocab_file_name': None,
            'freq_cutoff': None,
            'probes_name': None}
        ##########################################################################
        # overwrite default config dict
        overwritten_list = []
        print 'Overwriting default configs with user configs:'
        for user_config in user_configs:
            config_name = user_config[0]
            config_value = user_config[1]
            config_value = False if config_value == 'False' else config_value
            config_value = int(config_value) if config_value.isdigit() else config_value
            if not config_value == 'default':
                if config_name in configs_dict:
                    print config_name, '{} -> {}'.format(configs_dict[config_name], config_value)
                    configs_dict[config_name] = config_value
                    overwritten_list.append(config_name)
        ##########################################################################
        # check that all required configs specified
        for c in ['corpus_name']:
            if c not in overwritten_list: sys.exit('rnnlab WARNING: Did not find "{}" in user configs'.format(c))
        if 'freq_cutoff' not in overwritten_list and 'vocab_file_name' not in overwritten_list:
            sys.exit('rnnlab WARNING: Did not find "freq_cutoff" or "vocab_file_name" in user configs')
        if 'probes_name' not in overwritten_list: print 'rnnlab WARNING: "Did not find probes_name" in user_configs'
        ##########################################################################
        return configs_dict