import os, time, shutil, socket, csv, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import groupby
from operator import itemgetter
from database import DataBase
from trajdatabase import TrajDataBase
from rnnhelper import RNNHelper


class RNN(RNNHelper):
    """
    Creates and trains a single model and saves training data to pandas data frame
    """
    def __init__(self, flavor, user_configs):
        ##########################################################################
        # inherit super class variables
        super(RNN, self).__init__()
        ##########################################################################
        # define directories
        self.log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
        self.user_configs_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_user_configs.csv'))
        ##########################################################################
        # create rnn with superclass method
        self.rnn = super(RNN, self).create_rnn(user_configs, flavor)
        self.rnn.model_name = self.rnn.configs_dict['model_name']



    def train(self, df_blocks=None):
        ##########################################################################
        self.prepare_training()
        ##########################################################################
        # inits
        elapsed_start = time.time()
        num_mbs_trained, total_num_examples_seen, completed = 0, 0, False
        stop_block_int = self.rnn.corpus.num_total_docs * self.rnn.num_epochs
        stop_block_name = self.rnn.corpus.to_block_name(self.rnn.corpus.num_total_docs * self.rnn.num_epochs)
        avg_token_ba_list = []
        ##########################################################################
        # format df_blocks
        # if df_blocks specified, data will only be extracted for those blocks and if they overlap with save_ev
        if df_blocks is None: df_blocks = [self.rnn.corpus.to_block_name(i) for i in range(stop_block_int)]
        ##########################################################################
        # block
        for block_name, block_id in self.rnn.corpus.gen_train_block_name_and_id(
                    epochs=self.rnn.num_epochs, shuffle=self.rnn.randomize_blocks):
            self.print_train_stats(elapsed_start, block_name, block_id, num_mbs_trained)
            ##########################################################################
            # iteration
            if int(block_name) != 1:  # enables saving of data prior to any training
                for iteration_counter in xrange(self.rnn.num_iterations):
                    ##########################################################################
                    # batch
                    for (X, Y) in self.rnn.corpus.gen_batch(self.rnn.mb_size, self.rnn.bptt_steps, block_id):
                        num_mbs_trained += 1
                        total_num_examples_seen += self.rnn.mb_size
                        self.rnn.sess.run(self.rnn.train_step, feed_dict={self.rnn.x: X, self.rnn.y: Y})
            else:
                print 'Skipped training first block. Proceeding with data extraction from untrained model...'
            ##########################################################################
            # after each save_ev block
            is_make_df = int(block_name) % self.rnn.save_ev == 0 and block_name in df_blocks
            if is_make_df or int(block_name) in[1, stop_block_int]:
                ##########################################################################
                start = time.time()
                ##########################################################################
                # save checkpoint
                self.save_ckpt(block_name)
                ##########################################################################
                # make database and save
                df = self.make_df()
                database = DataBase(self.rnn.configs_dict, df, block_name)
                database.save_df()
                ##########################################################################
                # make trajectory data and append to trajdatabase (token_ba, test_pp and hca data)
                trajdatabase = TrajDataBase(self.rnn.configs_dict, self.rnn.corpus.num_train_docs)
                test_pp = self.calc_test_pp()
                new_entry, test_pp, avg_token_ba = trajdatabase.calc_new_entry(df, database.all_acts_df, test_pp, block_name)
                trajdatabase.append_entry(new_entry)
                #########################################################################
                # print to console
                print 'Test perplexity : {} |Avg token ba : {} |Database ops completed in {} secs'.format(
                    test_pp, avg_token_ba, int(abs(time.time() - start)))
                avg_token_ba_list.append(avg_token_ba)
                #########################################################################
                # update log with best_avg_token_ba and completed
                best_avg_token_ba = np.max(np.asarray(avg_token_ba_list))
                completed = True if stop_block_name == block_name else False
                self.update_log(best_avg_token_ba, completed)
        ##########################################################################
        # at end of training, close session and upload data
        self.complete_training()


    def save_ckpt(self, block_name):
        ##########################################################################
        ckpt_outfile = "weights_at_block_{}.ckpt".format(block_name)
        path = os.path.join(self.runs_dir, self.rnn.model_name, 'Weights')
        self.rnn.saver.save(self.rnn.sess, os.path.join(path, ckpt_outfile))


    def prepare_training(self):
        ##########################################################################
        # remove low priority data from previous rnns
        self.remove_old_data()
        ##########################################################################
        # make log entry
        self.make_log_entry()
        ##########################################################################
        # create data dirs
        for dir in ['Configs', 'Weights', 'Balanced_Accuracy', 'Data_Frame',
                    'Sim_Mat', 'Classifier', 'Figures']:
            path = os.path.join(self.runs_dir, self.rnn.model_name, dir)
            if not os.path.isdir(path):
                os.makedirs(path)
        ##########################################################################
        #  save token data
        path = os.path.join(self.runs_dir, self.rnn.model_name, 'Token_Data')
        file_name = 'token_data.npz'.format(self.rnn.model_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        np.savez(os.path.join(path, file_name),
                 token_list=self.rnn.corpus.token_list,
                 token_id_dict=self.rnn.corpus.token_id_dict,
                 probe_id_dict=self.rnn.corpus.probe_id_dict,  # this dict is relative to num_probes
                 probe_list=self.rnn.corpus.probe_list,
                 probe_cat_dict=self.rnn.corpus.probe_cat_dict,
                 cat_list=self.rnn.corpus.cat_list)
        ##########################################################################
        #  save configs_dict to npy
        path = os.path.join(self.runs_dir, self.rnn.model_name, 'Configs')
        file_name = 'configs_dict.npy'
        np.save(os.path.join(path, file_name), self.rnn.configs_dict)


    def remove_old_data(self, num_more_recent=5):
        ##########################################################################
        del_candidates_list = []
        models_run_on_this_machine = []
        model_names_deleted = []
        ##########################################################################
        # get runs log
        if os.path.isfile(self.log_path):
            log_content = csv.reader(open(self.log_path, 'r'))
            ##########################################################################
            # make del_candidates_list
            if log_content:
                for row in log_content:
                    if row[0].startswith(socket.gethostname()):
                        model_name, best_token_ba = row[0], row[-1]
                        flavor = model_name.split('_')[-1]
                        del_candidates_list.append((model_name, best_token_ba, flavor))
                        models_run_on_this_machine.append(model_name)
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
        else:
            sys.exit('rnnlab: Could not find {}.'.format(self.log_path))
        ##########################################################################
        # remove log entries corresponding with models deleted above if not completed (completed data is still informative)
        log_content = csv.reader(open(self.log_path, 'r'))
        runs_log_content_new = []
        for row in log_content:
            if not 'model_name' in row:
                completed = int(row[-2])
                if completed == 1 or row[0] not in models_run_on_this_machine or row[0] not in model_names_deleted:
                    runs_log_content_new.append(row)
            elif 'model_name' in row:
                runs_log_content_new.append(row)
        time.sleep(1)
        with open(self.log_path, 'w') as f:
            writer = csv.writer(f)
            for row in runs_log_content_new:
                writer.writerow(row)


    def make_log_entry(self):
        ##########################################################################
        # append model configs to log
        configs_dict = self.rnn.configs_dict.copy()
        for entry_to_pop in ['flavor', 'model_name', 'corpus_name']:
            configs_dict.pop(entry_to_pop)
        writer = csv.writer(open(self.log_path, 'a'))
        if not os.path.isfile(self.log_path):
            header = [str(key) for key in configs_dict.keys()]
            header.insert(0, 'model_name')
            header.append('completed')
            header.append('best_token_ba')
            writer.writerow(header)
            print "Writing header to log"
        all_params_list = [str(configs_dict[key]) for key in configs_dict.keys()]
        all_params_list.insert(0, self.rnn.model_name)
        all_params_list.append('0')  # completed
        all_params_list.append('0')  # best_token_ba
        writer.writerow(all_params_list)


    def update_log(self, best_avg_token_ba, completed):
        ##########################################################################
        log_content = csv.reader(open(self.log_path, 'r'))
        log_content_new = []
        for row in log_content:
            if row[0] == self.rnn.model_name:
                row[-1] = format(best_avg_token_ba, '.3f')
                row[-2] = int(completed)
            log_content_new.append(row)
        os.remove(self.log_path)  # remove log to reduce syncing issues
        with open(self.log_path, 'w') as f:
            writer = csv.writer(f)
            for row in log_content_new:
                writer.writerow(row)


    def print_train_stats(self, elapsed_start, block_name, block_id, num_mbs_trained):
        ##########################################################################
        secs = int(abs(elapsed_start - time.time()))
        hours = int(float(secs) / 3600)
        max_num_train_blocks = len(self.rnn.corpus.train_doc_ids) * self.rnn.num_epochs
        print '{} |Block Name: {}/{} Id: {:>4} |Batch: {:>10} |Elapsed: {:>2} hrs'.format(
            self.rnn.model_name, block_name, max_num_train_blocks, block_id, num_mbs_trained, hours)


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
            path = os.path.join(self.runs_dir, self.rnn.model_name, 'Data_Frame')
            for file_name in sorted(os.listdir(path))[:-1]:
                os.remove(os.path.join(path, file_name))
        ##########################################################################
        self.rnn.sess.close()
        tf.reset_default_graph()
        print '{} Training Session Closed and Graph reset\n\n'.format(self.rnn.model_name)


    def make_df(self, fast=True):
        ##########################################################################
        print 'Calculating hidden activations...'
        ##########################################################################
        # inits
        tokens_in_mb = []
        hidden_unit_labels = ['H{}'.format(i) for i in range(self.rnn.num_hidden_units)]
        df_column_labels = ['doc_name', 'probe', 'probe_id', 'Y', 'cat', 'token_pp'] + hidden_unit_labels
        mb_data_dict_keys = ['X'] + df_column_labels
        mb_data_dict = {key: [] for key in mb_data_dict_keys}
        df = pd.DataFrame(columns=df_column_labels)
        probe_X = np.zeros((self.rnn.mb_size, self.rnn.bptt_steps), dtype=int)
        probe_Y = np.zeros(self.rnn.mb_size, dtype=int)
        num_tokens_in_batch = 0
        probe_freq_dict = {probe: 0 for probe in self.rnn.corpus.probe_list}
        ##########################################################################
        # get mb_data_dict from each block_name and add to df
        for block_name, block_id in self.rnn.corpus.gen_train_block_name_and_id(epochs=1, shuffle=False):
            for (X, Y) in self.rnn.corpus.gen_batch(self.rnn.mb_size, self.rnn.bptt_steps, block_id):
                tokens = [self.rnn.corpus.token_list[token_id] for token_id in X[:, -1]]
                for n, token in enumerate(tokens):
                    ##########################################################################
                    # if token is probe, add data to mb_data_dict
                    if token in self.rnn.corpus.probe_id_dict and token not in tokens_in_mb:
                        if fast:
                            tokens_in_mb.append(token)
                        mb_data_dict['X'].append(X[n])
                        mb_data_dict['doc_name'].append(int(block_name))
                        mb_data_dict['probe'].append(token)
                        mb_data_dict['probe_id'].append(self.rnn.corpus.probe_id_dict[token])
                        mb_data_dict['Y'].append(Y[n])
                        mb_data_dict['cat'].append(self.rnn.corpus.probe_cat_dict[token])
                        ##########################################################################
                        # build batch
                        probe_X[num_tokens_in_batch] = X[n]
                        probe_Y[num_tokens_in_batch] = Y[n]
                        num_tokens_in_batch += 1
                        probe_freq_dict[token] += 1.0
                    ##########################################################################
                    # when batch ready, calculate acts_mat and pp_vec
                    if num_tokens_in_batch == self.rnn.mb_size:
                        [acts_mat, pp_vec] = self.rnn.sess.run(
                            [self.rnn.last_hidden_state, self.rnn.pp_mat],
                            feed_dict={self.rnn.x: probe_X, self.rnn.y: probe_Y})
                        ##########################################################################
                        # add acts_mat and pp_vec to mb_data_dict
                        for n, hidden_unit_label in enumerate(hidden_unit_labels):
                            mb_data_dict[hidden_unit_label] += acts_mat.T[n].tolist()
                        mb_data_dict['token_pp'] += pp_vec.tolist() # TODO was previously named 'pp' only
                        ##########################################################################
                        # reset batch
                        tokens_in_mb = []
                        num_tokens_in_batch = 0
                        probe_X = np.zeros((self.rnn.mb_size, self.rnn.bptt_steps), dtype=int)
                        probe_Y = np.zeros(self.rnn.mb_size, dtype=int)
        ##########################################################################
        # convert mb_data_dict to df
        num_data = len(mb_data_dict['token_pp'])  # discard leftover tokens for which no acts were calculated (because of mb)
        for df_column_label in df_column_labels:
            df[df_column_label] = mb_data_dict[df_column_label][:num_data]
        ##########################################################################
        return df



    def calc_test_pp(self):
        ##########################################################################
        print 'Calculating test perplexity...'
        ##########################################################################
        test_pp_sum, num_batches, test_pp = 0, 0, 0
        for (X, Y) in self.rnn.corpus.gen_batch(self.rnn.mb_size, self.rnn.bptt_steps, 'test'):
            test_pp_batch = self.rnn.sess.run(self.rnn.mean_pp, feed_dict={self.rnn.x: X, self.rnn.y: Y})
            test_pp_sum += test_pp_batch
            num_batches += 1
        test_pp = int(test_pp_sum / num_batches)
        ##########################################################################
        return test_pp
