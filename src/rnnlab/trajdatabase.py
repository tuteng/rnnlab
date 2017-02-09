import os, time
import numpy as np
import multiprocessing as mp
import pandas as pd



from dbutils import calc_ba_mats
from dbutils import load_corpus_data
from dbutils import load_token_data
from dbutils import load_rnnlabrc
from dbutils import calc_probe_sim_mat



class TrajDataBase:
    """
    Stores token_ba, test_pp and hca data across all training blocks.
    Storing these separately from main database allows faster retrieval of trajectory data

    Automatically created during rnn training
    Browser app instantiates this class for data analysis
    """

    def __init__(self, configs_dict, mode='r', complevel=9):
        ##########################################################################
        # define trajdfpath
        runs_dir = load_rnnlabrc('runs_dir')
        self.trajdfpath = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame', 'trajdf.h5')
        ##########################################################################
        # assign instance variables
        self.model_name = configs_dict['model_name']
        self.num_iterations = int(configs_dict['num_iterations'])
        ##########################################################################
        # load token & corpus data
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict, \
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict = load_token_data(self.model_name)
        self.probe_cf_traj_dict, self.num_train_doc_ids, \
        self.tf_idf_mat, self.lex_div_traj, self.num_input_units = load_corpus_data(self.model_name)
        ##########################################################################
        # open trajstore
        self.trajstore = pd.HDFStore(self.trajdfpath, complevel=complevel, complib='blosc', mode=mode)



    def make_xaxis(self, omit_first=False):
        ##########################################################################
        saved_block_names = self.trajstore.select_column('trajdf', 'index').values
        xaxis = map(lambda x: int(x) * self.num_iterations, [i for i in saved_block_names])
        if omit_first: xaxis = xaxis[1:]
        ##########################################################################
        return xaxis


    def calc_new_entry(self, df, all_acts_df, test_pp, block_name, use_classifier=False):
        ##########################################################################
        # init df with index
        new_entry = pd.DataFrame(index=[block_name]) # table is empty without index
        ##########################################################################
        # make and add token_ba list to new entry
        token_ba_list = self.calc_token_ba_list(all_acts_df, block_name)
        avg_token_ba = np.mean(token_ba_list)
        for probe, token_ba in zip(self.probe_list, token_ba_list):
            new_entry[probe] = token_ba
        new_entry['avg_token_ba'] = avg_token_ba
        ##########################################################################
        # add test_pp to new entry
        new_entry['test_pp'] = test_pp
        ##########################################################################
        # calc and add hca train and test values to new entry
        if use_classifier:
            acts_cols = df.filter(regex='H').values
            cat_col = df['cat'].values
            train_hca, test_hca = self.calc_hca(acts_cols, cat_col)
        else: train_hca, test_hca = np.nan, np.nan
        new_entry['train_hca'], new_entry['test_hca'] = train_hca, test_hca
        ##########################################################################
        # add block_name
        new_entry['block_name'] = block_name
        ##########################################################################
        return new_entry, test_pp, avg_token_ba, token_ba_list


    def append_entry(self, new_entry):
        ##########################################################################
        # append to trajstore
        self.trajstore.append('trajdf', new_entry,
                              data_columns=['test_pp', 'train_hca', 'test_hca', 'avg_token_ba'],
                              min_itemsize={'block_name': 10, 'index' : 10})
        ##########################################################################
        self.trajstore.close()


    def calc_hca(self, acts_cols, cat_col, epochs=30):
        ########################################################################################
        print 'Calculating classifier accuracy...'
        ########################################################################################
        # make data for classifier
        x_data = acts_cols
        y_data = np.zeros(len(cat_col))
        for n, cat in enumerate(cat_col): y_data[n] = self.cat_list.index(cat)
        assert len(x_data) == len(y_data)
        ########################################################################################
        # classifier # TODO make sure classifier works
        from classifier import calc_hca
        train_hca, test_hca = calc_hca(self.model_name, x_data, y_data, epochs)
        ########################################################################################
        return train_hca, test_hca


    def calc_token_ba_list(self, all_acts_df, block_name, thr_step=0.001, verbose=False):
        ##########################################################################
        # calc simmat
        probe_simmat = calc_probe_sim_mat(all_acts_df, self.probe_list)
        ##########################################################################
        print 'Calculating balanced accuracy...'
        ##########################################################################
        # make thr_ranges
        num_cpus = 6
        thr_start, thr_end = 0.7, 1.0  # TODO how flexible is this?
        thr_num_steps = round(((thr_end - thr_start) / thr_step) / num_cpus, 2)
        thr_lists = []
        while True:
            thr_list = np.arange(thr_start, thr_start + thr_num_steps * thr_step, thr_step)
            thr_lists.append(thr_list)
            thr_start = float(thr_list[-1])
            time.sleep(0.5)
            if round(thr_end, 2) == round(thr_start, 2):  # TODO can i use numpy equal approximation?
                break
        ##########################################################################
        if mp.cpu_count() < num_cpus:
            print 'rnnlab WARNING: CPU Count is < 6. Parallel calculation of token_ba may not work'
        elif verbose:
            print 'Calculating token ba using {} processes with {} steps...'.format(num_cpus, thr_step)
            print 'Threshold Ranges:'
            for i in thr_lists: print i, '\n'
        ##########################################################################
        # calc ba and get cat confusion mat data
        start = time.time()
        pool = mp.Pool(processes=num_cpus)
        async_results = [pool.apply_async(calc_ba_mats, args=(self.probe_list,
                                                              self.cat_list,
                                                              self.probe_cat_dict,
                                                              probe_simmat,
                                                              thr_list,
                                                              'token')) for thr_list in thr_lists]
        ba_mats = [result.get()[0] for result in async_results]
        ba_mat = np.hstack((mat for mat in ba_mats))
        cat_confusion_mat_data_list_of_lists = [result.get()[1] for result in async_results]
        cat_confusion_mat_data_list = [item for sublist in cat_confusion_mat_data_list_of_lists for item in sublist]
        pool.close()
        ##########################################################################
        print 'Took {} mins to calc ba'.format(abs(time.time() - start)/60.)
        ##########################################################################
        # make token_ba_list
        token_ba_mat_col_means = np.nanmean(ba_mat, 0) * 100
        best_token_ba_mat_col_id = np.argmax(token_ba_mat_col_means)
        token_ba_list = np.multiply(ba_mat[:, best_token_ba_mat_col_id], 100).tolist()
        ##########################################################################
        # save confusion data
        cat_confusion_mat_data = cat_confusion_mat_data_list[best_token_ba_mat_col_id]
        runs_dir = load_rnnlabrc('runs_dir')
        path = os.path.join(runs_dir, self.model_name, 'Balanced_Accuracy')
        file_name = 'cat_confusion_mat_data_block_{}.npz'.format(block_name)
        np.savez(os.path.join(path, file_name),
                 hits_by_cat_dict=cat_confusion_mat_data[0],
                 fas_by_cat_dict=cat_confusion_mat_data[1])
        ##########################################################################
        return token_ba_list



    def make_token_ba_traj_mat(self, sel_probes):
        ##########################################################################
        # query all dfs stored in model's dir
        sel_probes = sel_probes # probes_list is used in query
        token_ba_traj_df = self.trajstore.select('trajdf', where="columns in sel_probes")
        token_ba_traj_mat = token_ba_traj_df.values.transpose()
        ##########################################################################
        return token_ba_traj_mat



    def fast_query(self, block_name): # futureproofing: is this useful?
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            # any kind of query:
            query_value = store.select('trajdf', where="token_ba < 50")
        ##########################################################################
        return query_value




    def make_ba_pp_window_corr_data(self, window):
        ##########################################################################
        # load columns into series
        avg_token_ba_traj = self.trajstore.select_column('trajdf', 'avg_token_ba').values
        s1 = pd.Series(avg_token_ba_traj)
        test_pp_traj = self.trajstore.select_column('trajdf', 'test_pp').values
        s2 = pd.Series(test_pp_traj)
        ##########################################################################
        #  window corr
        ba_pp_mw_corr = s1.rolling(window=window).corr(s2)
        ba_pp_ew_corr = s1.expanding().corr(s2)
        ##########################################################################
        return ba_pp_mw_corr, ba_pp_ew_corr