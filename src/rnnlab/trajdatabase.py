import os, sys, time
from operator import itemgetter
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from utilities import calc_ba_mats
from rnnhelper import load_rc
from utilities import calc_probe_sim_mat
import pandas as pd

pd.set_option('io.hdf.default_format','table')

class TrajDataBase:
    """
    Stores token_ba, test_pp and hca data across all training blocks.
    Storing these separately from main database allows faster retrieval of trajectory data

    Automatically created during rnn training
    Can also be instantiated outside of training for pos-hoc analysis
    """

    def __init__(self, configs_dict, num_train_blocks, complevel=9):
        ##########################################################################
        # define directories
        dev_path = os.path.join('rnnlab', 'data')
        if os.path.isdir(dev_path):
            self.data_dir = dev_path
        else:
            self.data_dir = os.path.join('data')
        self.runs_dir = load_rc('runs_dir')
        self.trajdfpath = os.path.join(self.runs_dir, configs_dict['model_name'], 'Data_Frame', 'trajdf.h5')
        ##########################################################################
        # assign instance variables
        self.model_name = configs_dict['model_name']
        self.token_list, self.token_id_dict, self.probe_list, \
        self.probe_id_dict, self.probe_cat_dict, self.cat_list = self.load_token_data()
        self.num_train_blocks = num_train_blocks
        ##########################################################################
        # open trajstore
        print 'Opening trajectory store with complevel {}'.format(complevel)
        self.trajstore = pd.HDFStore(self.trajdfpath, complevel=complevel, complib='blosc')
        # self.trajstore.create_table_index('block_names', columns=['block_name'])  # TODO this is slowing down writing


    def calc_new_entry(self, df, all_acts_df, test_pp, block_name, use_classifier=False):
        ##########################################################################
        print 'Making new entry to trajstore...'
        ##########################################################################
        # init df with index
        new_entry = pd.DataFrame(index=[block_name]) # table is empty without index
        ##########################################################################
        # make and add token_ba list to new entry
        token_ba_list = self.calc_token_ba_list(all_acts_df)
        avg_token_ba = np.mean(token_ba_list)
        for probe, token_ba in zip(self.probe_list, token_ba_list):
            new_entry[probe] = token_ba
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
        # add block_name col?
        new_entry['block_name'] = block_name
        ##########################################################################
        return new_entry, test_pp, avg_token_ba


    def append_entry(self, new_entry):
        ##########################################################################
        # append to trajstore
        self.trajstore.append('trajdf', new_entry)
                     # data_columns=False,  # TODO True takes too long?
                     # expected_rows=self.num_train_blocks, # should make read faster
                     # index=False) #TODO debugging
        ##########################################################################
        print 'Saved new entry to trajstore'
        print 'In store:'
        print self.trajstore
        ##########################################################################
        # close strajstore
        self.trajstore.close()


    def get_test_pp(self, block_name):
        ##########################################################################
        # get test_pp
        block_name = block_name
        test_pp = self.trajstore.select('trajdf', where="columns == test_pp & index == block_name").values.item(0)
        ##########################################################################
        return test_pp


    def get_token_ba_list(self, block_name):
        ##########################################################################
        # get token_ba_list
        query = self.trajstore.select('trajdf', where="columns in probe_list & index == block_name")
        token_ba_list = query.values.tolist()
        ##########################################################################
        return token_ba_list



    def get_avg_col_multiple_dfs(self, col_name):  # TODO is this useful?
        ##########################################################################
        # query all dfs stored in model's dir for a single colum and take mean
        path = os.path.join(self.runs_dir, self.model_name, 'Data_Frame')
        query_list = []
        series = self.trajstore.select_column('trajdf', col_name)  # only works if col was indexed during saving
        query_list.append(np.mean(series.values))
        ##########################################################################
        return query_list


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
        train_hca, test_hca = np.nan, np.nan
        ########################################################################################
        return train_hca, test_hca


    def calc_token_ba_list(self, all_acts_df): # uses multiprocessing
        ##########################################################################
        # calc simmat
        probe_simmat = calc_probe_sim_mat(all_acts_df, self.probe_list)
        ##########################################################################
        # make thr_ranges
        num_cpus, thr_start, thr_end, thr_step = 6, 0.7, 1.0, 0.01  # TODO how flexible is this?
        thr_num_steps = round(((thr_end - thr_start) / thr_step) / num_cpus, 2)
        thr_lists = []
        while True:
            thr_list = np.arange(thr_start, thr_start + thr_num_steps * thr_step, thr_step)
            thr_lists.append(thr_list)
            thr_start = float(thr_list[-1])
            time.sleep(0.5)
            if round(thr_end, 2) == round(thr_start, 2):  # TODO why do i have to round here?
                break
        ##########################################################################
        # calc ba
        if mp.cpu_count() < num_cpus: sys.exit(
            'rnnlab: CPU Count is < 6. Parallel calculation of token_ba may not work')
        else:
            print 'Calculating token ba using {} processes...'.format(num_cpus)
        pool = mp.Pool(processes=num_cpus)
        async_results = [pool.apply_async(calc_ba_mats, args=(self.probe_list,
                                                              self.cat_list,
                                                              self.probe_cat_dict,
                                                              probe_simmat,
                                                              thr_list,
                                                              'token')) for thr_list in thr_lists]
        results = [result.get() for result in async_results]
        token_ba_mat = np.hstack((mat for mat in results))
        pool.close()
        ##########################################################################
        # token_ba_mat = self.calc_ba_mats(probe_simmat, num_thrs, thrs, 'token')
        token_ba_mat_col_means = np.nanmean(token_ba_mat, 0) * 100
        best_token_ba_mat_col_id = np.argmax(token_ba_mat_col_means)
        token_ba_list = np.multiply(token_ba_mat[:, best_token_ba_mat_col_id], 100).tolist()
        ##########################################################################
        return token_ba_list


    def load_token_data(self):
        ##########################################################################
        path = os.path.join(self.runs_dir, self.model_name, 'Token_Data')
        file_name = 'token_data.npz'.format(self.model_name)
        npzfile = np.load(os.path.join(path, file_name))
        token_list, token_id_dict = npzfile['token_list'].tolist(), npzfile['token_id_dict'].item()
        probe_list, probe_id_dict = npzfile['probe_list'].tolist(), npzfile['probe_id_dict'].item()
        probe_cat_dict = npzfile['probe_cat_dict'].item()
        cat_list = npzfile['cat_list'].tolist()
        ##########################################################################
        return token_list, token_id_dict, probe_list, probe_id_dict, probe_cat_dict, cat_list




    def make_token_ba_traj_mat(self, sel_probes):
        ##########################################################################
        # query all dfs stored in model's dir
        start = time.time()
        path = os.path.join(self.runs_dir, self.model_name, 'Data_Frame')
        sel_probes = sel_probes # probes_list is used in query
        token_ba_traj_df = self.trajstore.select('trajdf', where="columns in sel_probes")
        token_ba_traj_mat = token_ba_traj_df.values
        print token_ba_traj_mat.shape
        print 'time:', abs(time.time() - start)
        ##########################################################################
        return token_ba_traj_mat



    def fast_query(self, block_name): # futureproofing: is this useful?
        ##########################################################################
        path = os.path.join(self.runs_dir, self.model_name, 'Data_Frame')
        file_name = 'df_block_{}.h5'.format(block_name)
        with pd.HDFStore(os.path.join(path, file_name), mode='r') as store:
            # any kind of query:
            query_value = store.select('trajdf', where="token_ba < 50")
        ##########################################################################
        return query_value




    def make_token_ba_trajectories_fig(self, sel_probes, cat, is_title=False): # for probes in cat
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        linewidth = 2.0
        # fig
        ##########################################################################
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Token Balanced Accuracies for probes in {}'.format(self.model_name, cat)
        if is_title: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Training Blocks', fontsize=ax_font_size)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
        ##########################################################################
        # Hide the right and top spines and ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ##########################################################################
        # get token_ba trajectories and plot
        token_ba_traj_mat = self.make_token_ba_traj_mat(sel_probes)
        for token_ba_traj, probe in zip(token_ba_traj_mat, sel_probes):
            num_x = len(token_ba_traj)
            ax.plot(range(num_x), token_ba_traj, '-', linewidth=linewidth, label=probe) # TODO x axis labels are block 1,2,3,4..
        ##########################################################################
        # legend
        ax.set_position([0.1, 0.1, 0.5, 0.8])
        ax.legend(fontsize=leg_font_size, loc='center right', bbox_to_anchor=(1.1, 0.5)) # TODO instead of legend make annotations

        plt.tight_layout()
        ##########################################################################
        return fig


    def make_pp_curve_fig(self, smoothing_span=20):
        ##########################################################################
        # pp_curve = ?
        if smoothing_span > 1:
            from pandas.stats.moments import ewma
            pp_curve = ewma(np.asarray(pp_curve), span=smoothing_span)
        else:
            pp_curve = pp_curve
        ##########################################################################
        # TODO finish this









