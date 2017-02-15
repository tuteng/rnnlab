import os, sys
from operator import itemgetter
import numpy as np
import pandas as pd


class DataBase:
    """
    Stores data collected during rnn training

    Automatically created during rnn training
    Browser app instantiates this class for data analysis
    """

    def __init__(self, configs_dict, df, block_name=None):
        ##########################################################################
        # define dfpath
        runs_dir = load_rnnlabrc('runs_dir')
        self.dfpath = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame',
                                   'df_block_{}.h5'.format(block_name))
        self.trajdfpath = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame', 'trajdf.h5')
        ##########################################################################
        # assign instance variables
        self.configs_dict = configs_dict
        self.model_name = configs_dict['model_name']
        self.num_iterations = int(configs_dict['num_iterations'])
        self.block_name = block_name
        self.df = df
        ##########################################################################
        # load token data
        self.token_list = load_token_data(self.model_name, 'token_list')
        self.token_id_dict = load_token_data(self.model_name, 'token_id_dict')
        self.probe_list = load_token_data(self.model_name, 'probe_list')
        self.probe_cat_dict = load_token_data(self.model_name, 'probe_cat_dict')
        self.cat_list = load_token_data(self.model_name, 'cat_list')
        self.cat_probe_list_dict = load_token_data(self.model_name, 'cat_probe_list_dict')
        self.probe_id_dict = load_token_data(self.model_name, 'probe_id_dict')
        ##########################################################################
        # load corpus data
        self.probe_cf_traj_dict = load_corpus_data(self.model_name, 'probe_cf_traj_dict')
        self.num_input_units = load_corpus_data(self.model_name, 'num_input_units')

    def get_saved_block_names(self):
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            saved_block_names = store.select_column('trajdf', 'block_name').values
        ##########################################################################
        return saved_block_names

    def get_ba_breakdown_data(self):  # TODO this needs to be modified to retrieve non-averaged ba
        ##########################################################################
        # make df_cat_and_ba
        df_cat_ba = self.df[['cat', 'probe', 'token_ba']].drop_duplicates().groupby('cat', sort=False).mean()
        ##########################################################################
        # make cats_sorted_by_ba
        tuples = [tuple for tuple in df_cat_ba.itertuples()]
        tuples_sorted_by_ba = sorted(tuples, key=itemgetter(1))
        cats_sorted_by_ba = [tuple[0] for tuple in tuples_sorted_by_ba]
        ##########################################################################
        # make cat_ba_dict (this is not really needed)
        cat_ba_dict = df_cat_ba.to_dict()['token_ba']
        ##########################################################################
        # make token_ba_list
        token_ba_list = self.df[['probe','token_ba']].groupby('probe').first()['token_ba'].as_matrix()
        ##########################################################################
        return cats_sorted_by_ba, cat_ba_dict, token_ba_list

    def get_token_acts_df(self, probe_to_query):
        ##########################################################################
        probe_to_query = probe_to_query
        token_ids = self.df.query("probe == @probe_to_query").index.tolist()
        token_acts_df = self.df.loc[token_ids].filter(regex='H')
        ##########################################################################
        return token_acts_df

    def get_cat_acts_df(self, cat_to_query):
        ##########################################################################
        cat_to_query = cat_to_query
        df_cat = self.df.query("cat == @cat_to_query")
        cat_acts_df = df_cat.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        return cat_acts_df

    def get_all_acts_df(self, num_samples):
        ##########################################################################
        if num_samples is not None:
            ##########################################################################
            print 'Sampling {} probe activations to use in balanced accuracy...'.format(num_samples)
            ##########################################################################
            all_acts_labels, acts_mat_list = [], []
            for probe in self.probe_list:
                probe_acts_mat = self.df[self.df['probe'] == probe].sample(
                    num_samples, replace=True).filter(
                    regex='H').values  # TODO set replace to False, but prevent this from being a problem
                assert probe_acts_mat.shape == (num_samples, self.configs_dict['num_hidden_units'])
                acts_mat_list.append(probe_acts_mat)
                all_acts_labels += [probe] * len(probe_acts_mat)
            all_acts_mat = np.vstack((mat for mat in acts_mat_list))
            all_acts_df = pd.DataFrame(all_acts_mat)
        else:
            ##########################################################################
            print 'Collapsing all probe activtions to calculate balanced accuracy...'
            ##########################################################################
            all_acts_df = self.df.groupby('probe', sort=True).mean().filter(regex='H')
            all_acts_labels = self.probe_list
        ##########################################################################
        return all_acts_df, all_acts_labels

    def get_xaxis(self, omit_first=False):
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            saved_block_names = store.select_column('trajdf', 'index').values
        xaxis = map(lambda x: int(x) * self.num_iterations, [i for i in saved_block_names])
        if omit_first: xaxis = xaxis[1:]
        ##########################################################################
        return xaxis

    def save_to_disk(self, block_name, test_pp, avg_token_ba, token_ba_list):
        ##########################################################################
        # make entry
        new_entry = pd.DataFrame(index=[block_name], data={'test_pp': test_pp,
                                                           'avg_token_ba': avg_token_ba,
                                                           'block_name': block_name})
        for probe, token_ba in zip(self.probe_list, token_ba_list):
            new_entry[probe] = token_ba
        ##########################################################################
        # append entry to trajstore
        with pd.HDFStore(self.trajdfpath, complevel=9, complib='blosc', mode='a') as store:
            store.append('trajdf', new_entry,
                         data_columns=['test_pp', 'avg_token_ba'],
                         min_itemsize={'block_name': 10, 'index': 10})
        ##########################################################################
        # add data to non-traj database too
        token_ba_col = [token_ba_list[self.probe_list.index(probe)] for probe in self.df['probe']]
        self.df['token_ba'] = token_ba_col  # TODO do i need this in addition to index being block_name?
        ##########################################################################
        # save non-traj data
        with pd.HDFStore(self.dfpath, complevel=9, complib='blosc', mode='w', format='fixed') as store:
            store['df'] = self.df

    def get_avg_traj(self, which_traj):
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            avg_traj = store.select_column('trajdf', which_traj).values
        ##########################################################################
        return avg_traj

    def get_token_ba_traj_mat(self, probes_to_query):
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            probes_to_query = probes_to_query  # probes_list is used in query
            token_ba_traj_df = store.select('trajdf', where="columns in probes_to_query")
        token_ba_traj_mat = token_ba_traj_df.values.transpose()
        ##########################################################################
        return token_ba_traj_mat

    def fast_query(self):  # futureproofing: is this useful?
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            # any kind of query:
            query_value = store.select('trajdf', where="token_ba < 50")
        ##########################################################################
        return query_value

    def get_window_corr_data(self, window):
        ##########################################################################
        # load columns into series
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            avg_token_ba_traj = store.select_column('trajdf', 'avg_token_ba').values
            test_pp_traj = store.select_column('trajdf', 'test_pp').values
        s1 = pd.Series(avg_token_ba_traj)
        s2 = pd.Series(test_pp_traj)
        ##########################################################################
        #  window corr
        ba_pp_mw_corr = s1.rolling(window=window).corr(s2)
        ba_pp_ew_corr = s1.expanding().corr(s2)
        ##########################################################################
        return ba_pp_mw_corr, ba_pp_ew_corr

    def get_avg_token_ba_traj(self):
        ##########################################################################
        with pd.HDFStore(self.trajdfpath, mode='r') as store:
            avg_token_ba_traj = store.select_column('trajdf', 'avg_token_ba').values
        ##########################################################################
        return avg_token_ba_traj


def load_token_data(model_name, *args):
    ##########################################################################
    runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Token_Data')
    file_name = 'token_data.npz'.format(model_name)
    npzfile = np.load(os.path.join(path, file_name))
    ##########################################################################
    # load
    token_data = []
    for arg in args:
        var = npzfile[arg]
        if 'dict' in arg:
            var = var.item()
        elif 'list' in arg:
            var = var.tolist()
        token_data.append(var)
    ##########################################################################
    # unpack if one var requested
    if len(token_data) == 1: token_data = token_data[0]
    ##########################################################################
    return token_data


def load_corpus_data(model_name, *args):
    ##########################################################################
    runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Corpus_Data')
    file_name = 'corpus_data.npz'.format(model_name)
    npzfile = np.load(os.path.join(path, file_name))
    ##########################################################################
    # load
    corpus_data = []
    for arg in args:
        var = npzfile[arg]
        if 'dict' in arg: var = var.item()
        if 'list' in arg: var = var.tolist()
        if 'num' in arg: var = int(var)
        corpus_data.append(var)
    ##########################################################################
    # unpack if one var requested
    if len(corpus_data) == 1: corpus_data = corpus_data[0]
    ##########################################################################
    return corpus_data


def load_rnnlabrc(string):
    ##########################################################################
    # load rc from file
    rc = None
    with open(os.path.join(os.path.expanduser('~'), '.rnnlabrc'), 'r') as f:
        for line in f.readlines():
            if line.startswith(string):
                rc = line.split()[1]
    if rc is None:
        sys.exit('rnnlab: Did not find "{}" in .rnnlabrc'.format(rc))
    ##########################################################################
    return rc
