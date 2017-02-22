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
        self.maindf_path = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame',
                                   'df_block_{}.h5'.format(block_name))
        self.ba_trajdf_path = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame', 'ba_trajdf.h5')
        self.pp_trajdf_path = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame', 'pp_trajdf.h5')
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
        self.tf_idf_mat = load_corpus_data(self.model_name, 'tf_idf_mat')
        self.lex_div_traj = load_corpus_data(self.model_name, 'lex_div_traj')
        self.num_input_units = load_corpus_data(self.model_name, 'num_input_units')


    def save_to_disk(self, block_name, test_pp, probes_pp, avg_probe_pp_list, probes_ba, avg_probe_ba_list):
        ##########################################################################
        # ad to pp database
        ba_trajdf_entry = pd.DataFrame(index=[block_name],
                                       data={'probes_ba': probes_ba,
                                             'block_name': block_name})
        for probe, avg_probe_ba in zip(self.probe_list, avg_probe_ba_list):
            ba_trajdf_entry[probe] = avg_probe_ba
        with pd.HDFStore(self.ba_trajdf_path, complevel=9, complib='blosc', mode='a') as store:
            store.append('trajdf', ba_trajdf_entry,
                         min_itemsize={'block_name': 10, 'index': 10},
                         data_columns=['probes_ba'])
        ##########################################################################
        # add to pp database
        pp_trajdf_entry = pd.DataFrame(index=[block_name],
                                       data={'probes_pp': probes_pp,
                                             'block_name': block_name,
                                             'test_pp': test_pp})
        for probe, avg_probe_pp in zip(self.probe_list, avg_probe_pp_list):
            pp_trajdf_entry[probe] = avg_probe_pp
        with pd.HDFStore(self.pp_trajdf_path, complevel=9, complib='blosc', mode='a') as store:
            store.append('trajdf', pp_trajdf_entry,
                         min_itemsize={'block_name': 10, 'index': 10},
                         data_columns=['test_pp', 'probes_pp'])
        ##########################################################################
        # add to main database
        self.df['avg_probe_ba'] = [avg_probe_ba_list[self.probe_list.index(probe)] for probe in self.df['probe']]
        self.df['avg_probe_pp'] = [avg_probe_pp_list[self.probe_list.index(probe)] for probe in self.df['probe']]
        with pd.HDFStore(self.maindf_path, complevel=9, complib='blosc', mode='w', format='fixed') as store:
            store['df'] = self.df


    def get_saved_block_names(self):
        ##########################################################################
        with pd.HDFStore(self.ba_trajdf_path, mode='r') as store:
            saved_block_names = store.select_column('trajdf', 'index').values
        ##########################################################################
        return saved_block_names


    def get_ba_breakdown_data(self):  # TODO split this into separate methods each retrieving a single result
        ##########################################################################
        # make df_cat_and_ba
        df_cat_ba = self.df[['cat', 'probe', 'avg_probe_ba']].drop_duplicates().groupby('cat', sort=False).mean()
        ##########################################################################
        # make cats_sorted_by_ba
        tuples = [tuple for tuple in df_cat_ba.itertuples()]
        tuples_sorted_by_ba = sorted(tuples, key=itemgetter(1))
        cats_sorted_by_ba = [tuple[0] for tuple in tuples_sorted_by_ba]
        ##########################################################################
        # make cat_ba_dict (this is not really needed)
        cat_ba_dict = df_cat_ba.to_dict()['avg_probe_ba']
        ##########################################################################
        # make avg_probe_ba_list
        avg_probe_ba_list = self.get_avg_probe_ba_list()
        ##########################################################################
        return cats_sorted_by_ba, cat_ba_dict, avg_probe_ba_list

    def get_avg_probe_pp_list(self, clip=False):
        ##########################################################################
        avg_probe_pp_list = self.df[['probe', 'avg_probe_pp']].groupby(
            'probe').first()['avg_probe_pp'].values.tolist()
        if clip: avg_probe_pp_list = np.clip(avg_probe_pp_list, 1, self.num_input_units)
        ##########################################################################
        return avg_probe_pp_list


    def get_avg_probe_ba_list(self):
        ##########################################################################
        avg_probe_ba_list = self.df[['probe', 'avg_probe_ba']].groupby(
            'probe').first()['avg_probe_ba'].values.tolist()
        ##########################################################################
        return avg_probe_ba_list


    def get_avg_probe_pp(self, probe_to_query, clip=True):
        ##########################################################################
        probe_to_query = probe_to_query
        row_ids = self.df.query("probe == @probe_to_query").index.tolist()
        avg_probe_pp = self.df.loc[row_ids]['probe_pp'].mean()
        if clip: avg_probe_pp = np.clip(avg_probe_pp, 1, self.num_input_units)
        ##########################################################################
        return avg_probe_pp


    def get_avg_probe_ba(self, probe_to_query):
        ##########################################################################
        probe_to_query = probe_to_query
        row_ids = self.df.query("probe == @probe_to_query").index.tolist()
        avg_probe_ba = self.df.loc[row_ids]['probe_ba'].mean()
        ##########################################################################
        return avg_probe_ba

    def get_probe_bas(self, probe_to_query):
        ##########################################################################
        probe_to_query = probe_to_query
        row_ids = self.df.query("probe == @probe_to_query").dropna().index.tolist()
        probe_bas = self.df.loc[row_ids]['probe_ba'].values.tolist()
        ##########################################################################
        return probe_bas

    def get_probe_pps(self, probe_to_query, clip=True):
        ##########################################################################
        probe_to_query = probe_to_query
        row_ids = self.df.query("probe == @probe_to_query").index.tolist()
        probe_pps = self.df.loc[row_ids]['probe_pp'].values.tolist()
        if clip: probe_pps = np.clip(probe_pps, 1, self.num_input_units)
        ##########################################################################
        return probe_pps


    def get_traj(self, which_traj):
        ##########################################################################
        if which_traj == 'test_pp':
            with pd.HDFStore(self.pp_trajdf_path, mode='r') as store:
                avg_traj = store.select_column('trajdf', 'test_pp').values
        elif which_traj == 'probes_pp':
            with pd.HDFStore(self.pp_trajdf_path, mode='r') as store:
                avg_traj = store.select_column('trajdf', 'probes_pp').values
        elif which_traj == 'probes_ba':
            with pd.HDFStore(self.ba_trajdf_path, mode='r') as store:
                avg_traj = store.select_column('trajdf', 'probes_ba').values
        else:
            raise NotImplementedError
        ##########################################################################
        return avg_traj


    def get_trajs_mat(self, probes_to_query, which_traj):
        ##########################################################################
        if which_traj == 'avg_probe_pp':
            with pd.HDFStore(self.pp_trajdf_path, mode='r') as store:
                probes_to_query = probes_to_query  # probes_list is used in query
                df_traj = store.select('trajdf', where="columns in probes_to_query")
                avg_trajs_mat = df_traj.values.transpose()
        elif which_traj == 'avg_probe_ba':
            with pd.HDFStore(self.ba_trajdf_path, mode='r') as store:
                probes_to_query = probes_to_query  # probes_list is used in query
                df_traj = store.select('trajdf', where="columns in probes_to_query")
                avg_trajs_mat = df_traj.values.transpose()
        else:
            raise NotImplementedError
        ##########################################################################
        return avg_trajs_mat

    def get_probe_acts_df(self, probe_to_query):
        ##########################################################################
        probe_to_query = probe_to_query
        df_ids = self.df.query("probe == @probe_to_query").index.tolist()
        probe_acts_df = self.df.loc[df_ids].filter(regex='H')
        ##########################################################################
        return probe_acts_df


    def get_cat_acts_df(self, cat_to_query):
        ##########################################################################
        cat_to_query = cat_to_query
        df_cat = self.df.query("cat == @cat_to_query")
        cat_acts_df = df_cat.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        return cat_acts_df

    def get_cat_prototypes_df(self, method='weighted'):
        ##########################################################################
        if method == 'weighted':
            cat_prototypes_df = self.df.groupby('cat', sort=True).mean().filter(regex='H')
        elif method == 'unweighted':
            tmp_df = self.df.groupby('probe', sort=True).mean().reset_index()
            tmp_df['cat'] = tmp_df['probe'].apply(lambda probe: self.probe_cat_dict[probe])
            cat_prototypes_df = tmp_df.groupby('cat').mean().filter(regex='H')
        else:
            raise AttributeError('rnnlab: method argument is either "weighted" or "unweighted".')
        ##########################################################################
        return cat_prototypes_df

    def get_probes_acts_df(self, num_ba_samples=0):
        ##########################################################################
        if num_ba_samples != 0:
            ##########################################################################
            print 'Using first {} probe activations to calc balanced accuracy...'.format(num_ba_samples)
            ##########################################################################
            # make probes_acts_df
            sampled_probes_list, acts_mat_list = [], []
            for probe in self.probe_list:
                probe_acts_candidates_df = self.df[self.df['probe'] == probe]
                num_probe_acts = len(probe_acts_candidates_df)
                if num_probe_acts > num_ba_samples:
                    probe_acts_mat = probe_acts_candidates_df.head(num_ba_samples).filter(regex='H').values
                else:
                    probe_acts_mat = probe_acts_candidates_df.head(num_probe_acts).filter(regex='H').values
                acts_mat_list.append(probe_acts_mat)
                sampled_probes_list += [probe] * len(probe_acts_mat)
            all_acts_mat = np.vstack((mat for mat in acts_mat_list))
            probes_acts_df = pd.DataFrame(data=all_acts_mat, index=sampled_probes_list)
        else:
            ##########################################################################
            print 'Collapsing all probe activtions to calculate balanced accuracy...'
            ##########################################################################
            probes_acts_df = self.df.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        return probes_acts_df

    def get_train_iterations_axis(self):
        ##########################################################################
        with pd.HDFStore(self.ba_trajdf_path, mode='r') as store:
            saved_block_names = store.select_column('trajdf', 'index').values
        xaxis = map(lambda x: int(x) * self.num_iterations, [i for i in saved_block_names])
        ##########################################################################
        return xaxis


    def get_window_corr_data(self, window):
        ##########################################################################
        # load columns into series
        with pd.HDFStore(self.ba_trajdf_path, mode='r') as store:
            probes_ba_traj = store.select_column('trajdf', 'probes_ba').values
        with pd.HDFStore(self.pp_trajdf_path, mode='r') as store:
            test_pp_traj = store.select_column('trajdf', 'test_pp').values
        s1 = pd.Series(probes_ba_traj)
        s2 = pd.Series(test_pp_traj)
        ##########################################################################
        #  window corr
        ba_pp_mw_corr = s1.rolling(window=window).corr(s2)
        ba_pp_ew_corr = s1.expanding().corr(s2)
        ##########################################################################
        return ba_pp_mw_corr, ba_pp_ew_corr


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
        raise AttributeError('rnnlab: Did not find "{}" in .rnnlabrc'.format(string))

    if rc == 'None': rc = None
    elif rc.isdigit():
        rc = int(rc)
    ##########################################################################
    return rc
