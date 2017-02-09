import os
from operator import itemgetter
import numpy as np
import pandas as pd


from dbutils import calc_probe_sim_mat
from dbutils import load_token_data
from dbutils import load_corpus_data
from dbutils import load_rnnlabrc



class DataBase:
    """
    Stores dataframe constructed during rnn training

    Automatically created during rnn training
    Browser app instantiates this class for data analysis
    """

    def __init__(self, configs_dict, df, block_name):
        ##########################################################################
        # define dfpath
        runs_dir = load_rnnlabrc('runs_dir')
        self.dfpath = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame',
                                   'df_block_{}.h5'.format(block_name))
        ##########################################################################
        # assign instance variables
        self.model_name = configs_dict['model_name']
        self.block_name = block_name
        self.df = df
        ##########################################################################
        # load token & corpus data
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict, \
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict = load_token_data(self.model_name)
        self.probe_cf_traj_dict, self.num_train_doc_ids, \
        self.tf_idf_mat, self.lex_div_traj = load_corpus_data(self.model_name)


    def save_df(self, complevel=9):
        ##########################################################################
        with pd.HDFStore(self.dfpath,complevel=complevel,complib='blosc',mode='w', format='fixed') as store:
            store['df'] = self.df
        ##########################################################################
        print 'Saved dataframe with complevel {}'.format(complevel)


    def get_ba_breakdown_data(self):
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


    def make_token_acts_df(self, sel_probe):
        ##########################################################################
        sel_probe = sel_probe
        token_ids = self.df.query("probe == @sel_probe").index.tolist()
        token_acts_df = self.df.loc[token_ids].filter(regex='H')
        ##########################################################################
        return token_acts_df


    def make_cat_acts_df(self, cat_to_query):
        ##########################################################################
        cat_to_query = cat_to_query
        df_cat = self.df.query("cat == @cat_to_query")
        cat_acts_df = df_cat.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        return cat_acts_df


    def make_all_acts_df(self, agg_fn='mean', decimals=None):
        ##########################################################################
        print 'Making all_acts_df using "{}"...'.format(agg_fn)
        ##########################################################################
        # group by probe (sort has to be True)
        if agg_fn == 'none':
            all_acts_df = self.df.filter(regex='H') # TODO build infrastructure to make this work
        else:
            all_acts_df = self.df.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        # round df ?
        if decimals is not None: all_acts_df.round(decimals)
        ##########################################################################
        return all_acts_df


    def calc_cat_sim_mat(self):
        ##########################################################################
        # probe simmat
        probe_simmat = calc_probe_sim_mat(self.make_all_acts_df(), self.probe_list)
        ##########################################################################
        # inits
        num_probes = len(self.probe_list)
        num_cats = len(self.cat_list)
        # cat_sim_dict = {}
        ##########################################################################
        # make category sim dict
        cat_sim_dict = {cat_outer : {cat_inner : [] for cat_inner in self.cat_list}
                        for cat_outer in self.cat_list}
        for i in range(num_probes):
            probe1 = self.probe_list[i]
            cat1 = self.probe_cat_dict[probe1]
            for j in range(num_probes):
                if i != j:
                    probe2 = self.probe_list[j]
                    cat2 = self.probe_cat_dict[probe2]
                    sim = probe_simmat[i, j]
                    cat_sim_dict[cat1][cat2].append(sim)
        ##########################################################################
        # make category simmat
        cat_simmat = np.zeros([num_cats, num_cats], float)
        for i in range(num_cats):
            cat1 = self.cat_list[i]
            for j in range(num_cats):
                cat2 = self.cat_list[j]
                sims = np.array(cat_sim_dict[cat1][cat2]) # this contains a list of sims
                sim_mean = sims.mean()
                cat_simmat[self.cat_list.index(cat1), self.cat_list.index(cat2)] = sim_mean
        ##########################################################################
        return cat_simmat


    def gen_neighbor_name_and_sim(self, neighbors_for_probe): # TODO thsi should not be part of database.py
        ##########################################################################
        # generate neighbors_name, neighbors_sim
        num_total_neighbors = len(neighbors_for_probe)
        for neighbor_id in range(num_total_neighbors):
            ##########################################################################
            neighbor_name = neighbors_for_probe[neighbor_id][0]
            neighbor_sim = neighbors_for_probe[neighbor_id][1]
            if neighbor_id != 0:
                ##########################################################################
                yield neighbor_name, neighbor_sim


