import os, time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
from operator import itemgetter


from dbutils import calc_ba_mats, calc_probe_sim_mat, load_token_data, load_rnnlabrc


np.set_printoptions(suppress=True)

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
        self.save_ev = int(configs_dict['save_ev'])
        ##########################################################################
        # load token data
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict, \
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict, \
        self.probe_cf_traj_dict, self.num_train_doc_ids = load_token_data(self.model_name)
        ##########################################################################
        # open trajstore
        self.trajstore = pd.HDFStore(self.trajdfpath, complevel=complevel, complib='blosc', mode=mode)


    def calc_new_entry(self, df, all_acts_df, test_pp, block_name, use_classifier=False):
        ##########################################################################
        print 'Making new entry to trajstore...'
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
        # add block_name as to index
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
        print 'Saved new entry to trajstore'
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


    def calc_token_ba_list(self, all_acts_df, block_name, step_size=0.001, verbose=False): # TODO make this step art of user cofigs
        ##########################################################################
        # calc simmat
        probe_simmat = calc_probe_sim_mat(all_acts_df, self.probe_list)
        ##########################################################################
        print 'Calculating balanced accuracy...'
        ##########################################################################
        # make thr_ranges
        num_cpus, thr_start, thr_end, thr_step = 6, 0.7, 1.0, step_size  # TODO how flexible is this?
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
        if mp.cpu_count() < num_cpus:
            print 'rnnlab WARNING: CPU Count is < 6. Parallel calculation of token_ba may not work'
        elif verbose:
            print 'Calculating token ba using {} processes with {} steps...'.format(num_cpus, step_size)
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


    def make_token_ba_trajs_fig(self, sel_probes, is_titled=False):
        ##########################################################################
        # make token_ba_traj_mat
        token_ba_traj_mat = self.make_token_ba_traj_mat(sel_probes)
        ##########################################################################
        # choose seaborn style and palette
        import seaborn as sns  # if globally imported, will change all other figs unpredictably
        sns.set_style('white')
        palette = iter(sns.color_palette("hls", len(sel_probes)).as_hex()) # as hex may not work for matplotlib
        ##########################################################################
        # fig settings
        ymax, max_num_probes = 14, 75  # 75 is largest number of probes (mammals cat)
        ysize = max(6, ymax * len(sel_probes) / max_num_probes)  # prevents fig to be too small
        figsize = (12, ysize)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 8
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        cat = self.probe_cat_dict[sel_probes[0]]
        fig_name = '{} Token Balanced Accuracies for probes in {}'.format(self.model_name, cat)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Training Block', fontsize=ax_font_size)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylim([0, 100])
        ##########################################################################
        # plot
        for token_ba_traj, probe in zip(token_ba_traj_mat, sel_probes):
            x = range(0, len(token_ba_traj) *self.save_ev, self.save_ev)
        #     ax.plot(x, token_ba_traj, '-', linewidth=linewidth, label=probe, c=next(palette))
        ##########################################################################
        # legend
        ax.set_position([0.1, 0.1, 0.8, 0.85]) # 0.8  shrinks width to make room for legend
        ax.legend(fontsize=leg_font_size, loc='best', bbox_to_anchor=(1.11, 1.05))
        ##########################################################################
        return fig, x, token_ba_traj_mat, palette


    def make_cfreq_traj_fig(self, sel_probes, sel_cat, is_titled=False):
        ##########################################################################
        # make cat_ba_traj
        token_ba_traj_mat = self.make_token_ba_traj_mat(sel_probes) # dims: (probes, blocks)
        cat_ba_traj = np.mean(token_ba_traj_mat,  axis=0)
        ##########################################################################
        # choose seaborn style and palette
        import seaborn as sns 
        sns.set_style('white')
        palette = iter(sns.color_palette("hls", len(sel_probes)))
        ##########################################################################
        # fig settings
        ymax, max_num_probes = 10, 75  # 75 is largest number of probes (mammals cat)
        ysize = max(6, ymax * len(sel_probes) / max_num_probes)  # prevents fig to be too small
        figsize = (12, ysize)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 12
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Cumulative Frequency of probes in {}'.format(self.model_name, sel_cat)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Training Block', fontsize=ax_font_size)
        ax.set_ylabel('Cumulative Frequency', fontsize=ax_font_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ##########################################################################
        # calc Y_thr (so that annotations are made only for probes with y hgiher than y_thr)
        maxperc, max_num_probes = 95, 75
        percentile = max(85, maxperc * len(sel_probes) /max_num_probes)
        num_trained_blocks = len(cat_ba_traj) * self.save_ev  # in case model has not competed training
        y_thr = np.percentile([self.probe_cf_traj_dict[probe][:num_trained_blocks][-1]
                               for probe in sel_probes], percentile)
        ##########################################################################
        # plot cf
        for probe in sel_probes:
            probe_cf_traj_all_blocks = self.probe_cf_traj_dict[probe]
            probe_cf_traj = probe_cf_traj_all_blocks[:num_trained_blocks]
            x = range(len(probe_cf_traj))
            ax.plot(x, probe_cf_traj, '--', linewidth=linewidth, c=next(palette))
            ##########################################################################
            # annotate
            y = probe_cf_traj[-1]
            if y > y_thr:  # annotate only those targets with y higher than y_thr to reduce clutter
                plt.annotate(probe, xy=(x[-1], y), xytext=(-30, -10), textcoords='offset points',
                             va='center', fontsize=leg_font_size, bbox=dict(boxstyle='round', fc='w'))
        ##########################################################################
        ax.legend(fontsize=leg_font_size, loc='upper left')
        ##########################################################################
        return fig


    def make_ba_pp_window_corr_fig(self, window=20, is_title=False):
        ##########################################################################
        # load data
        avg_token_ba_traj = self.trajstore.select_column('trajdf', 'avg_token_ba').values
        s1 = pd.Series(avg_token_ba_traj)
        test_pp_traj = self.trajstore.select_column('trajdf', 'test_pp').values
        s2 = pd.Series(test_pp_traj)
        ##########################################################################
        #  window corr
        ba_pp_mw_corr = s1.rolling(window=window).corr(s2)
        ba_pp_ew_corr = s1.expanding().corr(s2)
        ##########################################################################
        # choose seaborn style and palette
        import seaborn as sns  # if globally imported, will change all other figs unpredictably
        sns.set_style('white')
        ##########################################################################
        # fig settings
        figsize = (6, 4)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 8
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Test Perplexity Trajectory'.format(self.model_name)
        if is_title: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axis
        ax.set_ylim([-1, 1])
        ax.set_xlabel('Training Block'.format(window), fontsize=ax_font_size)
        ax.set_ylabel('Correlation Coefficient', fontsize=ax_font_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ##########################################################################
        # plot line through y=0
        x = range(0, len(ba_pp_mw_corr) * self.save_ev, self.save_ev)
        ax.plot(x, [0]*len(x), '--', c='gray', linewidth=linewidth)
        ##########################################################################
        # plot
        x = range(0, len(ba_pp_mw_corr) * self.save_ev, self.save_ev)
        ax.plot(x, ba_pp_mw_corr, '-', linewidth=linewidth,
                label='mw-corr ({} blocks) between balAcc and test-pp'.format(window))
        x = range(0, len(ba_pp_mw_corr) * self.save_ev, self.save_ev)
        ax.plot(x, ba_pp_ew_corr, '-', linewidth=linewidth,
                label='ew-corr between balAcc and test-pp')
        ##########################################################################
        ax.legend(fontsize=leg_font_size, loc='best')
        ##########################################################################
        # layout
        plt.tight_layout()
        ##########################################################################
        return fig


    def make_compprobes_fig(self, sel_custom_prob_tuples, is_titled=False):
        ##########################################################################
        # seaborn
        import seaborn as sns
        sns.set_style('white')
        num_probe_classes = sum(1 for g in groupby(sel_custom_prob_tuples, itemgetter(1)))
        palette = iter(sns.color_palette("hls", num_probe_classes))
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Probe balanced Accuracy Trajectory Comparison'.format(self.model_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # plot
        n = 0
        for probe_class, group in groupby(sel_custom_prob_tuples, itemgetter(1)):
            sel_probes = [i[0] for i in list(group)]
            fig_, x, ys, _ = self.make_token_ba_trajs_fig(sel_probes)
            if n == 0: fig = fig_
            n += 1
            df = pd.DataFrame(ys)
            avg_ys = df.mean()
            std_ys = df.std()
            n_ys = len(df)
            se_ys = std_ys / (n_ys ** 0.5)
            import scipy
            ci_ys = se_ys * scipy.stats.t._ppf((1 + 0.95) / 2., n_ys - 1) # confiden ce interval

            ax = fig.gca()
            ax.errorbar(x, avg_ys, yerr=ci_ys, fmt='-o', c=next(palette),
                        label='{} (mean +- 95% CI) n={}'.format(probe_class.replace('_', ' '), len(sel_probes)))
        ##########################################################################
        ax.set_ylim([40, 100])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=14, loc='best')
        ##########################################################################
        return fig



