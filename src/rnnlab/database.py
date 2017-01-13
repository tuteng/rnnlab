import os, sys, time
from operator import itemgetter
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
from rnnhelper import load_rc


class DataBase:
    """
    Stores dataframe constructed during rnn training
    Calculates similarities between probes, and balanced accuracy, and can generate plots

    Automatically created during rnn training
    Can also be instantiated outside of training for pos-hoc analysis
    """

    def __init__(self, configs_dict, df, block_name):
        ##########################################################################
        # define directories
        working_dir = os.path.dirname(os.path.abspath(__file__))
        self.rnn_dir = os.path.abspath(working_dir + os.sep + '..' + os.sep + '..')
        self.data_dir = os.path.join(self.rnn_dir, 'data')
        self.runs_dir = load_rc('runs_dir')
        ##########################################################################
        # assign instance variables
        self.configs_dict = configs_dict
        self.probes_name = configs_dict['probes_name']
        self.model_name = configs_dict['model_name']
        self.df = df
        self.block_name = block_name
        self.acts_mat, self.acts_df = self.make_all_acts_mat()
        self.token_list, self.token_id_dict, self.probe_list, \
        self.probe_id_dict, self.probe_cat_dict, self.cat_list = self.load_token_data()


    def save_df(self): # only df that is instance of database class will be saved
        ##########################################################################
        path = os.path.join(self.runs_dir, self.model_name, 'Data_Frame')
        file_name = 'df_block_{}.h5'.format(self.block_name)
        self.df.to_hdf(os.path.join(path, file_name), 'df')
        print 'Saved df'


    def calc_hca_df_col(self, epochs=30):
        ########################################################################################
        print 'Calculating classifier accuracy...'
        from classifier import calc_hca
        x_data = np.asarray(self.df.filter(regex='H'))
        df_cat_col = list(self.df['cat'])
        assert len(x_data) == len(df_cat_col)
        y_data = np.zeros(len(df_cat_col))
        for n, cat in enumerate(df_cat_col):
            y_data[n] = self.cat_list.index(cat)
        train_hca, test_hca = calc_hca(self.model_name, self.block_name, x_data, y_data, epochs)
        train_hca_df_col, test_hca_df_col = [train_hca] * self.df.shape[0], [test_hca] * self.df.shape[0]
        ########################################################################################
        return train_hca_df_col, test_hca_df_col


    def calc_token_ba_df_col(self): # uses multiprocessing
        ##########################################################################
        # calc simmat
        probe_simmat = self.calc_probe_sim_mat()
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
        async_results = [pool.apply_async(calc_ba_mats, args=(self, probe_simmat, thr_list, 'token')) for thr_list in thr_lists]
        results = [result.get() for result in async_results]
        token_ba_mat = np.hstack((mat for mat in results))
        pool.close()
        ##########################################################################
        # token_ba_mat = self.calc_ba_mats(probe_simmat, num_thrs, thrs, 'token')
        token_ba_mat_col_means = np.nanmean(token_ba_mat, 0) * 100
        best_token_ba_mat_col_id = np.argmax(token_ba_mat_col_means)
        best_token_ba_mat_col = token_ba_mat[:, best_token_ba_mat_col_id] * 100
        token_ba_df_col = [best_token_ba_mat_col[self.probe_list.index(probe)] for probe in self.df['probe']]
        ##########################################################################
        return token_ba_df_col


    def expand_df(self, col_names):
        ##########################################################################
        if 'token_ba' in col_names:
            token_ba_df_col = self.calc_token_ba_df_col()
            self.df['token_ba'] = token_ba_df_col
        ##########################################################################
        if 'hca' in col_names:
            train_hca_df_col, test_hca_df_col = self.calc_hca_df_col()
            self.df['train_hca'] = train_hca_df_col
            self.df['test_hca'] = test_hca_df_col
        ##########################################################################
        return self.df


    def calc_probe_sim_mat(self):
        ##########################################################################
        print 'Calculating simmat...'
        ##########################################################################
        # calc sim mat
        probe_simmat = np.asarray(self.acts_df.T.corr(method='pearson'))
        ##########################################################################
        # save simmat
        path = os.path.join(self.runs_dir, self.model_name, 'Sim_Mat')
        file_name = '{}_simmat_block_{}.npz'.format(self.probes_name, self.block_name)
        np.savez(os.path.join(path, file_name),
                 sim_mat=probe_simmat,
                 probe_list=self.probe_list)
        assert probe_simmat.shape == (len(self.probe_list), len(self.probe_list))
        nan_ids = np.where(np.isnan(probe_simmat).all(axis=1))[0]
        assert len(nan_ids) == 0
        ##########################################################################
        return probe_simmat


    def get_ba_breakdown_data(self):
        ##########################################################################
        # make df_cat_and_ba
        df_cat_ba = self.df[['cat', 'token_ba']].drop_duplicates().groupby('cat', sort=False).mean()
        ##########################################################################
        # make cats_sorted_by_ba
        tuples = [tuple for tuple in df_cat_ba.itertuples()]
        tuples_sorted_by_ba = sorted(tuples, key=itemgetter(1))
        cats_sorted_by_ba = [tuple[0] for tuple in tuples_sorted_by_ba]
        ##########################################################################
        # make cat_ba_dict (this is not really needed)
        cat_ba_dict = df_cat_ba.to_dict()['token_ba']
        ##########################################################################
        # make cat_probe_list_dict
        cat_probe_list_dict = {cat: [probe for probe in self.probe_list
                                     if self.probe_cat_dict[probe] == cat] for cat in self.cat_list}
        ##########################################################################
        # make token_ba_row
        token_ba_row = self.df[['probe','token_ba']].groupby('probe').first()['token_ba'].as_matrix()
        ##########################################################################
        return cats_sorted_by_ba, cat_ba_dict, cat_probe_list_dict, token_ba_row


    def make_token_acts_df(self, probe):
        ##########################################################################
        token_ids = self.df[self.df['probe'] == probe].index.tolist()
        token_acts_df = self.df.loc[token_ids].filter(regex='H')
        ##########################################################################
        return token_acts_df


    def make_cat_acts_mat(self, cat, agg_fn='mean'):
        ##########################################################################
        print 'Making cat_acts_mat using "{}"...'.format(agg_fn)
        ##########################################################################
        df_cat = self.df[self.df['cat'] == cat]
        cat_acts_df = df_cat.groupby('probe', sort=True).mean().filter(regex='H')  # TODO make sure this works
        ##########################################################################
        # make cat_acts_mat
        cat_acts_mat = np.asarray(cat_acts_df)
        ##########################################################################
        return cat_acts_mat, cat_acts_df


    def make_all_acts_mat(self, agg_fn='mean'):
        ##########################################################################
        print 'Making all_acts_mat using "{}"...'.format(agg_fn)
        ##########################################################################
        # group by probe (sort has to be True)
        if agg_fn == 'none':
            all_acts_df = self.df.filter(regex='H') # TODO test if this works (need to sort it)
        else:
            all_acts_df = self.df.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        # make all_acts_mat
        all_acts_mat = np.asarray(all_acts_df)
        ##########################################################################
        return all_acts_mat, all_acts_df


    def calculate_dprime(self, hits, misses, fas, crs):
        ##########################################################################
        from scipy.stats import norm
        from math import exp, sqrt
        Z = norm.ppf
        # Floors an ceilings are replaced by half hits and half FA's
        halfHit = 0.5 / (hits + misses)
        halfFa = 0.5 / (fas + crs)
        # Calculate hitrate and avoid d' infinity
        hitRate = hits / (hits + misses)
        if hitRate == 1: hitRate = 1 - halfHit
        if hitRate == 0: hitRate = halfHit
        # Calculate false alarm rate and avoid d' infinity
        faRate = fas / (fas + crs)
        if faRate == 1: faRate = 1 - halfFa
        if faRate == 0: faRate = halfFa
        # Return d', beta, c and Ad'
        d_prime = Z(hitRate) - Z(faRate)
        beta = exp(Z(faRate) ** 2 - Z(hitRate) ** 2) / 2
        c = -(Z(hitRate) + Z(faRate)) / 2
        ad = norm.cdf(d_prime / sqrt(2))
        ##########################################################################
        return d_prime, beta, c, ad


    def calc_cat_sim_mat(self, probe_simmat):
        ##########################################################################
        # inits
        num_probes = len(self.probe_list)
        num_cats = len(self.cat_list)
        cat_dim_dict = {}
        ##########################################################################
        # make category sim dict
        for category1 in self.cat_list:
            for category2 in self.cat_list:
                cat_dim_dict[category1][category2] = []
        for i in range(num_probes):
            word1 = self.probe_list[i]
            category1 = self.probe_cat_dict[word1]
            for j in range(num_probes):
                if i != j:
                    word2 = self.probe_list[j]
                    category2 = self.probe_cat_dict[word2]
                    sim = probe_simmat[i, j]
                    cat_dim_dict[category1][category2].append(sim)
        ##########################################################################
        # make category simmat
        cat_simmat = np.zeros([num_cats, num_cats], float)
        for i in range(num_cats):
            cat1 = self.cat_list[i]
            for j in range(num_cats):
                cat2 = self.cat_list[j]
                sims = np.array(cat_dim_dict[cat1][cat2]) # this contains a list of sims
                sim_mean = sims.mean()
                cat_simmat[self.cat_list.index(cat1), self.cat_list.index(cat2)] = sim_mean
        ##########################################################################
        return cat_simmat


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


    def make_acts_2d_fig(self, label_probe=False, is_titled=False):
        ##########################################################################
        # svd
        u, s, v = linalg.svd(self.acts_mat)  # row_singular_vectors, singular_values, column_singular_vectors
        acts_2d_svd = u[:, :2]  # TODO make sure this is right
        ##########################################################################
        # tsne
        acts_2d_tsne = TSNE().fit_transform(self.acts_mat)
        ##########################################################################
        # get cat for each probe for plotting
        acts_cats = [self.probe_cat_dict[probe] for probe in self.probe_list]
        ##########################################################################
        # choose a style with seaborn
        import seaborn as sns # if globally imported, will change all other figs unpredictably
        sns.set_style('white')
        ##########################################################################
        # make scatter and add text
        palette = np.array(sns.color_palette("hls", len(self.cat_list)))
        fig, axarr = plt.subplots(1, 2, figsize=(12, 8))
        for n, x in enumerate([acts_2d_svd, acts_2d_tsne]):
            palette_ids = [self.cat_list.index(cat) for cat in acts_cats]
            axarr[n].scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[palette_ids])
            axarr[n].axis('off')
            axarr[n].axis('tight')
            axarr[n].set_title(['SVD', 't-SNE'][n], fontsize=16)
            ##########################################################################
            # add the labels for each cat
            for cat in self.cat_list:
                x_ids = np.where(np.asarray(acts_cats) == cat)[0]
                xtext, ytext = np.median(x[x_ids, :], axis=0)
                txt = axarr[n].text(xtext, ytext, str(cat), fontsize=12, color=palette[self.cat_list.index(cat)])
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=5, foreground="w"),
                    PathEffects.Normal()])
            ##########################################################################
            # add the labels for each probe
            if label_probe:
                for probe in self.probe_list:
                    x_ids = np.where(np.asarray(self.probe_list) == probe)[0]
                    xtext, ytext = np.median(x[x_ids, :], axis=0)
                    txt = axarr[n].text(xtext, ytext, str(probe), fontsize=8)
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=5, foreground="w"),
                        PathEffects.Normal()])
        ##########################################################################
        return fig


    def get_probes_from_cat(self, cat):
        ##########################################################################
        cat_ids =  self.df[self.df['cat'] == cat].index.tolist()
        probes = self.df.loc[cat_ids]['probe'].unique().tolist()
        ##########################################################################
        return probes


    def make_token_corcoeff_hist_fig(self, probe, bins=100, is_titled=False):
        ##########################################################################
        token_acts_df = self.make_token_acts_df(probe)
        df_corr = token_acts_df.T.corr(method='pearson')
        mask_mat = np.triu(np.ones(df_corr.shape)).astype(np.bool) # don't need this
        corr_mat_nans = np.asarray(df_corr.mask(mask_mat))
        corr_mat = corr_mat_nans[~np.isnan(corr_mat_nans)]
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        label_fontsize = 4
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} Hist of acts corcoeffs for "{}" '.format(self.model_name, self.block_name, probe)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=ax_font_size)
        ax.set_ylabel('Number of observations', fontsize=ax_font_size)
        ax.hist(corr_mat, bins)
        ##########################################################################
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ##########################################################################
        return fig


    def make_pp_curve_fig(self, smoothing_span=20): # TODO this doesn't make sense to be in database because it doesn't have access to all dfs
        ##########################################################################
        # pp_curve = ?
        if smoothing_span > 1:
            from pandas.stats.moments import ewma
            pp_curve = ewma(np.asarray(pp_curve), span=smoothing_span)
        else:
            pp_curve = pp_curve
        ##########################################################################
        # TODO finish this


    def make_acts_dh_fig(self, probe=None, num_colors = 0, vmin=0.0, vmax=1.0, is_titled=False):
        ##########################################################################
        # make acts_df
        if probe:
            token_acts_df = self.make_token_acts_df(probe)
            acts_mat = np.asarray(token_acts_df)
        else:
            acts_mat = self.acts_mat
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        label_fontsize = 4
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax_heatmap = plt.subplots(figsize=figsize)
        if probe:
            fig_name = '{} Block {} DH of acts for "{}" '.format(self.model_name, self.block_name, probe)
        else:
            fig_name = '{} Block {} DH of acts for all probes '.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size, y=1.2)
        ##########################################################################
        # axes
        ax_heatmap.yaxis.tick_right()
        ax_heatmap.set_xlabel('Hidden Units', fontsize=ax_font_size)
        num_acts = len(acts_mat)
        if probe:
            ax_heatmap.set_ylabel('{} Examples of "{}"'.format(num_acts, probe), fontsize=ax_font_size)
        else:
            ax_heatmap.set_ylabel('All {}  probes'.format(num_acts), fontsize=ax_font_size)
        divider = make_axes_locatable(ax_heatmap)
        ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
        ax_dendleft.set_frame_on(False)
        ax_dendtop = divider.append_axes("top", 1.0, pad=0.0, sharex=ax_heatmap)
        ax_dendtop.set_frame_on(False)
        ax_colorbar = divider.append_axes("left", 0.2, pad=0.8)
        ##########################################################################
        # set linewidth of dendrogram
        from matplotlib import rcParams
        rcParams['lines.linewidth'] = 0.5
        ##########################################################################
        # left dendrogram
        lnk0 = linkage(pdist(acts_mat))
        if num_colors is None or num_colors <= 1:
            left_cluster_thr = -1
        else:
            left_cluster_thr = 0.5 * (lnk0[1 - num_colors, 2] +
                                      lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0,
                         ax=ax_dendleft,
                         orientation='right',
                         color_threshold=left_cluster_thr,
                         no_labels=True)
        # top dendrogram
        lnk1 = linkage(pdist(acts_mat.T))
        if num_colors is None or num_colors <= 1:
            top_cluster_thr = -1
        else:
            top_cluster_thr = 0.5 * (lnk1[1 - num_colors, 2] +
                                     lnk1[-num_colors, 2])

        dg1 = dendrogram(lnk1,
                         ax=ax_dendtop,
                         color_threshold=top_cluster_thr,
                         no_labels=True)
        ##########################################################################
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = acts_mat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg1['leaves']]  # sorting cols
        ##########################################################################
        # heatmap
        im = ax_heatmap.imshow(z[::-1],
                               aspect='auto',
                               cmap=plt.cm.jet,
                               interpolation='nearest',
                               extent=(0, ax_dendtop.get_xlim()[1], 0, ax_dendleft.get_ylim()[1]),
                               vmin=vmin,
                               vmax=vmax)
        ##########################################################################
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax])
        cb.ax.set_xticklabels([vmin, vmax])
        cb.set_label('Strength of Activation', labelpad=-50, fontsize=ax_font_size)
        ##########################################################################
        # hide heatmap labels
        ax_heatmap.xaxis.set_ticklabels([])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_dendleft.xaxis.get_ticklines() +
                 ax_dendleft.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        lines = (ax_dendtop.xaxis.get_ticklines() +
                 ax_dendtop.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # make dendrogram labels invisible
        plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
                 visible=False)
        plt.setp(ax_dendtop.get_xticklabels() + ax_dendtop.get_yticklabels(),
                 visible=False)
        ##########################################################################
        return fig


    def make_cat_sim_dh_fig(self, num_colors = 0, vmin=0.0, vmax=1.0, is_titled=False):
        ##########################################################################
        # calc probe simmat
        probe_simmat = self.calc_probe_sim_mat()
        ##########################################################################
        # calc cat simmat
        cat_simmat, cat_simmat_labels = self.calc_cat_sim_mat(probe_simmat)
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        label_fontsize = 8
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax_heatmap = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} DH of cat simmat '.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size, y=1.03)
        ##########################################################################
        # axes
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_dendleft = divider.append_axes("right", 2.0, pad=1.0, sharey=ax_heatmap)
        ax_dendleft.set_frame_on(False)
        ax_colorbar = divider.append_axes("left", 0.2, pad=0.5)
        ##########################################################################
        # dendrogram
        lnk0 = linkage(pdist(cat_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_dendleft,
                         orientation='right',
                         color_threshold=left_threshold,
                         no_labels=True)
        ##########################################################################
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = cat_simmat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        ##########################################################################
        # heatmap
        max_extent = ax_dendleft.get_ylim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto', # TODO does cmap have an effect? cmap=plt.cm.jet,
                               interpolation='nearest',
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax])
        cb.ax.set_xticklabels([vmin, vmax])
        ##########################################################################
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        ncols = len(cat_simmat_labels)
        halfxw = 0.5 * xlim / ncols
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, ncols))
        ax_heatmap.xaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])  # for symmetry
        ylim = ax_heatmap.get_ylim()[1]
        nrows = len(cat_simmat_labels)
        halfyw = 0.5 * ylim / nrows
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, nrows))
        ax_heatmap.yaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_dendleft.xaxis.get_ticklines() +
                 ax_dendleft.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=label_fontsize)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=label_fontsize)
        # make dendrogram labels invisible
        plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
                 visible=False)
        ##########################################################################
        return fig


    def make_cat_cluster_fig(self, cat, is_titled=False):
        ##########################################################################
        # fig settings
        figsize = (10, 10)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} Clustering of {}'.format(self.model_name, self.block_name, cat)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # get cat_acts_mat
        cat_acts_mat, cat_acts_df = self.make_cat_acts_mat(cat)
        probes_in_cat = cat_acts_df.index.tolist()
        ##########################################################################
        # dendrogram
        dist_matrix = pdist(cat_acts_mat, 'euclidean')
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   leaf_label_func=lambda x: probes_in_cat[x],
                   orientation='left',
                   leaf_font_size=10)
        ##########################################################################
        return fig


    def make_ba_breakdown_fig(self, is_titled=False): # TODO put this next to scatter breakdown
        ##########################################################################
        # make cat_probe_list_dict and cats_sorted_by_ba
        cats_sorted_by_ba, cat_ba_dict, cat_probe_list_dict, token_ba_row = self.get_ba_breakdown_data()
        ##########################################################################
        # fig settings
        figsize = (14, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} BA per Category'.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Categories', fontsize=ax_font_size)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
        ax.set_xticks(np.arange(len(self.probe_list)) + 0.5, minor=False)
        ax.set_xticklabels(cats_sorted_by_ba, minor=False, fontsize=leg_font_size, rotation=90)
        ax.set_xlim([0, len(self.cat_list) + 0.5])
        ##########################################################################
        # plot
        for cat_id, cat in enumerate(cats_sorted_by_ba):
            cat_probe_list = cat_probe_list_dict[cat]
            cat_probe_ids = [self.probe_list.index(e_token) for e_token in cat_probe_list]
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_row[cat_probe_ids]
            ax.plot(xs, ys, 'b.', alpha=1)
        ##########################################################################
        return fig


    def make_ba_breakdown_scatter_fig(self, is_titled=False):
        ##########################################################################
        # make cat_probe_list_dict and cats_sorted_by_ba
        cats_sorted_by_ba, cat_ba_dict, cat_probe_list_dict, token_ba_row = self.get_ba_breakdown_data()
        ##########################################################################
        # fig settings
        figsize = (14, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} BA per Category'.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Categories', fontsize=ax_font_size)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
        ax.set_xticks(np.arange(len(self.probe_list)) + 0.5, minor=False)
        ax.set_xticklabels(cats_sorted_by_ba, minor=False, fontsize=leg_font_size, rotation=90)
        ax.set_xlim([0, len(self.cat_list) + 0.5])
        ##########################################################################
        # Hide the right and top spines and ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ##########################################################################
        # mean line
        mean_ba = np.nanmean(cat_ba_dict.values())
        ax.plot(range(0, len(cats_sorted_by_ba) + 1),
                [mean_ba for i in range(len(cats_sorted_by_ba) + 1)],
                '-', alpha=0.5, c='grey')
        ##########################################################################
        # plot
        annotated_y_ints_long_words_prev_cat = []
        for cat_id, cat in enumerate(cats_sorted_by_ba):
            ##########################################################################
            # plot points
            annotated_y_ints_long_words_curr_cat = []
            cat_probe_list = cat_probe_list_dict[cat]
            cat_probe_ids = [self.probe_list.index(e_token) for e_token in cat_probe_list]
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_row[cat_probe_ids]
            ax.plot(xs, ys, 'b.', alpha=0) # this needs to be plot for annotation to work
            ##########################################################################
            # annotate points
            annotated_y_ints = []
            for x,y, target in sorted(zip(xs, ys, cat_probe_list_dict[cat])):
                y_int = int(y)
                # if annotation coordinate exists or is affected by long word from previous cat, skip to next target
                if not y_int in annotated_y_ints and y_int not in annotated_y_ints_long_words_prev_cat:
                    plt.annotate(target, xy=(x, y_int), xytext=(2, 0), textcoords='offset points', va='bottom',
                                 fontsize=7)
                    annotated_y_ints.append(y_int)
                    if len(target) > 7:
                        annotated_y_ints_long_words_curr_cat.append(y_int)
            annotated_y_ints_long_words_prev_cat = annotated_y_ints_long_words_curr_cat
        ##########################################################################
        plt.tight_layout()  # has to be called after axes are filled
        ##########################################################################
        return fig

    
def calc_ba_mats(database, probe_simmat, thr_list, output, verbose=False): # output  is either 'token' or 'cat'
    ##########################################################################
    ba_mat = None
    num_probes = len(database.probe_list)
    num_cats = len(database.cat_list)
    num_thrs = len(thr_list)
    ##########################################################################
    # print header for analysis
    if verbose:
        print 'Calculating ba with thresholds {} to {}...'.format(thr_list[0], thr_list[-1])
        print '{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}'.format(
            'Threshold', 'Hits', 'Misses', 'HitRate', 'CR', 'FA', 'CRRate', 'BA', 'dprime', 'c')
    ##########################################################################
    # initialisations
    category_hits = np.zeros([num_cats, num_thrs], float)
    category_misses = np.zeros([num_cats, num_thrs], float)
    category_false_alarms = np.zeros([num_cats, num_thrs], float)
    category_correct_rejections = np.zeros([num_cats, num_thrs], float)
    item_hits = np.zeros([num_probes, num_thrs], float)
    item_misses = np.zeros([num_probes, num_thrs], float)
    item_false_alarms = np.zeros([num_probes, num_thrs], float)
    item_correct_rejections = np.zeros([num_probes, num_thrs], float)
    cat_ba_mat = np.zeros([num_cats, num_thrs], float)
    token_ba_mat = np.zeros([num_probes, num_thrs], float)
    ##########################################################################
    for n, thr in enumerate(thr_list):
        ##########################################################################
        # calc hits, misses, false alarms, correct rejections
        for token_id_1 in range(num_probes):
            token_1 = database.probe_list[token_id_1]
            cat_1 = database.probe_cat_dict[token_1]
            cat_id_1 = database.cat_list.index(cat_1)

            for token_id_2 in range(num_probes):
                if token_id_1 != token_id_2:
                    token_2 = database.probe_list[token_id_2]
                    cat_2 = database.probe_cat_dict[token_2]
                    sim = probe_simmat[token_id_1, token_id_2]

                    if sim != 'nan':
                        if cat_1 == cat_2:
                            if sim > thr:
                                category_hits[cat_id_1, n] += 1
                                item_hits[token_id_1, n] += 1
                            else:
                                category_misses[cat_id_1, n] += 1
                                item_misses[token_id_1, n] += 1
                        else:
                            if sim > thr:
                                category_false_alarms[cat_id_1, n] += 1
                                item_false_alarms[token_id_1, n] += 1
                            else:
                                category_correct_rejections[cat_id_1, n] += 1
                                item_correct_rejections[token_id_1, n] += 1
        ##########################################################################
        # calc category balanced accuracy
        if output == 'cat':
            for cat_id in range(num_cats):

                current_hits = category_hits[cat_id, n]
                current_misses = category_misses[cat_id, n]
                current_false_alarms = category_false_alarms[cat_id, n]
                current_correct_rejections = category_correct_rejections[cat_id, n]

                if (current_hits + current_misses) > 0:
                    sensitivity = current_hits / (
                        current_hits + current_misses)  # perc correct when correct answer answer was "same"
                else:
                    sensitivity = 'nan'
                if (current_correct_rejections + current_false_alarms) > 0:
                    specificity = current_correct_rejections / (
                        current_correct_rejections + current_false_alarms)  # perc correct when correct answer was "different"
                else:
                    specificity = 'nan'
                if (sensitivity != 'nan') and (specificity != 'nan'):
                    current_balanced_accuracy = (float(sensitivity) + float(specificity)) / 2.0
                else:
                    current_balanced_accuracy = 'nan'

                cat_ba_mat[cat_id, n] = current_balanced_accuracy
                ba_mat = cat_ba_mat
        ##########################################################################
        # calc token balanced accuracy
        elif output == 'token':
            for token_id in range(num_probes):

                current_hits = item_hits[token_id, n]
                current_misses = item_misses[token_id, n]
                current_false_alarms = item_false_alarms[token_id, n]
                current_correct_rejections = item_correct_rejections[token_id, n]

                if (current_hits + current_misses) > 0:
                    sensitivity = current_hits / (
                        current_hits + current_misses)  # perc correct when correct answer answer was "same"
                else:
                    sensitivity = 'nan'
                if (current_correct_rejections + current_false_alarms) > 0:
                    specificity = current_correct_rejections / (
                        current_correct_rejections + current_false_alarms)  # perc correct when correct answer was "different"
                else:
                    specificity = 'nan'
                if (sensitivity != 'nan') and (specificity != 'nan'):
                    current_item_BA = (float(sensitivity) + float(specificity)) / 2.0
                else:
                    current_item_BA = 'nan'

                token_ba_mat[token_id, n] = current_item_BA
                ba_mat = token_ba_mat
        ##########################################################################
        # print signal detection scores
        if verbose:
            current_hit_sums = category_hits[:, n].sum()
            current_miss_sums = category_misses[:, n].sum()
            current_hit_rate = current_hit_sums / (current_hit_sums + current_miss_sums)
            current_correct_rejection_sums = category_correct_rejections[:, n].sum()
            current_false_alarm_sums = category_false_alarms[:, n].sum()
            current_cr_rate = current_correct_rejection_sums / (
                current_correct_rejection_sums + current_false_alarm_sums)
            d_prime, beta, c, ad = database.calculate_dprime(
                current_hit_sums, current_miss_sums, current_false_alarm_sums, current_correct_rejection_sums)
            current_threshold_BA_mean = np.nanmean(cat_ba_mat[:, n])
            print '{:>10.4}{:>10}{:>10}{:>10.4}{:>10}{:>10}{:>10.4}{:>10.4}{:>10.2}{:>10.4}' \
                .format(thr,
                        current_hit_sums,
                        current_miss_sums,
                        current_hit_rate * 100,
                        current_correct_rejection_sums,
                        current_false_alarm_sums,
                        current_cr_rate * 100,
                        current_threshold_BA_mean * 100,
                        d_prime,
                        c)
    ##########################################################################
    # save
    path = os.path.join(database.runs_dir, database.model_name, 'Balanced_Accuracy')
    file_name = '{}_ba_block_{}.npz'.format(database.probes_name, database.block_name)
    np.savez(os.path.join(path, file_name),
             output=output,
             ba_mat=ba_mat,
             num_thrs=num_thrs)
    ##########################################################################
    return ba_mat