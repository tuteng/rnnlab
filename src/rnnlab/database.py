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
from utilities import calc_probe_sim_mat, load_token_data
import pandas as pd

class DataBase:
    """
    Stores dataframe constructed during rnn training
    Calculates similarities between probes, and balanced accuracy, and can generate plots

    Automatically created during rnn training
    Can also be instantiated outside of training for pos-hoc analysis
    """

    def __init__(self, configs_dict, df, block_name):
        ##########################################################################
        # define dfpath
        runs_dir = load_rc('runs_dir')
        self.dfpath = os.path.join(runs_dir, configs_dict['model_name'], 'Data_Frame',
                                   'df_block_{}.h5'.format(block_name))
        ##########################################################################
        # assign instance variables
        self.model_name = configs_dict['model_name']
        self.block_name = block_name
        self.df = df
        ##########################################################################
        # load token data
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict,\
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict, \
        self.probe_cf_traj_dict = load_token_data(runs_dir, self.model_name)
        ##########################################################################
        # calc instance variables
        self.all_acts_df = self.make_all_acts_df()
        self.probe_simmat = calc_probe_sim_mat(self.all_acts_df, self.probe_list)


    def add_col(self,col_name, col_data):
        ##########################################################################
        assert len(col_data) == len(self.df)
        self.df[col_name] = col_data
        ##########################################################################
        print 'Added {} to dataframe'.format(col_name)


    def save_df(self, complevel=9):
        ##########################################################################
        # with pd.HDFStore(self.dfpath,complevel=complevel,complib='blosc',mode='w') as store:
        #     store.append('df', self.df, format='table')

        with pd.HDFStore(self.dfpath,complevel=complevel,complib='blosc',mode='w', format='fixed') as store: # TODO make sure fixed format works
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
        # make token_ba_row
        token_ba_row = self.df[['probe','token_ba']].groupby('probe').first()['token_ba'].as_matrix()
        ##########################################################################
        return cats_sorted_by_ba, cat_ba_dict, token_ba_row


    def make_token_acts_df(self, sel_probe):
        ##########################################################################
        sel_probe = sel_probe
        token_ids = self.df.query("probe == @sel_probe").index.tolist()
        token_acts_df = self.df.loc[token_ids].filter(regex='H')
        ##########################################################################
        return token_acts_df


    def make_cat_acts_df(self, sel_cat, agg_fn='mean'):
        ##########################################################################
        print 'Making cat_acts_mat using "{}"...'.format(agg_fn)
        ##########################################################################
        sel_cat = sel_cat
        df_cat = self.df.query("cat == @sel_cat")
        cat_acts_df = df_cat.groupby('probe', sort=True).mean().filter(regex='H')  # TODO make sure this works
        ##########################################################################
        return cat_acts_df


    def make_all_acts_df(self, agg_fn='mean', decimals=None):
        ##########################################################################
        print 'Making all_acts_df using "{}"...'.format(agg_fn)
        ##########################################################################
        # group by probe (sort has to be True)
        if agg_fn == 'none':
            all_acts_df = self.df.filter(regex='H') # TODO test if this works (need to sort it)
        else:
            all_acts_df = self.df.groupby('probe', sort=True).mean().filter(regex='H')
        ##########################################################################
        # round df ?
        if decimals is not None: all_acts_df.round(decimals)
        ##########################################################################
        return all_acts_df


    def calc_cat_sim_mat(self):
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
                    sim = self.probe_simmat[i, j]
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



    def make_acts_2d_fig(self, label_probe=False, is_titled=False):
        ##########################################################################
        # choose seaborn style and palette
        import seaborn as sns  # if globally imported, will change all other figs unpredictably
        sns.set_style('white')
        palette = np.array(sns.color_palette("hls", len(self.cat_list)))
        ##########################################################################
        # svd
        u, s, v = linalg.svd(self.all_acts_df.values)  # row_singular_vectors, singular_values, column_singular_vectors
        acts_2d_svd = u[:, :2]  # TODO make sure this is right
        ##########################################################################
        # tsne
        acts_2d_tsne = TSNE().fit_transform(self.all_acts_df.values)
        ##########################################################################
        # get cat for each probe for plotting
        acts_cats = [self.probe_cat_dict[probe] for probe in self.probe_list]
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
        fig, axarr = plt.subplots(1, 2, figsize=(12, 8))
        fig_name = '{} Block {} Acts Dimensionality Reduction'.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axis
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
        ax.set_xlim([0, 1]) # this doesn't work well before model has trained
        ##########################################################################
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ##########################################################################
        return fig


    def gen_neighbor_name_and_sim(self, probe):
        ##########################################################################
        # convert probe_simmat to sim_tuples_list_list
        sim_tuples_list_list = []
        for row_id, row in enumerate(self.probe_simmat):
            sim_tuples_list_list.append([(target, sim) for target, sim in zip(self.probe_list, row)])
        ##########################################################################
        # generate neighbors_name, neighbors_sim
        neighbors_for_probe = sorted(sim_tuples_list_list[self.probe_id_dict[probe]], key=itemgetter(1), reverse=True)
        num_total_neighbors = len(neighbors_for_probe)
        for neighbor_id in range(num_total_neighbors):
            ##########################################################################
            neighbor_name = neighbors_for_probe[neighbor_id][0]
            neighbor_sim = neighbors_for_probe[neighbor_id][1]
            if neighbor_id != 0: yield neighbor_name, neighbor_sim


    def make_neighbors_table_fig(self, cat, num_neighbors=10, num_trunc_cols=5, is_titled=False):
        ##########################################################################
        # get col_labels
        col_labels = self.cat_probe_list_dict[cat]
        ##########################################################################
        # inits
        neighbors_mat_list = []
        col_labels_list = []
        ##########################################################################
        # make neighbors_mat_list and col_labels_list
        for i in range(0, len(col_labels), num_trunc_cols): # split col_labels into even sized lists
            truncated_col_labels = col_labels[i:i + num_trunc_cols]
            neighbors_mat = np.chararray((num_neighbors, num_trunc_cols), itemsize=20)
            neighbors_mat[:] = '' # initialize so that matplotlib can read table
            for probe_id, probe in enumerate(truncated_col_labels):
                generator = self.gen_neighbor_name_and_sim(probe)
                for neighbors_id in range(num_neighbors):
                    neighbor_name, neighbor_sim = next(generator)
                    neighbors_mat[neighbors_id, probe_id] = '{:>15} {:.2f}'.format(neighbor_name, neighbor_sim)
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_trunc_cols - len(truncated_col_labels)
            for i in range(length_diff): truncated_col_labels.append(' ')  # add space so table can be read properly
            col_labels_list.append(truncated_col_labels)
        ##########################################################################
        # fig settings
        num_tables = len(neighbors_mat_list)
        nrows, ncols = num_neighbors, num_trunc_cols
        hcell, wcell = 0.3, 2.5
        hpad, wpad = 0.1, 0
        figsize = (ncols * wcell + wpad, (nrows * hcell + hpad) * num_tables)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
        label_fontsize = 4
        linewidth = 2.0
        ##########################################################################
        # fig
        fig, axarr = plt.subplots(num_tables,1, figsize=tuple(figsize))
        fig_name = '{} Block {} Nearest Neighbors of words in {}'.format(self.model_name, self.block_name, cat)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # ax
        for n, neighbors_mat in enumerate(neighbors_mat_list):
            axarr[n].xaxis.set_visible(False) # hide to show table only
            axarr[n].yaxis.set_visible(False)
            axarr[n].axis('off')
            ##########################################################################
            # table sizing
            the_table = axarr[n].table(cellText=neighbors_mat, colLabels=col_labels_list[n], loc='center')
            table_props = the_table.properties()
            table_cells = table_props['child_artists']
            for cell in table_cells: cell.set_height(0.1)
        ##########################################################################
        return fig


    def make_acts_dh_fig(self, sel_probe=None, num_colors = 0, vmin=0.0, vmax=1.0, is_titled=False):
        ##########################################################################
        # make acts_df
        if sel_probe:
            token_acts_df = self.make_token_acts_df(sel_probe)
            acts_mat = np.asarray(token_acts_df)
        else:
            acts_mat = self.all_acts_df.values
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
        if sel_probe:
            fig_name = '{} Block {} DH of acts for "{}" '.format(self.model_name, self.block_name, sel_probe)
        else:
            fig_name = '{} Block {} DH of acts for all probes '.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size, y=1.2)
        ##########################################################################
        # axes
        ax_heatmap.yaxis.tick_right()
        ax_heatmap.set_xlabel('Hidden Units', fontsize=ax_font_size)
        num_acts = len(acts_mat)
        if sel_probe:
            ax_heatmap.set_ylabel('{} Examples of "{}"'.format(num_acts, sel_probe), fontsize=ax_font_size)
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
        # calc cat simmat
        cat_simmat = self.calc_cat_sim_mat()
        cat_simmat_labels = self.cat_list # TODO is this correct?
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
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               interpolation='nearest',
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax])
        cb.ax.set_xticklabels([vmin, vmax])
        cb.set_label('Correlation Coefficient', labelpad=-50, fontsize=ax_font_size)
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
        cat_acts_df = self.make_cat_acts_df(cat)
        probes_in_cat = cat_acts_df.index.tolist()
        ##########################################################################
        # dendrogram
        dist_matrix = pdist(cat_acts_df.values, 'euclidean')
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   leaf_label_func=lambda x: probes_in_cat[x],
                   orientation='left',
                   leaf_font_size=10)
        ##########################################################################
        return fig


    def make_ba_breakdown_fig(self, is_titled=False):
        ##########################################################################
        # make cat_probe_list_dict and cats_sorted_by_ba
        cats_sorted_by_ba, cat_ba_dict, token_ba_row = self.get_ba_breakdown_data()
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
            cat_probe_list = self.cat_probe_list_dict[cat]
            cat_probe_ids = [self.probe_list.index(e_token) for e_token in cat_probe_list]
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_row[cat_probe_ids]
            ax.plot(xs, ys, 'b.', alpha=1)
        ##########################################################################
        return fig


    def make_ba_breakdown_scatter_fig(self, is_titled=False):
        ##########################################################################
        # make cat_probe_list_dict and cats_sorted_by_ba
        cats_sorted_by_ba, cat_ba_dict, token_ba_row = self.get_ba_breakdown_data()
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
            cat_probe_list = self.cat_probe_list_dict[cat]
            cat_probe_ids = [self.probe_list.index(e_token) for e_token in cat_probe_list]
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_row[cat_probe_ids]
            ax.plot(xs, ys, 'b.', alpha=0) # this needs to be plot for annotation to work
            ##########################################################################
            # annotate points
            annotated_y_ints = []
            for x,y, target in sorted(zip(xs, ys, self.cat_probe_list_dict[cat])):
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

    
