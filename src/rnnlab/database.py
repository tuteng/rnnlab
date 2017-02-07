import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import rcParams
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
import pandas as pd


from dbutils import calc_probe_sim_mat, load_token_data, load_rnnlabrc


class DataBase:
    """
    Stores dataframe constructed during rnn training
    Calculates similarities between probes, and balanced accuracy, and can generate plots

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
        # load token data
        self.token_list, self.token_id_dict, self.probe_list, self.probe_id_dict,\
        self.probe_cat_dict, self.cat_list, self.cat_probe_list_dict, \
        self.probe_cf_traj_dict, self.num_train_doc_ids = load_token_data(self.model_name)


    def add_col(self,col_name, col_data):
        ##########################################################################
        assert len(col_data) == len(self.df)
        self.df[col_name] = col_data
        ##########################################################################
        print 'Added {} to dataframe'.format(col_name)


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



    def make_acts_2d_fig(self, sv_nums=(2,3), perplexity=30, label_probe=False, is_titled=False):
        ##########################################################################
        # choose seaborn style and palette
        import seaborn as sns
        sns.set_style('white')
        palette = np.array(sns.color_palette("hls", len(self.cat_list)))
        ##########################################################################
        # all_acts_df
        all_acts_df = self.make_all_acts_df()
        ##########################################################################
        # svd
        u, s, v = linalg.svd(all_acts_df.values)  # row_singular_vectors, singular_values, column_singular_vectors
        acts_2d_svd = u[:, sv_nums]
        ##########################################################################
        # tsne
        acts_2d_tsne = TSNE(perplexity=perplexity).fit_transform(all_acts_df.values)
        ##########################################################################
        # get cat for each probe for plotting
        acts_cats = [self.probe_cat_dict[probe] for probe in self.probe_list]
        ##########################################################################
        # fig settings
        figsize = (7, 3.25)
        title_font_size = 16
        label_fontsize = 6
        path_linewidth = 2.0
        markersize = 8
        ##########################################################################
        # fig
        fig, axarr = plt.subplots(1, 2, figsize=figsize)
        fig_name = '{} Block {} Acts Dimensionality Reduction'.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axis
        for n, x in enumerate([acts_2d_svd, acts_2d_tsne]):
            palette_ids = [self.cat_list.index(cat) for cat in acts_cats]
            axarr[n].scatter(x[:, 0], x[:, 1], lw=0, s=markersize, c=palette[palette_ids])
            axarr[n].axis('off')
            axarr[n].axis('tight')
            descr_str = ', '.join(['sv {}: var {:2.0f}%'.format(i, s[i]) for i in sv_nums])
            if is_titled: axarr[n].set_title(['SVD ({})'.format(descr_str), 't-SNE'][n], fontsize=title_font_size)
            ##########################################################################
            # add the labels for each cat
            for cat in self.cat_list:
                x_ids = np.where(np.asarray(acts_cats) == cat)[0]
                xtext, ytext = np.median(x[x_ids, :], axis=0)
                txt = axarr[n].text(xtext, ytext, str(cat), fontsize=label_fontsize,
                                    color=palette[self.cat_list.index(cat)])
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=path_linewidth, foreground="w"),
                    PathEffects.Normal()])
            ##########################################################################
            # add the labels for each probe
            if label_probe:
                for probe in self.probe_list:
                    x_ids = np.where(np.asarray(self.probe_list) == probe)[0]
                    xtext, ytext = np.median(x[x_ids, :], axis=0)
                    txt = axarr[n].text(xtext, ytext, str(probe), fontsize=label_fontsize)
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=path_linewidth, foreground="w"),
                        PathEffects.Normal()])
        ##########################################################################
        # layout
        fig.tight_layout()
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
        figsize = (9, 6)
        title_font_size = 16
        ax_font_size = 16
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} Hist of acts corcoeffs for "{}" '.format(self.model_name, self.block_name, probe)
        if is_titled: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # axes
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=ax_font_size)
        ax.set_ylabel('Number of observations of "{}'.format(probe), fontsize=ax_font_size)
        ax.hist(corr_mat, bins)
        ax.set_xlim([0, 1]) # this doesn't work well for block 0001
        ##########################################################################
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ##########################################################################
        return fig


    def gen_neighbor_name_and_sim(self, neighbors_for_probe, probe):
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


    def make_custom_neighbors_table_fig(self, sel_probes, num_neighbors=10, num_trunc_cols=3, is_titled=False):
        ##########################################################################
        # probe simmat
        probe_simmat = calc_probe_sim_mat(self.make_all_acts_df(), self.probe_list)
        ##########################################################################
        # inits
        neighbors_mat_list = []
        col_labels_list = []
        ##########################################################################
        # reformat simmat
        sim_tuples_list_list = []
        for row_id, row in enumerate(probe_simmat):
            sim_tuples_list_list.append([(target, sim) for target, sim in zip(self.probe_list, row)])
        ##########################################################################
        # make neighbors_mat_list and col_labels_list
        for i in range(0, len(sel_probes), num_trunc_cols):  # split sel_probes into even sized lists
            truncated_col_labels = sel_probes[i:i + num_trunc_cols]
            neighbors_mat = np.chararray((num_neighbors, num_trunc_cols), itemsize=20)
            neighbors_mat[:] = ''  # initialize so that matplotlib can read table
            for probe_id, probe in enumerate(truncated_col_labels):
                neighbors_for_probe = sorted(sim_tuples_list_list[self.probe_id_dict[probe]], key=itemgetter(1),
                                             reverse=True)
                generator = self.gen_neighbor_name_and_sim(neighbors_for_probe, probe)
                for neighbors_id in range(num_neighbors):
                    neighbor_name, neighbor_sim = next(generator)
                    neighbors_mat[neighbors_id, probe_id] = '{:>15} {:.2f}'.format(neighbor_name, neighbor_sim)
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_trunc_cols - len(truncated_col_labels)
            for i in range(length_diff): truncated_col_labels.append(' ')  # add space so table can be read properly
            col_labels_list.append(truncated_col_labels)
        ##########################################################################
        # fig settings
        title_font_size = 16
        table_fontsize = 6
        num_tables = len(neighbors_mat_list)
        figsize = (3.0, 4)
        ##########################################################################
        # fig
        fig, axarr = plt.subplots(num_tables, 1, figsize=tuple(figsize))
        fig_name = '{} Block {} Nearest Neighbors of words in Custom List'.format(self.model_name, self.block_name)
        if is_titled: fig.suptitle(fig_name, fontsize=title_font_size)
        ##########################################################################
        # ax
        for n, neighbors_mat in enumerate(neighbors_mat_list):
            axarr[n].axis('off')
            table_ = axarr[n].table(cellText=neighbors_mat_list[n], colLabels=col_labels_list[n],
                              loc='center', colWidths=[0.415] * num_trunc_cols)
            table_.auto_set_font_size(False)
            table_.set_fontsize(table_fontsize)
        ##########################################################################
        # layout
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)  # make space between subplots
        ##########################################################################
        return fig


    def make_neighbors_table_fig(self, cat, num_neighbors=10, num_trunc_cols=5, is_titled=True):
        ##########################################################################
        # probe simmat
        probe_simmat = calc_probe_sim_mat(self.make_all_acts_df(), self.probe_list)
        ##########################################################################
        # get col_labels
        col_labels = self.cat_probe_list_dict[cat]
        ##########################################################################
        # inits
        neighbors_mat_list = []
        col_labels_list = []
        ##########################################################################
        # reformat simmat
        sim_tuples_list_list = []
        for row_id, row in enumerate(probe_simmat):
            sim_tuples_list_list.append([(target, sim) for target, sim in zip(self.probe_list, row)])
        ##########################################################################
        # make neighbors_mat_list and col_labels_list
        for i in range(0, len(col_labels), num_trunc_cols): # split col_labels into even sized lists
            truncated_col_labels = col_labels[i:i + num_trunc_cols]
            neighbors_mat = np.chararray((num_neighbors, num_trunc_cols), itemsize=20)
            neighbors_mat[:] = '' # initialize so that matplotlib can read table
            for probe_id, probe in enumerate(truncated_col_labels):
                neighbors_for_probe = sorted(sim_tuples_list_list[self.probe_id_dict[probe]], key=itemgetter(1),
                                             reverse=True)
                generator = self.gen_neighbor_name_and_sim(neighbors_for_probe, probe)
                for neighbors_id in range(num_neighbors):
                    neighbor_name, neighbor_sim = next(generator)
                    neighbors_mat[neighbors_id, probe_id] = '{:>15} {:.2f}'.format(neighbor_name, neighbor_sim)
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_trunc_cols - len(truncated_col_labels)
            for i in range(length_diff): truncated_col_labels.append(' ')  # add space so table can be read properly
            col_labels_list.append(truncated_col_labels)
        ##########################################################################
        # fig settings
        title_font_size = 16
        num_tables = len(neighbors_mat_list)
        nrows, ncols = num_neighbors, num_trunc_cols
        hcell, wcell = 0.3, 2.5
        hpad, wpad = 0.1, 0
        figsize = (ncols * wcell + wpad, (nrows * hcell + hpad) * num_tables)
        ##########################################################################
        # fig
        fig, axarr = plt.subplots(num_tables,1, figsize=tuple(figsize))
        fig_name = '{} Block {} Nearest Neighbors of words in {}'.format(self.model_name, self.block_name, cat)
        if is_titled: fig.suptitle(fig_name, fontsize=title_font_size)
        ##########################################################################
        # ax
        for n, neighbors_mat in enumerate(neighbors_mat_list):
            axarr[n].xaxis.set_visible(False) # hide to show table only
            axarr[n].yaxis.set_visible(False)
            axarr[n].axis('off')
            ##########################################################################
            # table sizing
            table_ = axarr[n].table(cellText=neighbors_mat, colLabels=col_labels_list[n], loc='center')
            table_props = table_.properties()
            table_cells = table_props['child_artists']
            for cell in table_cells: cell.set_height(0.1)
        ##########################################################################
        # layout
        fig.tight_layout()
        if is_titled: fig.subplots_adjust(top=0.925) # make room for main title
        fig.subplots_adjust(hspace=0.2) # make sapce between subplots
        ##########################################################################
        return fig


    def make_acts_dh_fig(self, probe=None, num_colors = 0, vmin=0.0, vmax=1.0, is_titled=False):
        ##########################################################################
        print 'Making custom probe activations dh with vmin : {}'.format(vmin)
        ##########################################################################
        # make acts_df
        if probe:
            token_acts_df = self.make_token_acts_df(probe)
            acts_mat = np.asarray(token_acts_df)
        else:
            acts_mat = self.make_all_acts_df().values
        ##########################################################################
        print 'Acts mat | max: {:.2} min: {:.2}'.format(np.max(acts_mat), np.min(acts_mat))
        ##########################################################################
        # fig settings
        figsize = (9, 6)
        title_font_size = 16
        ax_font_size = 16
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


    def make_cat_sim_dh_fig(self, num_colors = 0, vmin=0.0, vmax=1.0, y_title=False, is_titled=False):
        ##########################################################################
        # calc cat simmat
        cat_simmat = self.calc_cat_sim_mat()
        cat_simmat_labels = self.cat_list
        ##########################################################################
        # fig settings
        figsize = (3.40,3.40) # 12, 8
        title_font_size = 16
        ax_font_size = 8
        label_fontsize = 6
        ##########################################################################
        # fig
        fig, ax_heatmap = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} DH of cat simmat '.format(self.model_name, self.block_name)
        if is_titled: plt.title(fig_name, fontsize=title_font_size, y=1.03)
        ##########################################################################
        # axes
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_denright = divider.append_axes("right", 0.5, pad=0.0, sharey=ax_heatmap)
        ax_denright.set_frame_on(False)
        ax_colorbar = divider.append_axes("top", 0.1, pad=0.2)
        ##########################################################################
        # dendrogram
        lnk0 = linkage(pdist(cat_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_denright,
                         orientation='right',
                         color_threshold=left_threshold,
                         no_labels=True)
        ##########################################################################
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = cat_simmat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        ##########################################################################
        # heatmap
        max_extent = ax_denright.get_ylim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               interpolation='nearest',
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=ax_font_size)
        cb.set_label('Correlation Coefficient', labelpad=-8, fontsize=ax_font_size)
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
        if y_title:
            ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, nrows))
            ax_heatmap.yaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_denright.xaxis.get_ticklines() +
                 ax_denright.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=label_fontsize)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=label_fontsize)
        # make dendrogram labels invisible
        plt.setp(ax_denright.get_yticklabels() + ax_denright.get_xticklabels(),
                 visible=False)
        ##########################################################################
        # layout
        fig.subplots_adjust(bottom=0.2) # make room for tick labels
        fig.tight_layout()
        ##########################################################################
        return fig


    def make_cat_cluster_fig(self, cat, xlim=False, bottom_off=True, num_pobes_limit=20, is_titled=False):
        ##########################################################################
        # get cat_acts_mat
        cat_acts_df = self.make_cat_acts_df(cat)
        probes_in_cat = cat_acts_df.index.tolist()
        num_probes_in_cat = len(probes_in_cat)
        ##########################################################################
        # limit data (to fit into small publication fig)
        if num_pobes_limit and num_probes_in_cat > num_pobes_limit:
            ids = np.random.choice(range(num_probes_in_cat), num_pobes_limit, replace=False)
            cat_acts_df = cat_acts_df.iloc[ids]
            probes_in_cat = [probes_in_cat[id] for id in ids]
        ##########################################################################
        # fig settings
        figsize = (3.2, 3.2) # 8,8
        title_font_size = 16
        rcParams['lines.linewidth'] = 2.0
        leaf_font_size = 8
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} Clustering of {}'.format(self.model_name, self.block_name, cat)
        if is_titled: fig.suptitle(fig_name, fontsize=title_font_size)
        ##########################################################################
        # dendrogram
        dist_matrix = pdist(cat_acts_df.values, 'euclidean')
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   ax=ax,
                   leaf_label_func=lambda x: probes_in_cat[x],
                   orientation='right',
                   leaf_font_size=leaf_font_size)
        ##########################################################################
        # axis
        ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if bottom_off:
            ax.xaxis.set_ticklabels([]) # hides ticklabels
            ax.tick_params(axis='both', which='both', bottom='off')
            ax.spines['bottom'].set_visible(False)
        if xlim: ax.set_xlim([0, xlim])
        ##########################################################################
        # layout
        fig.tight_layout()
        if is_titled: fig.subplots_adjust(top=0.925) # make room for main title
        ##########################################################################
        return fig


    def make_custom_cat_clust_fig(self, cats):
        ##########################################################################
        # fig settings
        figsize = (6, 6)
        title_font_size = 16
        leaf_font_size = 6
        rcParams['lines.linewidth'] = 2.0
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        fig_name = '{} Block {} Custom Clustering'.format(self.model_name, self.block_name)
        if False: plt.title(fig_name, fontsize=title_font_size)
        ##########################################################################
        # make cat_acts_mat
        cats_acts_mat_list = []
        cats_probe_list = []
        for cat in cats:
            cat_acts_df = self.make_cat_acts_df(cat)
            cats_acts_mat_list.append(cat_acts_df.values)
            cats_probe_list += cat_acts_df.index.tolist()
        cat_acts_mat = np.vstack((mat for mat in cats_acts_mat_list))
        ##########################################################################
        # dendrogram
        dist_matrix = pdist(cat_acts_mat, 'euclidean')
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   ax=ax,
                   labels = cats_probe_list,
                   orientation='right',
                   leaf_font_size=leaf_font_size)
        ##########################################################################
        # axis
        ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ##########################################################################
        return fig


    def make_ba_breakdown_fig(self, is_titled=False):
        ##########################################################################
        # get_ba_breakdown_data
        cats_sorted_by_ba, cat_ba_dict, token_ba_list = self.get_ba_breakdown_data()
        ##########################################################################
        # fig settings
        figsize = (12, 8)
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
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_list[cat_probe_ids]
            ax.plot(xs, ys, 'b.', alpha=1)
        ##########################################################################
        return fig


    def make_ba_breakdown_scatter_fig(self, is_titled=False):
        ##########################################################################
        # make cat_probe_list_dict and cats_sorted_by_ba
        cats_sorted_by_ba, cat_ba_dict, token_ba_list = self.get_ba_breakdown_data()
        ##########################################################################
        # fig settings
        figsize = (12, 8)
        title_font_size = 16
        ax_font_size = 16
        leg_font_size = 10
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
            xs, ys = [cat_id for i in range(len(cat_probe_ids))], token_ba_list[cat_probe_ids]
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

    

    def make_cat_confusion_mat_fig(self):
        ##########################################################################
        import seaborn as sns
        sns.set_style('white')
        ##########################################################################
        # get data
        runs_dir = load_rnnlabrc('runs_dir')
        path = os.path.join(runs_dir, self.model_name, 'Balanced_Accuracy')
        file_name = 'cat_confusion_mat_data_block_{}.npz'.format(self.block_name)
        npzfile = np.load(os.path.join(path, file_name))
        hits_by_cat_dict = npzfile['hits_by_cat_dict'].item()
        fas_by_cat_dict = npzfile['fas_by_cat_dict'].item()
        num_cats = len(fas_by_cat_dict)
        cats = sorted(fas_by_cat_dict.keys())
        ##########################################################################
        # make confusion mat ( each element normalized by tot number of combinations)
        cat_confusion_mat = np.zeros((num_cats, num_cats), dtype=float)
        for row_id, row_cat in enumerate(cats):
            for col_id, col_cat in enumerate(cats):
                num_probes_row_cat = len(self.cat_probe_list_dict[row_cat])
                num_probes_col_cat = len(self.cat_probe_list_dict[col_cat])
                n = num_probes_row_cat * num_probes_col_cat - num_probes_row_cat
                if row_id == col_id:  # hits
                    hits = float(hits_by_cat_dict[row_cat][col_cat])
                    cat_confusion_mat[row_id, col_id] = hits / n *100
                else:  # fas
                    fas = float(fas_by_cat_dict[row_cat][col_cat])
                    cat_confusion_mat[row_id, col_id] = fas / n *100
        ##########################################################################
        # fig settings
        figsize = (10, 10)
        ##########################################################################
        # fig
        fig, ax = plt.subplots(figsize=figsize)
        ##########################################################################
        # mask
        mask = np.zeros_like(cat_confusion_mat, dtype=np.bool)
        mask[np.triu_indices_from(mask, 1)] = True
        ##########################################################################
        # plot
        sns.heatmap(cat_confusion_mat.astype(np.int), ax=ax, square=True, annot=True,
                    annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                    vmin=0, vmax=100, cmap='jet', mask=mask, fmt='d')
        ##########################################################################
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 50, 100])
        cbar.set_ticklabels(['0%', '50%', '100%'])
        ##########################################################################
        # ax (needs to be below plot for axes to be labeled)
        ax.set_yticklabels(sorted(cats, reverse=True), rotation=0)
        ax.set_xticklabels(cats, rotation=90)
        for t in ax.texts: t.set_text(t.get_text() + "%")
        ##########################################################################
        # layout
        plt.tight_layout()
        ##########################################################################
        return fig