import scipy
from operator import itemgetter
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from bokeh.models import Range1d
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.charts import Bar
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from itertools import groupby


from utils import score_2_neighbor_rankings
from utils import make_cat_conf_mat
from utils import calc_probe_sim_mat
from utils import calc_cat_sim_mat
from utils import gen_neighbor_name_and_sim
from utils import calc_num_probe_acts_clusters
from utils import plot_best_fit_line
from utils import block_to_mb
from utils import human_format

from database import load_rnnlabrc
runs_dir = load_rnnlabrc('runs_dir')

np.set_printoptions(suppress=True, precision=2, threshold=100)


def make_acts_2d_fig(database, sv_nums=(2, 3), perplexity=30, label_probe=False, is_subtitled=True):
    ##########################################################################
    # load data
    probes_acts_df = database.get_probes_acts_df(num_ba_samples=0)
    u, s, v = linalg.svd(probes_acts_df.values)  # row_singular_vectors, singular_values, column_singular_vectors
    acts_2d_svd = u[:, sv_nums]
    acts_2d_tsne = TSNE(perplexity=perplexity).fit_transform(probes_acts_df.values)
    acts_cats = [database.probe_cat_dict[probe] for probe in database.probe_list]
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns
    sns.set_style('white')
    palette = np.array(sns.color_palette("hls", len(database.cat_list)))
    ##########################################################################
    # fig
    figsize = (6, 10)
    title_font_size = 16
    label_fontsize = 6
    path_linewidth = 2.0
    markersize = 8
    fig, axarr = plt.subplots(2, 1, figsize=figsize)
    ##########################################################################
    # axis
    for n, x in enumerate([acts_2d_svd, acts_2d_tsne]):
        palette_ids = [database.cat_list.index(cat) for cat in acts_cats]
        axarr[n].scatter(x[:, 0], x[:, 1], lw=0, s=markersize, c=palette[palette_ids])
        axarr[n].axis('off')
        axarr[n].axis('tight')
        descr_str = ', '.join(['sv {}: var {:2.0f}%'.format(i, s[i]) for i in sv_nums])
        if is_subtitled: axarr[n].set_title(['SVD ({})'.format(descr_str), 't-SNE'][n], fontsize=title_font_size)
        ##########################################################################
        # add the labels for each cat
        for cat in database.cat_list:
            x_ids = np.where(np.asarray(acts_cats) == cat)[0]
            xtext, ytext = np.median(x[x_ids, :], axis=0)
            txt = axarr[n].text(xtext, ytext, str(cat), fontsize=label_fontsize,
                                color=palette[database.cat_list.index(cat)])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=path_linewidth, foreground="w"), PathEffects.Normal()])
        ##########################################################################
        # add the labels for each probe
        if label_probe:
            for probe in database.probe_list:
                x_ids = np.where(np.asarray(database.probe_list) == probe)[0]
                xtext, ytext = np.median(x[x_ids, :], axis=0)
                txt = axarr[n].text(xtext, ytext, str(probe), fontsize=label_fontsize)
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=path_linewidth, foreground="w"), PathEffects.Normal()])
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_token_acts_avg_act_corr_fig(databases, probe):
    ##########################################################################
    # load data
    cat = databases[-1].probe_cat_dict[probe]
    last_avg_cat_act = np.mean(databases[-1].get_cat_acts_df(cat).values, axis=0)
    last_avg_token_act = np.mean(databases[-1].get_probe_acts_df(probe).values, axis=0)
    num_token_acts = len(databases[-1].get_probe_acts_df(probe))
    traj_mat_ax0 = np.zeros((num_token_acts, len(databases)))
    traj_mat_ax1 = np.zeros((num_token_acts, len(databases)))
    for n, database in enumerate(databases):
        token_acts_mat = database.get_probe_acts_df(probe).values
        traj_mat_ax0[:, n] = [np.corrcoef(token_act, last_avg_token_act)[1, 0] for token_act in token_acts_mat]
        traj_mat_ax1[:, n] = [np.corrcoef(token_act, last_avg_cat_act)[1, 0] for token_act in token_acts_mat]
    xys_ax0, xys_ax1 = [], []
    for row_ax0, row_ax1 in zip(traj_mat_ax0, traj_mat_ax1):
        x = databases[-1].get_mbs_axis()
        y = row_ax0
        xys_ax0.append((x, y))
        y = row_ax1
        xys_ax1.append((x, y))
    ##########################################################################
    # fig 
    figsize = (6, 4)
    ax_font_size = 8
    fig, axarr = plt.subplots(2, 1, figsize=figsize)
    ##########################################################################
    # axes
    for n, ax in enumerate(axarr):
        if n == 0:
            ax.set_ylabel('Corr with last avg act of "{}"'.format(probe), fontsize=ax_font_size)
        else:
            ax.set_ylabel('Corr with last avg act of {}'.format(cat), fontsize=ax_font_size)
        ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.set_ylim([0, 1])
    ##########################################################################
    # plot
    for (x, y) in xys_ax0:
        axarr[0].plot(x, y, '-')
    for (x, y) in xys_ax1:
        axarr[1].plot(x, y, '-')
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_custom_neighbors_table_fig(database, probes, num_neighbors=10,
                                    num_trunc_cols=3):
    ##########################################################################
    # load data
    probes_acts_df = database.get_probes_acts_df()
    probe_simmat = calc_probe_sim_mat(probes_acts_df)
    ##########################################################################
    # make neighbors_mat
    neighbors_mat_list = []
    col_labels_list = []
    for i in range(0, len(probes), num_trunc_cols):  # split probes into even sized lists
        truncated_col_labels = probes[i:i + num_trunc_cols]
        neighbors_mat = np.chararray((num_neighbors, num_trunc_cols), itemsize=20)
        neighbors_mat[:] = ''  # initialize so that mpl can read table
        ##########################################################################
        # make column
        for col_id, probe in enumerate(truncated_col_labels):
            probe_id = database.probe_id_dict[probe]
            neighbor_tuples_list = [(probe_, sim) for probe_, sim in zip(database.probe_list, probe_simmat[probe_id])
                                    if probe_ != probe]
            neighbor_tuples = sorted(neighbor_tuples_list, key=itemgetter(1), reverse=True)[:num_neighbors]
            neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                 for tuple in neighbor_tuples if tuple[0] != probe]
            neighbors_mat[:, col_id] = neighbors_mat_col
        ##########################################################################
        # collect info for plotting
        neighbors_mat_list.append(neighbors_mat)
        length_diff = num_trunc_cols - len(truncated_col_labels)
        for i in range(length_diff): truncated_col_labels.append(' ')  # add space so table can be read properly
        col_labels_list.append(truncated_col_labels)
    ##########################################################################
    # fig
    table_fontsize = 6
    num_tables = len(neighbors_mat_list)
    figsize = (3.0, 4)
    fig, axarr = plt.subplots(num_tables, 1, figsize=figsize)
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


def make_neighbors_table_fig(database, cat, num_neighbors=10, num_trunc_cols=5):
    ##########################################################################
    # load data
    probes_acts_df = database.get_probes_acts_df()
    probe_simmat = calc_probe_sim_mat(probes_acts_df)
    probes = database.cat_probe_list_dict[cat]
    ##########################################################################
    # make neighbors_mat
    neighbors_mat_list = []
    col_labels_list = []
    for i in range(0, len(probes), num_trunc_cols):  # split probes into even sized lists
        truncated_col_labels = probes[i:i + num_trunc_cols]
        neighbors_mat = np.chararray((num_neighbors, num_trunc_cols), itemsize=20)
        neighbors_mat[:] = ''  # initialize so that mpl can read table
        ##########################################################################
        # make column
        for col_id, probe in enumerate(truncated_col_labels):
            probe_id = database.probe_id_dict[probe]
            neighbor_tuples_list = [(probe_, sim) for probe_, sim in zip(database.probe_list, probe_simmat[probe_id])
                                    if probe_ != probe]
            neighbor_tuples = sorted(neighbor_tuples_list, key=itemgetter(1), reverse=True)[:num_neighbors]
            neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                 for tuple in neighbor_tuples if tuple[0] != probe]
            neighbors_mat[:, col_id] = neighbors_mat_col
        ##########################################################################
        # collect info for plotting
        neighbors_mat_list.append(neighbors_mat)
        length_diff = num_trunc_cols - len(truncated_col_labels)
        for i in range(length_diff): truncated_col_labels.append(' ')  # add space so table can be read properly
        col_labels_list.append(truncated_col_labels)
    ##########################################################################
    # fig
    num_tables = len(neighbors_mat_list)
    nrows, ncols = num_neighbors, num_trunc_cols
    hcell, wcell = 0.3, 2.5
    hpad, wpad = 0.1, 0
    figsize = (ncols * wcell + wpad, (nrows * hcell + hpad) * num_tables)
    fig, axarr = plt.subplots(num_tables, 1, figsize=figsize)
    fig.suptitle('Neighbors for probes in {}'.format(cat))
    ##########################################################################
    # ax
    for n, neighbors_mat in enumerate(neighbors_mat_list):
        axarr[n].xaxis.set_visible(False)  # hide to show table only
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
    fig.subplots_adjust(top=0.925)  # make room for main title
    fig.subplots_adjust(hspace=0.2)  # make sapce between subplots
    ##########################################################################
    return fig


def make_acts_dh_fig(database, probe=None, max_num_acts=1000):
    ##########################################################################
    # make probes_acts_df
    if probe:
        probe_acts_df = database.get_probe_acts_df(probe)
        acts_mat = probe_acts_df.values[:max_num_acts]
    else:
        probes_acts_df = database.get_probes_acts_df()
        acts_mat = probes_acts_df.values
    ##########################################################################
    vmin, vmax = np.min(acts_mat), np.max(acts_mat)
    print 'Acts mat | min: {:.2} max: {:.2}'.format(vmin, vmax)
    ##########################################################################
    # fig
    figsize = (6, 5)
    ax_font_size = 8
    fig, ax_heatmap = plt.subplots(figsize=figsize)
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
    print acts_mat.shape
    lnk0 = linkage(pdist(acts_mat))
    dg0 = dendrogram(lnk0,
                     ax=ax_dendleft,
                     orientation='right',
                     color_threshold=-1,
                     no_labels=True)
    # top dendrogram
    lnk1 = linkage(pdist(acts_mat.T))
    dg1 = dendrogram(lnk1,
                     ax=ax_dendtop,
                     color_threshold=-1,
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
    cb.set_label('Strength of Activation', labelpad=-75, fontsize=ax_font_size)
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


def make_cat_sim_dh_fig(database, num_colors=None, y_title=False):
    ##########################################################################
    # load data
    cat_simmat = calc_cat_sim_mat(database)
    cat_simmat_labels = database.cat_list
    vmin, vmax = 0.5, 1.0
    assert not vmin > np.min
    ##########################################################################
    print 'Cat simmat | min: {} max {}'.format(np.min(cat_simmat), np.max(cat_simmat))
    ##########################################################################
    # fig
    figsize = (3.40, 3.40)
    ax_font_size = 8
    label_fontsize = 6
    fig, ax_heatmap = plt.subplots(figsize=figsize)
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
    fig.subplots_adjust(bottom=0.2)  # make room for tick labels
    fig.tight_layout()
    ##########################################################################
    return fig


def make_cat_cluster_fig(database, cat, xlim=False, bottom_off=False, num_probes_limit=20):
    ##########################################################################
    # load data
    cat_prototypes_df = database.get_cat_acts_df(cat)
    probes_in_cat = cat_prototypes_df.index.tolist()
    num_probes_in_cat = len(probes_in_cat)
    if num_probes_limit and num_probes_in_cat > num_probes_limit:
        ids = np.random.choice(range(num_probes_in_cat), num_probes_limit, replace=False)
        cat_prototypes_df = cat_prototypes_df.iloc[ids]
        probes_in_cat = [probes_in_cat[id] for id in ids]
    ##########################################################################
    # fig settings
    figsize = (6, 6)
    rcParams['lines.linewidth'] = 2.0
    leaf_font_size = 10
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # dendrogram
    dist_matrix = pdist(cat_prototypes_df.values, 'euclidean')
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
        ax.xaxis.set_ticklabels([])  # hides ticklabels
        ax.tick_params(axis='both', which='both', bottom='off')
        ax.spines['bottom'].set_visible(False)
    if xlim: ax.set_xlim([0, xlim])
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_custom_cat_clust_fig(database, cats):
    ##########################################################################
    # fig settings
    figsize = (6, 6)
    leaf_font_size = 6
    rcParams['lines.linewidth'] = 2.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # make cat_acts_mat
    cats_acts_mat_list = []
    cats_probe_list = []
    for cat in cats:
        cat_prototypes_df = database.get_cat_acts_df(cat)
        cats_acts_mat_list.append(cat_prototypes_df.values)
        cats_probe_list += cat_prototypes_df.index.tolist()
    cat_acts_mat = np.vstack((mat for mat in cats_acts_mat_list))
    ##########################################################################
    # dendrogram
    dist_matrix = pdist(cat_acts_mat, 'euclidean')
    linkages = linkage(dist_matrix, method='complete')
    dendrogram(linkages,
               ax=ax,
               labels=cats_probe_list,
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


def make_pairplot_fig(database):
    ##########################################################################
    # load data
    avg_probe_ba_list = database.get_avg_probe_ba_list()
    avg_probe_pp_list = database.get_avg_probe_pp_list()
    probe_cf_list = [database.probe_cf_traj_dict[probe][-1] for probe in database.probe_list]
    df = pd.DataFrame(data={'avg_probe_ba': avg_probe_ba_list,
                            'avg_probe_pp': avg_probe_pp_list,
                            'probe_cf': probe_cf_list})
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # plot
    fig = sns.pairplot(df, kind="reg")
    ##########################################################################
    # axis
    for ax in fig.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=-90)
    ##########################################################################
    return fig


def make_ba_vs_pp_fig(database):
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # load data
    avg_probe_ba_list = database.get_avg_probe_ba_list()
    avg_probe_pp_list = database.get_avg_probe_pp_list(clip=True)
    probe_cat_list = [database.probe_cat_dict[probe] for probe in database.probe_list]
    probe_cat_id_list = [database.cat_list.index(cat) for cat in probe_cat_list]
    xys = []
    for cat_id, cat in enumerate(database.cat_list):
        y = np.asarray(avg_probe_ba_list)[np.asarray(probe_cat_id_list) == cat_id]
        x = np.asarray(avg_probe_pp_list, dtype=np.int)[np.asarray(probe_cat_id_list) == cat_id]
        xys.append((x, y, cat))
    ##########################################################################
    # fig
    figsize = (12, 8)
    markersize = 10
    ax_font_size = 8
    leg_fontsize = 8
    num_cats = len(database.cat_list)
    subplot_rows = int(np.sqrt(num_cats))
    subplot_cols = subplot_rows + 1
    fig, axarr = plt.subplots(subplot_rows, subplot_cols, figsize=figsize, sharex='all', sharey='all')
    fig.text(0.5, 0.05, 'Avg probe Perplexity', ha='center', va='center')
    fig.text(0.05, 0.5, 'Avg Probe Balanced Accuracy', ha='center', va='center', rotation='vertical')
    ##########################################################################
    cat_id = 0
    for row_id in range(subplot_rows):
        for col_id in range(subplot_cols):
            ##########################################################################
            # axis
            axarr[row_id, col_id].margins(0.2)
            axarr[row_id, col_id].set_ylim([0, 100])
            axarr[row_id, col_id].spines['right'].set_visible(False)
            axarr[row_id, col_id].spines['top'].set_visible(False)
            axarr[row_id, col_id].tick_params(axis='both', which='both', top='off', right='off')
            axarr[row_id, col_id].set_xticks([0, database.num_input_units])
            axarr[row_id, col_id].set_xticklabels(['0', str(database.num_input_units)],
                                                  minor=False, fontsize=ax_font_size, rotation=90)
            ##########################################################################
            if not cat_id == len(xys):
                x, y, cat = xys[cat_id]
                cat_id += 1
            else:
                break
            ##########################################################################
            # plot
            axarr[row_id, col_id].scatter(x, y, s=markersize, zorder=2)
            axarr[row_id, col_id].axhline(y=50, linestyle='--', zorder=1, c='grey', linewidth=1.0)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            axarr[row_id, col_id].text(0.05, 0.9, cat, transform=axarr[row_id, col_id].transAxes,
                                       fontsize=leg_fontsize, verticalalignment='bottom', bbox=props)
            ##########################################################################
            # best fit line
            xys_ = zip(x, y)
            plot_best_fit_line(axarr[row_id, col_id], xys_, leg_fontsize, x_pos=0.05, y_pos=0.1, zorder=3)
    ##########################################################################
    # layout
    fig.subplots_adjust(bottom=0.15)
    ##########################################################################
    return fig


def make_ba_breakdown_fig(database):
    ##########################################################################
    # load data
    cats_sorted_by_ba, cat_ba_dict, avg_probe_ba_list = database.get_ba_breakdown_data()
    xys = []
    for n, cat in enumerate(cats_sorted_by_ba):
        cat_probe_list = database.cat_probe_list_dict[cat]
        cat_probe_ids = [database.probe_list.index(e_token) for e_token in cat_probe_list]
        xs = [n + 0.5] * len(cat_probe_list)
        ys = np.asarray(avg_probe_ba_list)[cat_probe_ids]
        markersizes = [database.probe_cf_traj_dict[probe][-1] for probe in cat_probe_list]
        xys.append((xs, ys, markersizes))
    ##########################################################################
    # fig
    figsize = (12, 8)
    ax_font_size = 8
    leg_fontsize = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axes
    ax.set_xlabel('Categories', fontsize=ax_font_size)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
    ax.set_xticks(np.arange(len(database.probe_list)) + 0.5, minor=False)
    ax.set_xticklabels(cats_sorted_by_ba, minor=False, fontsize=ax_font_size, rotation=90)
    ax.set_xlim([0, len(database.cat_list) + 0.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # plot
    for (x, y, markersize) in xys:
        ax.scatter(x, y, s=markersize, edgecolor='white', facecolor='black', zorder=2)
    mean_ba = np.nanmean(cat_ba_dict.values())
    ax.axhline(y=mean_ba, alpha=0.5, linestyle='--', c='grey', zorder=1)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, 'markersize ~ probe frequency', transform=ax.transAxes, fontsize=leg_fontsize,
             verticalalignment='bottom', bbox=props)
    ##########################################################################
    return fig


def make_ba_breakdown_annotated_fig(database):
    ##########################################################################
    # make cat_probe_list_dict and cats_sorted_by_ba
    cats_sorted_by_ba, cat_ba_dict, avg_probe_ba_list = database.get_ba_breakdown_data()
    ##########################################################################
    # fig
    figsize = (12, 8)
    ax_font_size = 8
    leg_font_size = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axes
    ax.set_xlabel('Categories', fontsize=ax_font_size)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
    ax.set_xticks(np.arange(len(database.probe_list)) + 0.5, minor=False)
    ax.set_xticklabels(cats_sorted_by_ba, minor=False, fontsize=leg_font_size, rotation=90)
    ax.set_xlim([0, len(database.cat_list) + 0.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # mean line
    mean_ba = np.nanmean(cat_ba_dict.values())
    ax.axhline(y=mean_ba, alpha=0.5, c='grey', linestyle='--', zorder=1)
    ##########################################################################
    # plot
    annotated_y_ints_long_words_prev_cat = []
    for cat_id, cat in enumerate(cats_sorted_by_ba):
        ##########################################################################
        # plot points
        annotated_y_ints_long_words_curr_cat = []
        cat_probe_list = database.cat_probe_list_dict[cat]
        cat_probe_ids = [database.probe_list.index(e_token) for e_token in cat_probe_list]
        xs, ys = [cat_id for i in range(len(cat_probe_ids))], np.asarray(avg_probe_ba_list)[cat_probe_ids]
        ax.plot(xs, ys, 'b.', alpha=0)  # this needs to be plot for annotation to work
        ##########################################################################
        # annotate points
        annotated_y_ints = []
        for x, y, probe in sorted(zip(xs, ys, database.cat_probe_list_dict[cat])):
            y_int = int(y)
            # if annotation coordinate exists or is affected by long word from previous cat, skip to next probe
            if not y_int in annotated_y_ints and y_int not in annotated_y_ints_long_words_prev_cat:
                plt.annotate(probe, xy=(x, y_int), xytext=(2, 0), textcoords='offset points', va='bottom',
                             fontsize=7)
                annotated_y_ints.append(y_int)
                if len(probe) > 7:
                    annotated_y_ints_long_words_curr_cat.append(y_int)
        annotated_y_ints_long_words_prev_cat = annotated_y_ints_long_words_curr_cat
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_cat_confusion_mat_fig(database):
    ##########################################################################
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # load data
    cat_conf_mat, mask = make_cat_conf_mat(database)
    cat_list = database.cat_list
    ##########################################################################
    # fig
    figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # plot
    sns.heatmap(cat_conf_mat.astype(np.int), ax=ax, square=True, annot=False,
                annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                vmin=0, vmax=100, cmap='jet', mask=mask, fmt='d')
    ##########################################################################
    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    cbar.set_label('Hits & False Alarms')
    ##########################################################################
    # ax (needs to be below plot for axes to be labeled)
    ax.set_yticklabels(sorted(cat_list, reverse=True), rotation=0)
    ax.set_xticklabels(cat_list, rotation=90)
    for t in ax.texts: t.set_text(t.get_text() + "%")
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_cat_conf_diff_fig(databases):
    ##########################################################################
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # load data
    cat_conf_mat_list = []
    mask = None
    for database in databases:
        cat_conf_mat, mask = make_cat_conf_mat(database)
        cat_conf_mat_list.append(cat_conf_mat)
    cat_list = databases[0].cat_list
    cat_conf_mat_diff = np.subtract(*cat_conf_mat_list)
    ##########################################################################
    # fig
    figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # plot
    sns.heatmap(cat_conf_mat_diff.astype(np.int), ax=ax, square=True, annot=False,
                annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                vmin=0, vmax=100, cmap='jet', mask=mask, fmt='d')
    ##########################################################################
    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    cbar.set_label('Hits & FAs Difference')
    ##########################################################################
    # ax (needs to be below plot for axes to be labeled)
    ax.set_yticklabels(sorted(cat_list, reverse=True), rotation=0)
    ax.set_xticklabels(cat_list, rotation=90)
    for t in ax.texts: t.set_text(t.get_text() + "%")
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_avg_probe_ba_trajs_fig(database, probes):
    ##########################################################################
    # load data
    avg_probe_ba_trajs_mat = database.get_trajs_mat(probes, 'avg_probe_ba')
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns
    sns.set_style('white')
    palette = iter(sns.color_palette("hls", len(probes)).as_hex())
    ##########################################################################
    # fig settings
    figsize = (600, 200)  # bokeh is  in pixels
    linewidth = 2.0
    ##########################################################################
    # fig
    hover = HoverTool(tooltips=[('batch', '@x'), ('probe', '@probe'), ('balAcc', '@y')])
    fig = figure(plot_width=figsize[0], plot_height=figsize[1],
                 tools=[hover, 'pan, wheel_zoom, crosshair, save'])
    ##########################################################################
    # axis
    fig.y_range = Range1d(0, 100)
    ##########################################################################
    # plot
    x = database.get_mbs_axis()
    for n, y in enumerate(avg_probe_ba_trajs_mat):
        source = ColumnDataSource(data=dict(x=x, y=y, probe=[probes[n]] * len(x)))
        fig.line(x='x', y='y', line_color=next(palette), line_width=linewidth, source=source)
    ##########################################################################
    return fig


def make_avg_probe_pp_trajs_fig(database, probes):
    ##########################################################################
    # load data
    avg_probe_pp_trajs_mat = database.get_trajs_mat(probes, 'avg_probe_pp')
    max_y = np.mean(avg_probe_pp_trajs_mat[:, 0])
    avg_probe_pp_trajs_mat = np.clip(avg_probe_pp_trajs_mat, 1, max_y)
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns
    sns.set_style('white')
    palette = iter(sns.color_palette("hls", len(probes)).as_hex())
    ##########################################################################
    # fig settings
    figsize = (600, 200)  # bokeh is  in pixels
    linewidth = 2.0
    ##########################################################################
    # fig
    hover = HoverTool(tooltips=[('batch', '@x'), ('probe', '@probe'), ('perp', '@y')])
    fig = figure(plot_width=figsize[0], plot_height=figsize[1],
                 tools=[hover, 'pan, wheel_zoom, crosshair, save'])
    ##########################################################################
    # axis
    fig.y_range = Range1d(0, max_y)
    ##########################################################################
    # plot
    x = database.get_mbs_axis()
    for n, y in enumerate(avg_probe_pp_trajs_mat):
        source = ColumnDataSource(data=dict(x=x, y=y, probe=[probes[n]] * len(x)))
        fig.line(x='x', y='y', line_color=next(palette), line_width=linewidth, source=source)
    ##########################################################################
    return fig


def make_cfreq_traj_fig(database, probes):
    ##########################################################################
    # load data
    probe_cf_traj_dict = database.probe_cf_traj_dict
    xys = []
    for probe in probes:
        x = database.get_mbs_axis()[1:]
        y = probe_cf_traj_dict[probe][:len(x)] # y does not take iterations into account
        if x:
            last_y, last_x = y[-1], x[-1]
        else:
            last_y, last_x = 0, 0  # in case x is empty
        xys.append((x, y, last_x, last_y, probe))
    y_thr = np.max([xy[3] for xy in xys])/2
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns 
    sns.set_style('white')
    palette = iter(sns.color_palette("hls", len(probes)))
    ##########################################################################
    # fig settings
    figsize = (6,3)
    ax_font_size = 8
    leg_font_size = 8
    linewidth = 1.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axes
    ax.set_xlabel('Training Block', fontsize=ax_font_size)
    ax.set_ylabel('Cumulative Frequency', fontsize=ax_font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ##########################################################################
    # plot
    for (x, y, last_x, last_y, probe) in xys:
        ax.plot(x, y, '-', linewidth=linewidth, c=next(palette))
        if last_y > y_thr:
            plt.annotate(probe, xy=(last_x, last_y),
                         xytext=(0, 0), textcoords='offset points',
                         va='center', fontsize=leg_font_size, bbox=dict(boxstyle='round', fc='w'))
    ##########################################################################
    ax.legend(fontsize=leg_font_size, loc='upper left')
    ##########################################################################
    return fig


def make_ba_pp_window_corr_fig(trajdatabase, window=20):
    ##########################################################################
    # load data
    ba_pp_mw_corr, ba_pp_ew_corr = trajdatabase.get_window_corr_data(window)
    ##########################################################################
    # choose seaborn style and palette
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig settings
    figsize = (6, 3)
    ax_font_size = 8
    leg_font_size = 8
    linewidth = 2.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
    ax.set_ylabel('Correlation', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,d}'.format(int(x))))
    ##########################################################################
    # plot line through y=0
    ax.axhline(y=0, linestyle='--', c='gray', linewidth=linewidth)
    ##########################################################################
    # plot
    ax.plot(trajdatabase.get_mbs_axis(), ba_pp_mw_corr, '-', linewidth=linewidth,
            label='mw-corr ({} blocks) between balAcc and test-pp'.format(window))
    ax.plot(trajdatabase.get_mbs_axis(), ba_pp_ew_corr, '-', linewidth=linewidth,
            label='ew-corr between balAcc and test-pp')
    ##########################################################################
    ax.legend(fontsize=leg_font_size, loc='best')
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_comp_probes_ba_fig(database, probe_tuples):
    ##########################################################################
    # load data
    xys = []
    for probe_class, group in groupby(probe_tuples, itemgetter(1)):
        probes = [i[0] for i in list(group)]
        avg_probe_ba_trajs_mat = database.get_trajs_mat(probes, 'avg_probe_ba')
        df = pd.DataFrame(avg_probe_ba_trajs_mat)
        y, std, n = df.mean(), df.std(), len(df)
        se = std / (n ** 0.5)
        ci = se * scipy.stats.t._ppf((1 + 0.95) / 2., n - 1)
        x = database.get_mbs_axis()
        xys.append((x, y, ci, probe_class, n))
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    num_probe_classes = sum(1 for _ in groupby(probe_tuples, itemgetter(1)))
    palette = iter(sns.color_palette("hls", num_probe_classes))
    ##########################################################################
    # fig
    figsize = (6, 3)
    ax_font_size = 8
    leg_font_size = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.set_ylim([40, 100])
    ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
    ax.set_ylabel('Balanced Accuracy', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ##########################################################################
    # plot
    for (x, y, ci, probe_class, n) in xys:
        ax.plot(x, y, c=next(palette), label='{} (mean +/- 95% CI) n={}'.format(probe_class.replace('_', ' '), n))
        ax.fill_between(x, y + ci, y - ci, alpha=0.15)
    ax.axhline(50, linestyle='--', color='grey')
    ##########################################################################
    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=leg_font_size, loc='best')
    ##########################################################################
    return fig


def make_ba_bds_fig(databases, palette, dotted=False, xaxis_labeled=False, grid=False):
    #########################################################################
    # load data
    ba_breakdown_avg_lines, model_names = [], []
    for n, database in enumerate(databases):
        model_names.append(database.model_name)
        if n == 0:
            cats_sorted_by_model_name, cat_ba_dict, _ = database.get_ba_breakdown_data()
        else:
            _, cat_ba_dict, _ = database.get_ba_breakdown_data()
        ba_breakdown_avg_line = [cat_ba_dict[cat] for cat in cats_sorted_by_model_name]
        ba_breakdown_avg_lines.append(ba_breakdown_avg_line)
    #########################################################################
    # fig
    figsize = (3.2, 3.2)
    ax_font_size = 8
    leg_font_size = 8
    tick_label_fontsize = 6
    markersize = 4
    linewidth = 2.0
    fig, ax = plt.subplots(figsize=figsize) # TODO get rid of all fig names like this here
    ##########################################################################
    # axes
    if xaxis_labeled: ax.set_xlabel('Categories', fontsize=ax_font_size)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
    ax.set_xticks(np.arange(len(database.cat_list)) + 0.5, minor=False)
    ax.set_xticklabels(cats_sorted_by_model_name, minor=False, fontsize=tick_label_fontsize, rotation=90)
    ax.set_xlim([0, len(database.cat_list) + 0.5])
    ax.set_axisbelow(True) # put grid under plot lines
    if grid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
    ax.tick_params(axis='both', which='both', top='off', right='off', labelsize=tick_label_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ##########################################################################
    # plot
    num_cats = len(database.cat_list)
    x = [i + 0.5 for i in range(num_cats)]
    for n, ba_breakdown_avg_line in enumerate(ba_breakdown_avg_lines):
        color = next(palette)
        ax.plot(x, ba_breakdown_avg_line, '-', color=color, linewidth=linewidth, label=model_names[n])
        if dotted: ax.plot(x, ba_breakdown_avg_line, '.', color=color, markersize=markersize)
    ##########################################################################
    # legend
    leg = plt.legend(fontsize=leg_font_size, loc='best', frameon=True)
    leg.get_frame().set_linewidth(1.0)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_probes_ba_traj_fig(databases, palette):
    ##########################################################################
    # load data
    xys = []
    for database in databases:
        print database.model_name
        avg_token_ba_traj = database.get_traj('probes_ba')
        x = database.get_mbs_axis()
        y = avg_token_ba_traj
        xys.append((x, y))
    ##########################################################################
    # choose seaborn style
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 3)
    ax_font_size = 8
    linewidth = 2.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.set_ylim([50, 75])
    ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
    ax.set_ylabel('Mean Balanced Accuracy', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.grid(True)
    ##########################################################################
    # plot
    for (x, y) in xys:
        color = next(palette)
        ax.plot(x, y, '-', linewidth=linewidth, color=color)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_test_pp_traj_fig(databases, palette):
    ##########################################################################
    # load data
    xys = []
    for database in databases:
        test_pp_traj = database.get_traj('test_pp')
        x = database.get_mbs_axis()
        y = test_pp_traj
        xys.append((x, y))
    ##########################################################################
    #  seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig settings
    figsize = (6, 3)
    ax_font_size = 8
    linewidth = 2.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axes
    ax.set_ylim([0, np.mean(xys[0][1])])
    ax.set_ylabel('Test Perplexity Score', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.grid(True)
    ##########################################################################
    # plot
    for (x, y) in xys:
        color = next(palette)
        ax.plot(x, y, '-', linewidth=linewidth, color=color)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_probe_pp_traj_fig(databases, palette):
    ##########################################################################
    # load data
    xys = []
    for database in databases:
        probes_pp_traj = database.get_traj('probes_pp')
        x = database.get_mbs_axis()
        y = probes_pp_traj
        xys.append((x, y))
    ##########################################################################
    #  seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig settings
    figsize = (6, 3)
    ax_font_size = 8
    linewidth = 2.0
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axes
    ax.set_ylim([0, np.mean(xys[0][1])])
    ax.set_ylabel('Probes Perplexity Score', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xlabel('Number of Batches', fontsize=ax_font_size)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.grid(True)
    ##########################################################################
    # plot
    for (x, y) in xys:
        color = next(palette)
        ax.plot(x, y, '-', linewidth=linewidth, color=color)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_probe_sim_comp_fig(databases, palette, num_bins = 1000, num_samples=1000):
    ##########################################################################
    assert len(databases) == 2 # scatter plot is 2d
    ##########################################################################
    # load sim data
    probe_simmat_values_list,model_names = [], []
    for database in databases:
        model_names.append(database.model_name)
        probes_acts_df = database.get_probes_acts_df()
        probe_simmat = calc_probe_sim_mat(probes_acts_df)
        probe_simmat[np.tril_indices(probe_simmat.shape[0], -1)] = np.nan
        probe_simmat_values = probe_simmat[~np.isnan(probe_simmat)]
        probe_simmat_values_list.append(probe_simmat_values)
    x = probe_simmat_values_list[0]
    y = probe_simmat_values_list[1]
    ##########################################################################
    #  seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 6)
    ax_font_size = 8
    leg_fontsize = 16
    markersize = 2.0
    linewidth = 1.0
    fig, axarr = plt.subplots(2, figsize=figsize)
    ##########################################################################
    # axis 0
    axarr[0].set_xlabel('Similarities {}'.format(model_names[0]), fontsize=ax_font_size)
    axarr[0].set_ylabel('Similarities {}'.format(model_names[1]), fontsize=ax_font_size)
    axarr[0].set_xlim([0, 1])
    axarr[0].set_ylim([0, 1])
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # plot scatter (sampled to increase speed)
    idx = np.random.choice(np.arange(len(x)), num_samples, replace=False)
    x_scatter, y_scatter = x[idx], y[idx]
    axarr[0].scatter(x_scatter, y_scatter, s=markersize, c='black', lw=0)
    ##########################################################################
    # plot best fit line
    xys = zip(x, y)
    plot_best_fit_line(axarr[0], xys, leg_fontsize)
    ##########################################################################
    # axis 1
    axarr[1].set_xlabel('Similarity', fontsize=ax_font_size)
    axarr[1].set_ylabel('Frequency', fontsize=ax_font_size)
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].tick_params(axis='both', which='both', top='off', right='off')
    axarr[1].xaxis.grid(True)
    ##########################################################################
    # plot sim hist
    for n, probe_simmat_values in enumerate(probe_simmat_values_list):
        step_size = 1.0 / num_bins
        bins = np.arange(0, 1, step_size)
        hist, _ = np.histogram(probe_simmat_values, bins=bins)
        x_binned = bins[:-1]
        axarr[1].plot(x_binned, hist, '-', linewidth=linewidth, c=next(palette))
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig



def make_neighbors_rbo_fig(databases, probes, num_neighbors=10):
    ##########################################################################
    # only use 2 database
    assert len(databases) == 2
    ##########################################################################
    # load data
    neighbors_dict = {probe: [] for probe in probes}
    for database in databases:
        probes_acts_df = database.get_probes_acts_df()
        probe_simmat = calc_probe_sim_mat(probes_acts_df)
        for probe_id, probe in enumerate(probes):
            sim_tuples_unsorted = [(target, sim) for target, sim in zip(database.probe_list, probe_simmat[probe_id])]
            neighbors_tuples = sorted(sim_tuples_unsorted, key=itemgetter(1), reverse=True)
            probe_neighbors = [tuple[0] for tuple in neighbors_tuples[1:num_neighbors]]
            neighbors_dict[probe].append(probe_neighbors)
    rbo_list = []
    for probe, neighbors_list in neighbors_dict.iteritems():
        neighbors1, neighbors2 = neighbors_dict[probe]
        rbo_list.append(score_2_neighbor_rankings(neighbors1, neighbors2))
    ##########################################################################
    # plot
    df = pd.DataFrame(data={'rbo': rbo_list, 'probes': probes})
    fig = Bar(df, values='rbo', label='probes', plot_width=600, plot_height=400, legend=False)
    fig.yaxis.axis_label = "Rank Biased Overlap"
    fig.y_range = Range1d(0, 1)
    ##########################################################################
    return fig


def make_cat_token_ba_comp_fig(databases, cat, max_num_probes=50, ylim=20):
    ##########################################################################
    # load data
    database1, database2, = databases
    probes = database1.cat_probe_list_dict[cat]
    probe_id_dict = database1.probe_id_dict
    probe_ids = [probe_id_dict[probe] for probe in probes]
    avg_probe_ba_list1 = database1.get_avg_probe_ba_list()
    avg_probe_ba_list2 = database2.get_avg_probe_ba_list()
    ba_diff_list_all = np.subtract(avg_probe_ba_list1, avg_probe_ba_list2)
    ba_diff_tuples = [(ba_diff_list_all[i], probe) for i, probe in zip(probe_ids, probes)]
    ba_diff_tuples_sorted = sorted(ba_diff_tuples, key=itemgetter(0), reverse=True)[:max_num_probes]
    ba_diff_list, probes = zip(*ba_diff_tuples_sorted)  # unpack tuples
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 3)
    ax_font_size = 8
    linewidth = 2
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off', bottom='off')
    ax.set_ylabel('Balanced Accuracy Difference', fontsize=ax_font_size)
    ax.set_xlabel('Probes in {}'.format(cat), fontsize=ax_font_size)
    xticks = np.add(range(len(probes)), 0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(probes, fontsize=ax_font_size, rotation=90)
    ax.set_ylim([-ylim, ylim])
    ##########################################################################
    # plot
    ax.plot(xticks, ba_diff_list, '.-', c='black', linewidth=linewidth)
    ax.axhline(0, linestyle='--', color='grey')
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_num_clusters_ba_diff_corr_fig(databases, annotate_x_thr=6):
    ##########################################################################
    # load data
    database1, database2, = databases
    avg_probe_ba_list1 = database1.get_avg_probe_ba_list()
    avg_probe_ba_list2 = database2.get_avg_probe_ba_list()
    ba_diff_list_all = np.subtract(avg_probe_ba_list1, avg_probe_ba_list2)
    num_clusters_list = []
    for probe in database1.probe_list:
        num_clusters = calc_num_probe_acts_clusters(database1, probe)
        num_clusters_list.append(num_clusters)
    x = num_clusters_list
    y = ba_diff_list_all
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 6)
    markersize = 10
    ax_font_size = 8
    leg_font_size = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_ylabel('Balanced Accuracy Difference', fontsize=ax_font_size)
    ax.set_xlabel('Number of Probe Acts Clusters', fontsize=ax_font_size)
    ##########################################################################
    # plot
    ax.scatter(x, y, s=markersize, facecolor='black', zorder=2)
    ax.axhline(y=0, linestyle='--', c='grey', zorder=1)
    ##########################################################################
    # annotate
    for x_, y_, probe in zip(x, y, database1.probe_list):
        if x_ > annotate_x_thr:
            txt = plt.annotate(probe, xy=(x_, y_), xytext=(2, 0), textcoords='offset points',
                               va='bottom', fontsize=leg_font_size)
            txt.set_path_effects([PathEffects.Stroke(linewidth=1.0, foreground="w"), PathEffects.Normal()])
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_probe_freq_ba_diff_corr_fig(databases, annotate_x_thr=5000):
    ##########################################################################
    # load data
    database1, database2, = databases
    avg_probe_ba_list1 = database1.get_avg_probe_ba_list()
    avg_probe_ba_list2 = database2.get_avg_probe_ba_list()
    ba_diff_list_all = np.subtract(avg_probe_ba_list1, avg_probe_ba_list2)
    probe_freq_list = []
    for probe in database1.probe_list:
        probe_freq = database1.probe_cf_traj_dict[probe][-1]
        probe_freq_list.append(probe_freq)
    x = probe_freq_list
    y = ba_diff_list_all
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 6)
    markersize = 10
    ax_font_size = 8
    leg_font_size = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_ylabel('Balanced Accuracy Difference', fontsize=ax_font_size)
    ax.set_xlabel('Frequency', fontsize=ax_font_size)
    ##########################################################################
    # plot
    ax.scatter(x, y, s=markersize, facecolor='black', zorder=2)
    ax.axhline(y=0, linestyle='--', c='grey', zorder=1)
    ##########################################################################
    # annotate
    for x_, y_, probe in zip(x, y, database1.probe_list):
        if x_ > annotate_x_thr:
            txt = plt.annotate(probe, xy=(x_, y_), xytext=(2, 0), textcoords='offset points',
                               va='bottom', fontsize=leg_font_size)
            txt.set_path_effects([PathEffects.Stroke(linewidth=1.0, foreground="w"), PathEffects.Normal()])
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_probe_ba_vs_pp_fig(database, probe):
    ##########################################################################
    # load data
    probe_pp_list = database.get_probe_pps(probe)
    probe_ba_list = database.get_probe_bas(probe)
    x = filter(lambda x: str(x) != 'nan', probe_pp_list)
    y = filter(lambda x: str(x) != 'nan', probe_ba_list)
    x, y = x[:len(y)], y[:len(x)]
    print 'x'
    print x
    print
    print 'y'
    print y
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (3, 3)
    markersize = 10
    ax_font_size = 8
    leg_fontsize = 10
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.set_xlim([0, database.num_input_units])
    ax.set_xticks([1, database.num_input_units])
    ax.set_xticklabels(['1', str(database.num_input_units)], minor=False, fontsize=ax_font_size)
    ax.set_ylim([0, 100])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_ylabel('Balanced Accuracy (%) of "{}"'.format(probe), fontsize=ax_font_size)
    ax.set_xlabel('Perplexity', fontsize=ax_font_size)
    ##########################################################################
    # plot
    if len(y) == 1:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.5, 'Data not available', transform=ax.transAxes,
                fontsize=leg_fontsize, verticalalignment='bottom', bbox=props)
    else:
        ax.scatter(x, y, s=markersize, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, leg_fontsize)
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_avg_probe_pp_ba_diff_corr_fig(databases, annotate_x_thr=50000):
    ##########################################################################
    # load data
    database1, database2, = databases
    avg_probe_ba_list1 = database1.get_avg_probe_ba_list()
    avg_probe_ba_list2 = database2.get_avg_probe_ba_list()
    ba_diff_list_all = np.subtract(avg_probe_ba_list1, avg_probe_ba_list2)
    avg_probe_pp_list = database1.get_avg_probe_pp_list
    x = avg_probe_pp_list
    y = ba_diff_list_all
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig
    figsize = (6, 6)
    markersize = 10
    ax_font_size = 8
    leg_font_size = 8
    fig, ax = plt.subplots(figsize=figsize)
    ##########################################################################
    # axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_ylabel('Balanced Accuracy Difference', fontsize=ax_font_size)
    ax.set_xlabel('Avg Token Perplexity', fontsize=ax_font_size)
    ##########################################################################
    # plot
    ax.scatter(x, y, s=markersize, facecolor='black', zorder=2)
    ax.axhline(y=0, linestyle='--', c='grey', zorder=1)
    ##########################################################################
    # annotate
    for x_, y_, probe in zip(x, y, database1.probe_list):
        if x_ > annotate_x_thr:
            txt = plt.annotate(probe, xy=(x_, y_), xytext=(2, 0), textcoords='offset points',
                               va='bottom', fontsize=leg_font_size)
            txt.set_path_effects([PathEffects.Stroke(linewidth=1.0, foreground="w"), PathEffects.Normal()])
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_probe_freq_hist_fig(database, probes):
    ##########################################################################
    # load data
    xys = []
    for probe in probes:
        x = database.get_mbs_axis()[2:]  # -1 because of np.diff, -1 because no cf data stored zero block
        y = np.diff(database.probe_cf_traj_dict[probe], 1)[:len(x)]
        xys.append((x, y, probe))
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    palette = iter(sns.color_palette("hls", len(probes)))
    ##########################################################################
    # fig
    figsize = (6, 4)
    ax_font_size = 8
    leg_font_size = 8
    linewidth = 2.0
    fig, ax = plt.subplots(1, figsize=figsize)
    ##########################################################################
    # axis
    ax.set_xlabel('Training Block', fontsize=ax_font_size)
    ax.set_ylabel('Frequency', fontsize=ax_font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # plot
    for (x, y, probe) in xys:
        ax.plot(x, y, '-', linewidth=linewidth, c=next(palette),
                label='{} (total freq : {})'.format(probe, int(database.probe_cf_traj_dict[probe][-1])))
    ##########################################################################
    # legend
    ax.legend(fontsize=leg_font_size, loc='best')
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_comp_binned_freqs_fig(trajdatabase, probe_tuples):
    ##########################################################################
    # load data
    xys = []
    ys_ = []
    for probe_class, group in groupby(probe_tuples, itemgetter(1)):
        probes = [i[0] for i in list(group)]
        x = trajdatabase.get_mbs_axis()[:-1]  # cut short because of diff
        for probe in probes:
            y_ = np.hstack((np.zeros(1), np.diff(trajdatabase.probe_cf_traj_dict[probe], 1)))[:len(x)]
            ys_.append(y_)
        y, std, n = np.mean(ys_, axis=0), np.std(ys_, axis=0), len(ys_)
        se = std / (n ** 0.5)
        ci = se * scipy.stats.t._ppf((1 + 0.95) / 2., n - 1)
        xys.append((x, y, ci, probe_class))
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    num_probe_classes = sum(1 for _ in groupby(probe_tuples, itemgetter(1)))
    palette = iter(sns.color_palette("hls", num_probe_classes))
    ##########################################################################
    # fig
    figsize = (6, 4)
    ax_font_size = 8
    leg_font_size = 8
    leg_fontsize = 10
    linewidth = 2.0
    fig, ax = plt.subplots(1, figsize=figsize)
    ##########################################################################
    # axis
    ax.set_xlabel('Training Block', fontsize=ax_font_size)
    ax.set_ylabel('Frequency', fontsize=ax_font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ##########################################################################
    # plot
    if len(y) == 1:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.5, 'Data not available', transform=ax.transAxes,
                fontsize=leg_fontsize, verticalalignment='bottom', bbox=props)
    else:
        for (x, y, ci, probe_class) in xys:
            ax.plot(x, y, '-', linewidth=linewidth, c=next(palette),
                    label='{} avg probe freq +/- 95% CI'.format(probe_class))
            ax.fill_between(x, y + ci, y - ci, alpha=0.15)
        ax.axhline(0, linestyle='--', color='grey')
    ##########################################################################
    # legend
    ax.legend(fontsize=leg_font_size, loc='best')
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig



def make_cat_count_pie_chart_fig(database):
    ##########################################################################
    # load data
    df_probe_count_by_cat = database.df[['cat', 'probe']].groupby('cat').count()
    probe_count_by_cat = df_probe_count_by_cat.T.values[0]
    sorted_cats = df_probe_count_by_cat.index
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    palette = np.random.permutation(sns.color_palette("hls", len(sorted_cats)))
    ##########################################################################
    # fig
    figsize = (6, 6)
    linewidth = 1.0
    leg_font_size = 8
    fig, ax = plt.subplots(1, figsize=figsize)
    ##########################################################################
    # axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.axis('equal')
    ##########################################################################
    # plot pie chart
    wedges, texts = ax.pie(probe_count_by_cat, labels=sorted_cats, colors=palette)
    for w,t in zip(wedges, texts):
        w.set_linewidth(linewidth)
        t.set_fontsize(leg_font_size)
    ##########################################################################
    # correlate probe_count_by_cat with cat_ba
    cats_sorted_by_ba, cat_ba_dict, _ = database.get_ba_breakdown_data()
    df = pd.DataFrame(data={'cat': sorted_cats,
                            'probe_count_by_cat' : probe_count_by_cat,
                            'cat_ba': [cat_ba_dict[cat] for cat in sorted_cats]})

    corr = df.corr()['cat_ba'].loc['probe_count_by_cat']
    plt.annotate('Correlation cat_ba & probe count: {:.2f}'.format(corr),
                 xy=(0.05, 0.0), xycoords='axes fraction')
    ##########################################################################
    return fig


def make_corpus_traj_fig(database):
    ##########################################################################
    # load data
    tf_idf_mat = database.tf_idf_mat
    lex_div_traj = database.lex_div_traj
    ##########################################################################
    # make x and tf_idf_corr_traj
    num_train_docs = tf_idf_mat.shape[0]
    x = range(num_train_docs)
    tf_idf_corr_traj = []
    for row_id in range(num_train_docs):
        if row_id == 0:
            corr = 0
        else:
            df = pd.DataFrame({ 't': tf_idf_mat[row_id], 't-1': tf_idf_mat[row_id-1]})
            corr = df.corr()['t'].loc['t-1']
        tf_idf_corr_traj.append(corr)
    ##########################################################################
    # fig
    linewidth = 1.0
    hover = HoverTool(tooltips=[("train_doc #", "@x")])
    tools = [hover, 'pan, wheel_zoom, crosshair, save']
    fig = figure(plot_width = 600, plot_height=300, tools=tools)
    ##########################################################################
    # plot
    fig.line(x, tf_idf_corr_traj, line_width=linewidth, legend='tf_idf_corr_traj')
    fig.line(x, lex_div_traj, line_width=linewidth, legend='lex_div_traj', color='red')
    fig.circle(x, lex_div_traj, line_width=linewidth, color='red', size=1)
    ##########################################################################
    return fig