import os
import csv
import time
import shutil
import sys
import datetime, pytz
import StringIO
import subprocess
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.plotting import figure
import matplotlib.ticker as tkr
from operator import itemgetter


from dbutils import load_rnnlabrc
from dbutils import calc_probe_sim_mat
from database import DataBase
from trajdatabase import TrajDataBase


runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))


def gen_user_configs():
    ##########################################################################
    # define directories
    user_configs_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_user_configs.csv'))
    if not os.path.isfile(user_configs_path): sys.exit('rnnlab: {} not found'.format(user_configs_path))
    ##########################################################################
    # check that there are no duplicated configs
    reader = csv.reader(open(user_configs_path, 'r'))
    rows = []
    for n, row in enumerate(reader):
        if n != 0: rows.append(tuple(row))
    if len(set(rows)) != len(rows): print 'rnnlab WARNING: Duplicate configs detected in {}'.format(user_configs_path)
    ##########################################################################
    # gen user_configs (tuple)
    reader = csv.reader(open(user_configs_path, 'r'))
    for n, row in enumerate(reader):
        if n == 0:
            configs_names = row
        else:
            user_configs = [(name, config) for name, config in zip(configs_names, row)]
            ##########################################################################
            yield user_configs


def to_block_name(block_id):
    ##########################################################################
    return ('000000' + str(block_id))[-6:]


def block_name_exists(model_name, block_name):
    ##########################################################################
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    if os.path.isfile(os.path.join(path, 'df_block_{}.h5'.format(block_name))):
        return True
    else:
        return False


def get_log_mtime(timezone='America/Los_Angeles'): # TODO timestamp is not utc because it is local
    ##########################################################################
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    utc_last_updated = datetime.datetime.fromtimestamp(
        os.path.getmtime(log_path)).replace(tzinfo=pytz.utc)
    log_mtime_unformatted = utc_last_updated.astimezone(pytz.timezone(timezone))
    format = "%Y-%m-%d %H:%M"
    log_mtime = log_mtime_unformatted.strftime(format)
    ##########################################################################
    return log_mtime


def load_custom_probes_tuples():
    ##########################################################################
    custom_probes = []
    with open(os.path.join('static', 'custom_probes.txt'), 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                probe = line.split()[0]
                probe_class = line.split()[1]
                custom_probes.append((probe, probe_class))
    ##########################################################################
    return custom_probes


def make_imgs(*fig_tuples):
    ##########################################################################
    imgs = []
    for fig, origin in fig_tuples:
        if origin == 'bokeh':
            print 'Preparing bokeh img...'
            fig.xgrid.grid_line_color = None
            fig.toolbar.logo = None
            img = {} # components needs to be called once only
            img['script'], img['div'] = components(fig)
        else:
            print 'Preparing mpl img...'
            figfile = StringIO.StringIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)
            img = base64.b64encode(figfile.getvalue())
        imgs.append(img)
    ##########################################################################
    return imgs


def check_disk_space():
    ##########################################################################
    df = subprocess.Popen(["df", "{}".format(runs_dir)], stdout=subprocess.PIPE)
    df_str = df.communicate()[0]
    used = int(df_str.split('\n')[1].split()[4].strip('%'))
    ##########################################################################
    print 'Checking disk space of filesystem containing runs_dir:'
    print df_str
    ##########################################################################
    if used > 90: sys.exit('rnnlab: Disk space usage > 90%')


def write_to_bashrc(bashrc, alias):
    ##########################################################################
    with open(bashrc, 'r') as f:
        lines = f.readlines()
        if alias not in lines:
            out = open(bashrc, 'a')
            out.write(alias)
            out.close()
            string_to_print = 'rnnlab: Created bash alias. Restart bash and type "rnnlab" to start browser app'
        else:
            string_to_print = 'rnnlab: Type "rnnlab" to start browser app'
    ##########################################################################
    print '========================================================================='
    print string_to_print
    print '========================================================================='


def make_rnnlab_alias(app_dirname):
    ##########################################################################
    app_file_name = 'browser_app.py'
    app_path = os.path.join(app_dirname, app_file_name)
    alias = 'alias rnnlab="python {}"\n'.format(app_path)
    homefolder = os.path.expanduser('~')
    ##########################################################################
    try: # works on ubuntu
        bash_path_ubuntu = os.path.abspath('{}/.bashrc'.format(homefolder))
        write_to_bashrc(bash_path_ubuntu, alias)
    except IOError: # works on mac
        bash_path_mac = os.path.abspath('{}/.bash_profile'.format(homefolder))
        write_to_bashrc(bash_path_mac, alias)
    ##########################################################################
    except:
        print sys.exc_info()[0]
        print 'Could not create bash alias for running browser app. To start it, please type "python {}"'\
            .format(os.path.join(app_dirname, app_file_name))


def load_database(model_name, block_name):
    ##########################################################################
    # load configs
    configs_dict = load_configs_dict(model_name)
    ##########################################################################
    # make database
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    file_name = 'df_block_{}.h5'.format(block_name)
    df = pd.read_hdf(os.path.join(path, file_name), 'df')
    database = DataBase(configs_dict, df, block_name)
    ##########################################################################
    return database


def load_trajdatabase(model_name):
    ##########################################################################
    # load configs
    configs_dict = load_configs_dict(model_name)
    ##########################################################################
    # make trajdatabase
    trajdatabase = TrajDataBase(configs_dict)
    ##########################################################################
    return trajdatabase


def load_configs_dict(model_name):
    ##########################################################################
    # load configs
    path = os.path.join(runs_dir, model_name, 'Configs')
    file_name = 'configs_dict.npy'
    configs_dict = dict(np.load(os.path.join(path, file_name)).item())
    ##########################################################################
    return configs_dict


def remove_log_entry(model_name):
    ##########################################################################
    log_entries_list, headers = get_log_entries_list()
    ##########################################################################
    # get all entries except for that belonging to model_name
    new_log_entries_list = [headers]
    for log_entry in log_entries_list:
        model_name_ = log_entry[0]
        if model_name_ != model_name:
            new_log_entries_list.append(log_entry)
        elif 'model_name' in log_entry:
            new_log_entries_list.append(log_entry)
    ##########################################################################
    time.sleep(1)
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    with open(log_path, 'w') as f:
        writer = csv.writer(f)
        for log_entry in new_log_entries_list:
            writer.writerow(log_entry)


def is_training_completed(model_name):
    ##########################################################################
    log_entries_list, headers = get_log_entries_list()
    ##########################################################################
    for log_entry in log_entries_list:
        if log_entry[0] == model_name:
            is_completed = bool(int(log_entry[-2]))
            ##########################################################################
            return is_completed


def delete_model(model_name):
    ##########################################################################
    path_to_delete = os.path.join(runs_dir, model_name)
    ##########################################################################
    # prompt user if model has completed trainig
    if is_training_completed(model_name):
        if not 'yes' in raw_input('Model training has completed. Enter yes to continue deletion\n'):
            print 'Aborted deletion'
            return
    ##########################################################################
    # delete model
    shutil.rmtree(path_to_delete)
    remove_log_entry(model_name)
    ##########################################################################
    print 'Deleted {}'.format(model_name)


def get_log_entries_list():
    ##########################################################################
    # get log
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
    ##########################################################################
    headers = lines[0]
    log_entries_list = lines[1:]
    ##########################################################################
    return log_entries_list, headers


def load_filtered_log_entries(headers_to_display, allow_incomplete=True):
    ##########################################################################
    # get log
    log_entries_list, headers = get_log_entries_list()
    ##########################################################################
    col_ids = [headers.index(header_to_display) for header_to_display in headers_to_display
               if header_to_display in headers]
    filtered_headers = [i for n, i in enumerate(headers) if n in col_ids]
    c = [0,1] if allow_incomplete else [1]
    filtered_log_entries = [[i for n, i in enumerate(log_entry) if n in col_ids]
                            for log_entry in log_entries_list
                            if int(log_entry[-2]) in c
                            and float(log_entry[-1]) >= 50.0] # prevents concurrent reading and writing
    ##########################################################################
    # reverse so that newest entries are on top
    filtered_log_entries = filtered_log_entries[::-1]
    ##########################################################################
    return filtered_log_entries, filtered_headers


def get_saved_block_names(model_name):
    ##########################################################################
    trajdatabase = load_trajdatabase(model_name)
    saved_block_names = trajdatabase.trajstore.select_column('trajdf', 'index').values
    trajdatabase.trajstore.close()
    ##########################################################################
    return saved_block_names


def get_block_names_to_display(model_name, step=7):
    ##########################################################################
    block_names_to_display = get_saved_block_names(model_name)[0::step]
    ##########################################################################
    return block_names_to_display


def block_to_iteration(model_name, block_name):
    ##########################################################################
    # get configs_dict
    configs_dict = load_configs_dict(model_name)
    num_iterations = int(configs_dict['num_iterations'])
    iteration = int(block_name) * num_iterations
    ##########################################################################
    return iteration


def make_block_names2_dict(model_name1, block_names1, limit_to_same_flavor=False):
    ##########################################################################
    num_reps1 = load_configs_dict(model_name1)['num_reps']
    if limit_to_same_flavor: flavors = [model_name1.split('_')[1]]
    else: flavors = ['srn', 'lstm', 'irnn', 'scrn']
    filtered_log_entries, filtered_headers = load_filtered_log_entries(['model_name','num_reps'])
    model_names2 = [log_entry[0] for log_entry in filtered_log_entries
                    if log_entry[0].split('_')[1] in flavors
                    and log_entry[0] != model_name1
                    and int(log_entry[1]) == num_reps1] # compare only models with same num_reps
    ##########################################################################
    # make block_names2_dict
    block_names2_dict = {}
    for model_name2 in model_names2:
        ##########################################################################
        # get block_names2 (comparable blocks t block_names1 with respect to iterations)
        block_names2 = []
        saved_block_names2 = get_saved_block_names(model_name2)
        for saved_block_name2 in saved_block_names2:
            iteration2 = block_to_iteration(model_name2, saved_block_name2)
            for block_name1 in block_names1:
                iteration1 = block_to_iteration(model_name1, block_name1)
                if iteration2 == iteration1:
                    block_names2.append(saved_block_name2)
                    continue
        ##########################################################################
        # add block_names2
        block_names2_dict[model_name2] = block_names2
    ##########################################################################
    return model_names2, block_names2_dict


def make_ba_bds_fig(model_names_to_compare, block_names_to_compare, palette,
                    dotted=False, xaxis_labeled=False, grid=False, is_titled=False):
    #########################################################################
    # split data
    assert len(model_names_to_compare) == 2
    model_name1, model_name2 = model_names_to_compare
    block_name1, block_name2 = block_names_to_compare
    #########################################################################
    # get data model_name 1
    database = load_database(model_name1, block_name1)
    cats_sorted_by_model_name1, cat_ba_dict1, _ = database.get_ba_breakdown_data()
    ba_breakdown_avg_line1 = [cat_ba_dict1[cat] for cat in cats_sorted_by_model_name1]
    #########################################################################
    # get data model_name 2
    database = load_database(model_name2, block_name2)
    cats_sorted_by_model_name2, cat_ba_dict2, _ = database.get_ba_breakdown_data()
    ba_breakdown_avg_line2 = [cat_ba_dict2[cat] for cat in cats_sorted_by_model_name1]
    #########################################################################
    # merge data
    ba_breakdown_avg_lines = [ba_breakdown_avg_line1, ba_breakdown_avg_line2]
    #########################################################################
    # fig settings
    figsize = (3.2, 3.2) # 8,6
    title_font_size = 16
    ax_font_size = 8
    leg_font_size = 8
    tick_label_fontsize = 6
    markersize = 4
    linewidth = 2.0
    ##########################################################################
    # fig
    fig, ax = plt.subplots(figsize=figsize)
    fig_name = 'Block {}  Balanced Accuracy by Category Model Comparison'.format(block_name1)
    if is_titled: plt.title(fig_name, fontsize=title_font_size)
    ##########################################################################
    # axes
    if xaxis_labeled: ax.set_xlabel('Categories', fontsize=ax_font_size)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
    ax.set_xticks(np.arange(len(database.cat_list)) + 0.5, minor=False)
    ax.set_xticklabels(cats_sorted_by_model_name1, minor=False, fontsize=tick_label_fontsize, rotation=90)
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
        ax.plot(x, ba_breakdown_avg_line, '-', color=color, linewidth=linewidth, label=model_names_to_compare[n])
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


def make_avg_token_ba_trajs_fig(model_names, palette, is_title=False):
    ##########################################################################
    # load x,y
    avg_token_ba_trajs = []
    for model_name in model_names:
        trajdatabase = load_trajdatabase(model_name)
        avg_token_ba_traj = trajdatabase.trajstore.select_column('trajdf', 'avg_token_ba').values
        saved_block_names = get_saved_block_names(model_name)
        saved_iterations = map(lambda x: block_to_iteration(model_name, x), [int(i) for i in saved_block_names])
        x = saved_iterations
        y = avg_token_ba_traj[:len(x)] # trajdatabase data can be longer or shorter than data from database
        avg_token_ba_trajs.append((x,y))
        trajdatabase.trajstore.close()
    ##########################################################################
    # choose seaborn style
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig settings
    figsize = (6, 6)
    title_font_size = 16
    ax_font_size = 16
    linewidth = 2.0
    ##########################################################################
    # fig
    fig, ax = plt.subplots(figsize=figsize)
    fig_name = 'Balanced Accuracy Trajectories'
    if is_title: plt.title(fig_name, fontsize=title_font_size)
    ##########################################################################
    # axis
    ax.set_ylim([50, 75])
    ax.set_xlabel('Training Iteration', fontsize=ax_font_size)
    ax.set_ylabel('Average Balanced Accuracy', fontsize=ax_font_size)
    ax.get_xaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: '{:,}'.format(x))) # TODO formatting doesn't work
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # plot
    for (x,y) in avg_token_ba_trajs:
        color = next(palette)
        ax.plot(x, y, '-', linewidth=linewidth, color=color)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_test_pp_trajs_fig(model_names, palette, is_title=False):
    ##########################################################################
    # load x,y
    test_pp_trajs = []
    for model_name in model_names:
        trajdatabase = load_trajdatabase(model_name)
        test_pp_traj = trajdatabase.trajstore.select_column('trajdf', 'test_pp').values
        saved_block_names = get_saved_block_names(model_name)
        saved_iterations = map(lambda x: block_to_iteration(model_name, x), [int(i) for i in saved_block_names])
        x = saved_iterations
        y = test_pp_traj[:len(x)] # trajdatabase data can be longer or shorter than data from database
        test_pp_trajs.append((x, y))
        trajdatabase.trajstore.close()
    ##########################################################################
    #  seaborn
    import seaborn as sns
    sns.set_style('white')
    ##########################################################################
    # fig settings
    figsize = (1, 1) # this doesn't affect bokeh
    title_font_size = 16
    ax_font_size = 16
    linewidth = 2.0
    ##########################################################################
    # fig
    fig, ax = plt.subplots() # figsize=figsize)
    fig_name = 'Test Perplexity Trajectories'
    if is_title: plt.title(fig_name, fontsize=title_font_size)
    ##########################################################################
    # axes
    ax.set_ylabel('Test Perplexity Score', fontsize=ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xlabel('Training Iteration', fontsize=ax_font_size)
    ##########################################################################
    # plot
    for (x,y) in test_pp_trajs:
        color = next(palette)
        ax.plot(x, y, '-', linewidth=linewidth, color=color)
    ##########################################################################
    # layout
    plt.tight_layout()
    ##########################################################################
    return fig


def make_probe_sim_comp_fig(model_names, block_names, palette, num_bins = 1000, num_samples=1000, is_title=False):
    ##########################################################################
    assert len(model_names) == 2 # scatter plot is 2d
    ##########################################################################
    # load sim data
    probe_simmat_values_list = []
    for model_name, block_name in zip(model_names, block_names):
        database = load_database(model_name, block_name)
        probe_simmat = calc_probe_sim_mat(database.make_all_acts_df(), database.probe_list)
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
    # fig settings
    figsize = (6, 6)
    title_font_size = 16
    ax_font_size = 16
    leg_fontsize = 16
    markersize = 2.0
    linewidth = 1.0
    ##########################################################################
    # fig
    fig, axarr = plt.subplots(2, figsize=figsize)
    fig_name = 'Probe Similarities Comparison'
    if is_title: plt.title(fig_name, fontsize=title_font_size)
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
    best_fit_fxn = np.polyfit(x, y, 1, full=True)
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]
    axarr[0].plot(xl, yl, c='red')
    ##########################################################################
    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=4)
    axarr[0].text(0.01, 0.9, '$R^2$ = {}'.format(Rsqr), fontsize=leg_fontsize)
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


def score_2_neighbor_rankings(neighbors1, neighbors2, p=0.98):
    ##########################################################################
    if neighbors1 == None: neighbors1 = []
    if neighbors2 == None: neighbors2 = []
    ##########################################################################
    # rot by length and check that shortest length is not 0 (s - short, l -long)
    sl, ll = sorted([(len(neighbors1), neighbors1), (len(neighbors2), neighbors2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0
    ##########################################################################
    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0 # higher sum results in higher rbo
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1
        ##########################################################################
        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        ##########################################################################
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        ##########################################################################
        sum1 += x_d[d] / d * pow(p, d) # power -> bounds result between 0 and 1
    ##########################################################################
    # in case lists are of different lengths
    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)
    ##########################################################################
    # small adjustment (for what?)
    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)
    ##########################################################################
    # Equation
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    ##########################################################################
    return rbo_ext


def make_neighbors_rbo_fig(model_names, block_names, probes, num_neighbors=10):# the smaller the more sensitive
    ##########################################################################
    # only use 2 model names
    assert len(model_names) == 2
    ##########################################################################
    neighbors_dict = {probe: [] for probe in probes}
    for model_name, block_name in zip(model_names, block_names):
        database = load_database(model_name, block_name)
        probe_simmat = calc_probe_sim_mat(database.make_all_acts_df(), database.probe_list)
        ##########################################################################
        # get probe_neighbors
        for probe_id, probe in enumerate(probes):
            sim_tuples_unsorted = [(target, sim) for target, sim in zip(database.probe_list, probe_simmat[probe_id])]
            neighbors_tuples = sorted(sim_tuples_unsorted, key=itemgetter(1), reverse=True)
            probe_neighbors = [tuple[0] for tuple in neighbors_tuples[1:num_neighbors]]
            neighbors_dict[probe].append(probe_neighbors)
    ##########################################################################
    rbo_list = []
    for probe, neighbors_list in neighbors_dict.iteritems():
        neighbors1, neighbors2 = neighbors_dict[probe]
        rbo_list.append(score_2_neighbor_rankings(neighbors1, neighbors2))
    ##########################################################################
    from bokeh.charts import Bar
    df = pd.DataFrame(data={'rbo': rbo_list, 'probes': probes})
    fig = Bar(df, values='rbo', label='probes', plot_width=600, plot_height=400, legend=False)
    fig.yaxis.axis_label = "Rank Biased Overlap"
    ##########################################################################
    return fig


def make_probe_freq_hist_fig(model_name, sel_probes, num_bins=10, is_titled=False):
    ##########################################################################
    database = load_database(model_name, to_block_name(0))
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    palette = iter(sns.color_palette("hls", len(sel_probes)))
    ##########################################################################
    # fig settings
    figsize = (6, 4)
    title_font_size = 16
    ax_font_size = 16
    leg_font_size = 12
    linewidth = 2.0
    ##########################################################################
    # fig
    fig, ax = plt.subplots(1, figsize=figsize)
    fig_name = '{} Probe Frequency Histogram'.format(model_name)
    if is_titled: plt.title(fig_name, fontsize=title_font_size)
    ##########################################################################
    # axis
    ax.set_xlabel('Training Block', fontsize=ax_font_size)
    ax.set_ylabel('Frequency', fontsize=ax_font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ##########################################################################
    # plot
    for probe in sel_probes:
        probe_freqs_per_doc = np.diff(database.probe_cf_traj_dict[probe], 1)
        doc_ids = np.arange(probe_freqs_per_doc.shape[0])
        bins = range(0, len(probe_freqs_per_doc), num_bins)
        hist, _ = np.histogram(doc_ids, bins=bins, weights=probe_freqs_per_doc)
        x = range(0, len(hist) * num_bins, num_bins)
        ax.plot(x, hist, '-', linewidth=linewidth, c=next(palette),
                label= '{} (total freq : {})'.format(probe, int(database.probe_cf_traj_dict[probe][-1])))
    ##########################################################################
    # legend
    ax.legend(fontsize=leg_font_size, loc='best')
    ##########################################################################
    # layout
    fig.tight_layout()
    ##########################################################################
    return fig


def make_cat_count_pie_chart_fig(model_name, is_titled=False):
    ##########################################################################
    # data
    last_block_name = get_saved_block_names(model_name)[-1]
    database = load_database(model_name, last_block_name)
    df_probe_count_by_cat = database.df[['cat', 'probe']].groupby('cat').count()
    probe_count_by_cat = df_probe_count_by_cat.T.values[0]
    sorted_cats = df_probe_count_by_cat.index
    ##########################################################################
    # seaborn
    import seaborn as sns
    sns.set_style('white')
    palette = np.random.permutation(sns.color_palette("hls", len(sorted_cats)))
    ##########################################################################
    # fig settings
    figsize = (6, 6)
    title_font_size = 16
    linewidth = 1.0
    leg_font_size = 8
    ##########################################################################
    # fig
    fig, ax = plt.subplots(1, figsize=figsize)
    fig_name = '{} Probe Frequency Histogram'.format(model_name)
    if is_titled: plt.title(fig_name, fontsize=title_font_size)
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


def make_corpus_traj_fig(model_name, is_titled=False):
    ##########################################################################
    # load data
    path = os.path.join(runs_dir, model_name, 'Token_Data') # TODO should be loaded from Corpus_Data
    file_name = 'corpus_data.npz'.format(model_name)
    npzfile = np.load(os.path.join(path, file_name))
    tf_idf_mat = np.asarray(npzfile['tf_idf_mat'])
    lex_div_traj = npzfile['lex_div_traj']
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
    # fig settings
    linewidth = 1.0
    leg_font_size = 8
    ##########################################################################
    # fig
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