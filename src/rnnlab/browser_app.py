import os, datetime, pytz, csv, socket, base64, shutil
from flask import Flask, session
from flask import render_template
from flask import request
from flask import send_file
from itertools import cycle
import pandas as pd
import numpy as np
import StringIO
from database import DataBase
from trajdatabase import TrajDataBase
from utilities import load_rc
from utilities import remove_log_entry
from utilities import remove_model_data
from bokeh import mpl
from bokeh.models import Range1d
from bokeh.models import HoverTool
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Line
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category10
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd

##########################################################################
app = Flask(__name__)
##########################################################################
DEFAULTS = {'sel_block_name': 'Select',
            'sel_cat': 'Select',
            'sel_probe': 'Select',
            'sel_maction': 'Select',
            'mactions': ['Select','Compare', 'Delete'],
            'headers': ['model_name', 'optimizer','learning_rate',
                        'bptt_steps', 'num_hidden_units', 'num_iterations', 'best_token_ba'],
            'allow_incomplete' : True}
##########################################################################
log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
##########################################################################


def get_trained_block_names(model_name, default_blocks=('0001','1000','2000','2800')):
    ##########################################################################
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    ##########################################################################
    trained_block_names = [DEFAULTS['sel_block_name'], 'Trajectory']
    for b in default_blocks:
        if os.path.isfile(os.path.join(path, 'df_block_{}.h5'.format(b))):
            trained_block_names.append(b)
    ##########################################################################
    # append last available block_name (get second to last one to prevent concurrentreading and writing )
    saved_df_filenames = sorted([b for b in os.listdir(path) if b.startswith('df')])
    if len(saved_df_filenames) > 2:
        last_block_name = filter(lambda x: x.isdigit(), os.path.splitext(saved_df_filenames[-2])[0])
        if not default_blocks[-1] in trained_block_names:
            trained_block_names.append(last_block_name)
    ##########################################################################
    return trained_block_names


def get_requests():
    ##########################################################################
    sel_model_name = request.args.get('model_name')
    sel_block_name = request.args.get('block_name')
    if not sel_block_name: sel_block_name = DEFAULTS['sel_block_name']
    sel_probe = request.args.get('probe')
    if not sel_probe: sel_probe = DEFAULTS['sel_probe']
    sel_cat = request.args.get('cat')
    if not sel_cat: sel_cat = DEFAULTS['sel_cat']
    sel_cat2 = request.args.get('cat2')
    if not sel_cat2: sel_cat = DEFAULTS['sel_cat']
    sel_maction = request.args.get('maction')
    if not sel_maction: sel_maction = DEFAULTS['sel_maction']
    ##########################################################################
    return sel_model_name, sel_block_name, sel_probe, sel_cat, sel_cat2, sel_maction


def load_databases(model_name, block_name):
    ##########################################################################
    # load configs
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Configs')
    file_name = 'configs_dict.npy'
    configs_dict = dict(np.load(os.path.join(path, file_name)).item())
    ##########################################################################
    # make database
    if block_name != 'Trajectory':
        print 'Loading database for {} {}...'.format(model_name, block_name)
        path = os.path.join(runs_dir, model_name, 'Data_Frame')
        file_name = 'df_block_{}.h5'.format(block_name)
        df = pd.read_hdf(os.path.join(path, file_name), 'df')
        database = DataBase(configs_dict, df, block_name)
    else:
        database = None
    ##########################################################################
    # make trajdatabase
    print 'Loading trajdatabase for {}...'.format(model_name)
    trajdatabase = TrajDataBase(configs_dict)
    ##########################################################################
    return database, trajdatabase


def get_log_mtime():
    ##########################################################################
    utc_last_updated = datetime.datetime.fromtimestamp(
        os.path.getmtime(log_path)).replace(tzinfo=pytz.utc)
    log_mtime_unformatted = utc_last_updated.astimezone(pytz.timezone('America/Los_Angeles'))
    format = "%Y-%m-%d %H:%M"
    log_mtime = log_mtime_unformatted.strftime(format)
    ##########################################################################
    return log_mtime


def load_filtered_log_entries():
    ##########################################################################
    # get log
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        log_content = []
        for line in reader: log_content.append(line)
    headers = log_content[0]
    ##########################################################################
    col_ids = [headers.index(header) for header in list(DEFAULTS['headers'])]
    filtered_headers = [i for n, i in enumerate(headers) if n in col_ids]
    check_completed = 0 if DEFAULTS['allow_incomplete'] else 1
    filtered_log_entries = [[i for n, i in enumerate(log_entry) if n in col_ids]
                            for log_entry in log_content[1:]
                            if int(log_entry[-2]) == check_completed
                            and float(log_entry[-1]) >= 50.0] # prevents concurrent reading and writing
    ##########################################################################
    # reverse so that newest entries are on top
    filtered_log_entries = filtered_log_entries[::-1]
    ##########################################################################
    return filtered_log_entries, filtered_headers


def get_model_names_to_compare(flavor, block_name):
    ##########################################################################
    # get model_names_same_flavor
    filtered_log_entries, filtered_headers = load_filtered_log_entries()
    model_names_same_flavor = [entry[0] for entry in filtered_log_entries if entry[0].endswith(flavor)]
    ##########################################################################
    model_names_to_compare = []
    for model_name in model_names_same_flavor:
        ##########################################################################
        # get trained block_names
        runs_dir = os.path.abspath(load_rc('runs_dir'))
        path = os.path.join(runs_dir, model_name, 'Data_Frame')
        saved_file_names = sorted([b for b in os.listdir(path) if b.startswith('df')])
        trained_block_names = [filter(lambda x: x.isdigit(), os.path.splitext(saved_file_name)[0])
                               for saved_file_name in saved_file_names]
        ##########################################################################
        # make model_names_to_compare
        if block_name in trained_block_names:
            model_names_to_compare.append(model_name)
    ##########################################################################
    return model_names_to_compare


def get_custom_probes():
    ##########################################################################
    custom_probes = []
    num_probe_classes = 0
    prev_probe_class = None
    with open(os.path.join('static', 'custom_probes.txt'), 'r') as f:
        for line in f.readlines():
            probe = line.split()[0]
            probe_class = line.split()[1]
            custom_probes.append((probe, probe_class))
            if probe_class != prev_probe_class: num_probe_classes += 1
            prev_probe_class = probe_class
    ##########################################################################
    return custom_probes, num_probe_classes


@app.route('/', methods=['GET', 'POST'])
def home():
    ##########################################################################
    # inits
    bokeh_head = """
        <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.4.min.css" type="text/css" />
        <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.4.min.js"></script>
        <script type="text/javascript">
            Bokeh.set_log_level("info");
        </script>
        """
    probes = [DEFAULTS['sel_probe']]
    cats = [DEFAULTS['sel_cat'], 'Custom List']
    cats2 = [DEFAULTS['sel_cat']]
    block_names = [DEFAULTS['sel_block_name']]
    mactions = DEFAULTS['mactions']
    button_class = 'button-on'
    acts_2d_img = None
    token_acts_dh_img = None
    token_corcoeff_hist_img = None
    acts_dh_img = None
    ba_breakdown_scatter_img = {}
    ba_breakdow_img = None
    cat_cluster_img = None
    cat_sim_dh_img = None
    neighbors_table_img = None
    token_ba_trajs_img = {}
    test_pp_traj_img = {}
    avg_token_ba_traj_img = {}
    cfreq_traj_img = None
    ba_pp_mw_corr_img = None
    cats_sorted_by_ba_master = None
    ba_comparison_reference_img = None
    ##########################################################################
    print '\nLoaded home'
    ##########################################################################
    # load any requests
    sel_model_name, sel_block_name, sel_probe, sel_cat, sel_cat2, sel_maction = get_requests()
    ##########################################################################
    # if specified, delete model
    if sel_maction == 'Delete':
        remove_log_entry(sel_model_name)
        remove_model_data(sel_model_name)
    ##########################################################################
    # get log entries
    log_entries, headers = load_filtered_log_entries()
    ##########################################################################
    if log_entries:
        ##########################################################################
        # if not model_name selected, select first as default, store in session
        if not sel_model_name or sel_maction == 'Delete':
            sel_model_name = log_entries[0][0]
            session['model_name'] = None
            session['block_name'] = None
        ##########################################################################
        # if different_model_name selected, set block_name to default
        if session['model_name'] != sel_model_name and session['block_name'] != 'Trajectory':
            sel_block_name = DEFAULTS['sel_block_name']
        session['model_name'] = sel_model_name
        session['block_name'] = sel_block_name
        ##########################################################################
        # only display trained block_names in dropdown
        block_names = get_trained_block_names(sel_model_name)
        ##########################################################################
        if sel_block_name != DEFAULTS['sel_block_name']:
            ##########################################################################
            if sel_maction == 'Compare':
                button_class = 'button-off' # turn off all buttons not used for comparison
                num_comparisons = 5  # TODO this needs to be dynamically set
                palette = cycle(Category10[num_comparisons])
                #########################################################################
                # get ba_breakdown_avg_lines for all models in log that have trained to sel_block_name
                flavor = sel_model_name.split('_')[1]
                ba_breakdown_avg_lines = []
                for model_name in get_model_names_to_compare(flavor, sel_block_name):
                    database, trajdatabase = load_databases(model_name, sel_block_name)
                    ba_breakdown_avg_line, cats_sorted_by_ba_master = \
                        database.make_ba_breakdown_avg_line(cats_sorted_by_ba_master)
                    ba_breakdown_avg_lines.append(ba_breakdown_avg_line)
                #########################################################################
                # fig settings
                figsize = (12, 8)
                title_font_size = 16
                ax_font_size = 16
                leg_font_size = 10
                linewidth = 2.0
                ##########################################################################
                # fig
                fig, ax = plt.subplots(figsize=figsize)
                ##########################################################################
                # axes
                ax.set_xlabel('Categories', fontsize=ax_font_size)
                ax.set_ylabel('Balanced Accuracy (%)', fontsize=ax_font_size)
                ax.set_xticks(np.arange(len(database.probe_list)) + 0.5, minor=False)
                ax.set_xticklabels(cats_sorted_by_ba_master, minor=False, fontsize=leg_font_size, rotation=90)
                ax.set_xlim([0, len(database.cat_list) + 0.5])
                ax.yaxis.grid(True)
                ##########################################################################
                # plot
                num_cats = len(database.cat_list)
                x = range(num_cats)
                for n, ba_breakdown_avg_line in enumerate(ba_breakdown_avg_lines):
                    color = next(palette)
                    ax.plot(x, ba_breakdown_avg_line, '-', color=color, linewidth=1.0)
                    ax.plot(x, ba_breakdown_avg_line, '.', color=color, markersize=15.0, label='srn {}'.format(n))
                ##########################################################################
                plt.legend(fontsize=leg_font_size, loc='best')
                plt.tight_layout()
                ##########################################################################
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                ba_comparison_reference_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            elif sel_block_name == 'Trajectory': # this loads only trajectory data for faster browser experience
                database, trajdatabase = load_databases(sel_model_name, sel_block_name)
                ##########################################################################
                # make test_pp_traj_img
                print 'Making test_pp_traj_img'
                fig = trajdatabase.make_test_pp_traj_fig()
                p = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, save')
                p.y_range=Range1d(0, 500)
                p.xgrid.grid_line_color = None
                p.toolbar.logo = None
                p.plot_width = 1200
                test_pp_traj_img['script'], test_pp_traj_img['div'] = components(p)
                ##########################################################################
                # make avg_token_ba_traj_img
                print 'Making avg_token_ba_traj_img'
                fig = trajdatabase.make_avg_token_ba_traj_fig()
                p = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, save')
                p.y_range = Range1d(50, 80)
                p.xgrid.grid_line_color = None
                p.toolbar.logo = None
                p.plot_width=1200
                avg_token_ba_traj_img['script'], avg_token_ba_traj_img['div'] = components(p)
                ##################################q########################################
                # make avg_token_ba_traj_img
                print 'Making ba_pp_mw_corr_img'
                fig = trajdatabase.make_ba_pp_window_corr_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                ba_pp_mw_corr_img = base64.b64encode(figfile.getvalue())
            ##################################q########################################
            elif sel_cat == 'Custom List':
                database, trajdatabase = load_databases(sel_model_name, sel_block_name)
                custom_probes, num_probe_classes = get_custom_probes()
                palette = cycle(Category10[max(3, num_probe_classes)]) # category10 minimum is 3
                ##########################################################################
                # custom list token ba trajectories
                n = 0
                for probe_class, group in groupby(custom_probes, itemgetter(1)):
                    sel_probes = [i[0] for i in list(group)]
                    fig_, x, ys, _ = trajdatabase.make_token_ba_trajs_fig(sel_probes, sel_cat)
                    if n == 0: fig = fig_
                    n +=1
                    df = pd.DataFrame(ys)
                    avg_ys = df.mean()
                    std_ys = df.std()
                    n_ys = len(df)
                    se_ys = std_ys / (n_ys**0.5)
                    import scipy
                    confiv_ys = se_ys * scipy.stats.t._ppf((1+0.95)/2., n_ys-1)

                    ax = fig.gca()
                    ax.errorbar(x, avg_ys, yerr=confiv_ys, fmt='-o', c=next(palette), label='{}'.format(probe_class))
                ##########################################################################
                ax.set_ylim([40, 100])
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=14, loc='best')
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_ba_trajs_img = base64.b64encode(figfile.getvalue())
            ##################################q########################################
            elif sel_cat == DEFAULTS['sel_cat']:
                database, trajdatabase = load_databases(sel_model_name, sel_block_name)
                ##########################################################################
                # make cats to select from
                cats += database.cat_list
                cats2 += database.cat_list
                ##########################################################################
                # make ba_breakdown_scatter_img
                print 'Making ba_breakdown_scatter_img'
                fig = database.make_ba_breakdown_scatter_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                ba_breakdown_scatter_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make ba_breakdown_img
                print 'Making ba_breakdown_img'
                fig = database.make_ba_breakdown_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                ba_breakdow_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make_acts_2d_fig
                print 'Making acts_2d_img'
                fig = database.make_acts_2d_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                acts_2d_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make cat_sim_dh_img
                print 'Making cat_sim_dh_img'
                fig = database.make_cat_sim_dh_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                cat_sim_dh_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            elif sel_probe == DEFAULTS['sel_probe']:
                database, trajdatabase = load_databases(sel_model_name, sel_block_name)
                ##########################################################################
                # make cats to select from
                cats += database.cat_list
                cats2 += database.cat_list
                ##########################################################################
                # make probes to select from
                probes += database.cat_probe_list_dict[sel_cat]
                probes.sort()
                sel_probes = probes[1:] # removes 'Select' #TODO do i need to remove it?
                ##########################################################################
                # make neighbors_table_img
                print 'Making neighbors_table_img'
                fig = database.make_neighbors_table_fig(sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                neighbors_table_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make cat_cluster_img
                if sel_cat2 == DEFAULTS['sel_cat']:
                    print 'Making cat_cluster_img (1 category)'
                    fig = database.make_cat_cluster_fig(sel_cat)
                    figfile = StringIO.StringIO()
                    fig.savefig(figfile, format='png')
                    figfile.seek(0)
                    cat_cluster_img = base64.b64encode(figfile.getvalue())
                else: # cluster two categories
                    print 'Making cat_cluster_img (2 categories)'
                    fig = database.make_two_cat_cluster_fig(sel_cat, sel_cat2)
                    figfile = StringIO.StringIO()
                    fig.savefig(figfile, format='png')
                    figfile.seek(0)
                    cat_cluster_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make token_ba trajectories_fig
                print 'Making token_ba_trajs_img for probes in "{}"'.format(sel_cat)
                fig, x, ys, palette = trajdatabase.make_token_ba_trajs_fig(sel_probes, sel_cat)
                hover = HoverTool(
                    tooltips=[('block', '@block'), ('probe', '@probe'), ('balAcc', '$y')])
                p = mpl.to_bokeh(fig, tools=[hover, 'pan, wheel_zoom, crosshair, save'])
                for n, y in enumerate(ys):
                    source = ColumnDataSource(
                        data=dict(block=x, balAcc=y, probe=[sel_probes[n]]*len(x)))
                    circle = Line(x='block', y='balAcc', line_color=next(palette), line_width=2)
                    p.add_glyph(source, circle)
                p.y_range = Range1d(40, 100)
                p.xgrid.grid_line_color = None
                p.toolbar.logo = None
                p.plot_width = 1200
                token_ba_trajs_img['script'], token_ba_trajs_img['div'] = components(p)
                ##########################################################################
                # make cfreq_traj_fig
                print 'Making cfreq_traj_img for probes in "{}"'.format(sel_cat)
                fig = trajdatabase.make_cfreq_traj_fig(sel_probes, sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                cfreq_traj_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            else:
                database, trajdatabase = load_databases(sel_model_name, sel_block_name)
                ##########################################################################
                # make cats to select from
                cats += database.cat_list
                cats2 += database.cat_list
                ##########################################################################
                # make probes to select from
                probes += database.cat_probe_list_dict[sel_cat]
                probes.sort()
                ##########################################################################
                # make acts_dh_fig for a single token
                print 'Making (single probe) acts_dh_img'
                fig = database.make_acts_dh_fig(sel_probe)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_acts_dh_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make acts_dh_fig for all tokens
                print 'Making (all probes) acts_dh_img '
                fig = database.make_acts_dh_fig()
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                acts_dh_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make_token_corcoeff_hist_fig
                print 'Making token_corcoeff_hist_img'
                fig = database.make_token_corcoeff_hist_fig(sel_probe)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_corcoeff_hist_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            # close trajstore
            trajdatabase.trajstore.close()
    ##########################################################################
    else:
        print 'WARNING : No log entries found'
        headers = 'Log does not contain data for completed RNNs'.split()
    ##########################################################################
    # render to html
    return render_template('home.html',
                           button_class=button_class,
                           mactions=mactions,
                           hostname=socket.gethostname(),
                           bokeh_head=bokeh_head,
                           log_entries=log_entries,
                           headers=headers,
                           log_mtime=get_log_mtime(),
                           block_names=block_names,
                           probes=probes,
                           cats=cats,
                           cats2=cats2,
                           sel_block_name=sel_block_name,
                           sel_model_name=sel_model_name,
                           sel_probe=sel_probe,
                           sel_maction=sel_maction,
                           sel_cat=sel_cat,
                           sel_cat2=sel_cat2,
                           acts_2d_img=acts_2d_img,
                           token_acts_dh_img=token_acts_dh_img,
                           acts_dh_img=acts_dh_img,
                           token_corcoeff_hist_img=token_corcoeff_hist_img,
                           ba_breakdown_scatter_img=ba_breakdown_scatter_img,
                           ba_breakdow_img=ba_breakdow_img,
                           cat_cluster_img=cat_cluster_img,
                           cat_sim_dh_img=cat_sim_dh_img,
                           neighbors_table_img=neighbors_table_img,
                           token_ba_trajs_img=token_ba_trajs_img,
                           test_pp_traj_img=test_pp_traj_img,
                           avg_token_ba_traj_img=avg_token_ba_traj_img,
                           cfreq_traj_img=cfreq_traj_img,
                           ba_pp_mw_corr_img=ba_pp_mw_corr_img,
                           cats_sorted_by_ba_master=cats_sorted_by_ba_master,
                           ba_comparison_reference_img=ba_comparison_reference_img)


@app.route('/template/', methods=['GET', 'POST']) # TODO template for fig
def token_acts_dh_img():
    ##########################################################################
    # make fig
    fig = None
    # save fig to binary
    img = StringIO.StringIO()
    fig.savefig(img)
    img.seek(0)
    ##########################################################################
    return send_file(img, mimetype='image/png')


##########################################################################
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT' # for sessions
    app.run(port=5000, debug=True, host='0.0.0.0')
