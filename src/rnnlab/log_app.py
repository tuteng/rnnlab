import os, datetime, pytz, csv, socket, base64, random
from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
import pandas as pd
import numpy as np
import StringIO
from database import DataBase
from trajdatabase import TrajDataBase
from utilities import load_rc
from bokeh import mpl
from bokeh.models import Range1d
from bokeh.embed import components


##########################################################################
app = Flask(__name__)
##########################################################################
DEFAULTS = {'block_names': ['0001','0050', '1000', '2000', '2800'],
            'cats': ['Select','BODY', 'KITCHEN', 'MAMMAL', 'FAMILY', 'TOYS', 'BIRD',
                       'CLOTHING', 'NUMBERS','FURNITURE', 'MUSIC', 'INSECT', 'DAYS',
                       'TIMES', 'HOUSEHOLD', 'BATHROOM','DESSERT', 'TOOLS', 'ELECTRONICS',
                       'MONTHS', 'DRINK', 'SPACE', 'FRUIT','VEHICLES', 'WEATHER',
                       'VEGETABLE', 'MEAT', 'GAMES', 'PLANTS', 'SHAPE'],
            'sel_block_name': 'Select',
            'sel_cat': 'Select',
            'sel_probe': 'Select',
            'headers': ['model_name', 'learning_rate', 'num_iterations', 'best_token_ba'],
            'allow_incomplete' : True}
##########################################################################
log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
##########################################################################


def get_trained_block_names(model_name):
    ##########################################################################
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    ##########################################################################
    trained_block_names = ['Select']
    for b in DEFAULTS['block_names']:
        if os.path.isfile(os.path.join(path, 'df_block_{}.h5'.format(b))):
            trained_block_names.append(b)
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
    ##########################################################################
    return sel_model_name, sel_block_name, sel_probe, sel_cat


def load_databases(model_name, block_name):
    ##########################################################################
    # load configs
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Configs')
    file_name = 'configs_dict.npy'
    configs_dict = dict(np.load(os.path.join(path, file_name)).item())
    ##########################################################################
    # make database
    print 'Loading database for {} {}...'.format(model_name, block_name)
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    file_name = 'df_block_{}.h5'.format(block_name)
    df = pd.read_hdf(os.path.join(path, file_name), 'df')
    database = DataBase(configs_dict, df, block_name)
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
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        log_content = []
        for line in reader: log_content.append(line)
    headers = log_content[0]
    col_ids = [headers.index(header) for header in list(DEFAULTS['headers'])]
    filtered_headers = [i for n, i in enumerate(headers) if n in col_ids]
    complete = 0 if DEFAULTS['allow_incomplete'] else 1
    filtered_log_entries = [[i for n, i in enumerate(log_entry) if n in col_ids] for log_entry in log_content[1:]
                            if int(log_entry[-2]) == complete and socket.gethostname() in log_entry[0]]
    ##########################################################################
    return filtered_log_entries, filtered_headers


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
    block_names = DEFAULTS['block_names']
    acts_2d_img = None
    token_acts_dh_img = None
    token_corcoeff_hist_img = None
    acts_dh_img = None
    ba_breakdown_scatter_img = None
    ba_breakdow_img = None
    cat_cluster_img = None
    cat_sim_dh_img = None
    neighbors_table_img = None
    token_ba_trajs_img = None
    test_pp_traj_img = {}
    avg_token_ba_traj_img = {}
    cfreq_traj_img = None
    ba_pp_mw_corr_img = None
    ##########################################################################
    # load log entries and any requests
    print 'Loading home...'
    log_entries, headers = load_filtered_log_entries()
    sel_model_name, sel_block_name, sel_probe, sel_cat = get_requests()
    ##########################################################################
    if log_entries:
        ##########################################################################
        # if not model_name selected, select first as default
        if not sel_model_name: sel_model_name = log_entries[0][0]
        ##########################################################################
        # only display trained block_names in dropdown
        block_names = get_trained_block_names(sel_model_name)
        ##########################################################################
        if sel_block_name != DEFAULTS['sel_block_name']:
            ##########################################################################
            # load database
            database, trajdatabase = load_databases(sel_model_name, sel_block_name) # TODO how to cache this?
            ##########################################################################
            # make test_pp_traj_img
            print 'Making test_pp_traj_img'
            fig = trajdatabase.make_test_pp_traj_fig()
            p = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, hover')
            p.y_range=Range1d(0, 500)
            p.xgrid.grid_line_color = None
            p.toolbar.logo = None
            test_pp_traj_img['script'], test_pp_traj_img['div'] = components(p)
            ##########################################################################
            # make avg_token_ba_traj_img
            print 'Making avg_token_ba_traj_img'
            fig = trajdatabase.make_avg_token_ba_traj_fig()
            p = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, hover')
            p.y_range = Range1d(50, 80)
            p.xgrid.grid_line_color = None
            p.toolbar.logo = None
            avg_token_ba_traj_img['script'], avg_token_ba_traj_img['div'] = components(p)
            ##################################q########################################
            # make avg_token_ba_traj_img
            print 'Making ba_pp_mw_corr_img'
            fig = trajdatabase.make_ba_pp_window_corr_fig()
            figfile = StringIO.StringIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)
            ba_pp_mw_corr_img = base64.b64encode(figfile.getvalue())
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
            if sel_cat != DEFAULTS['sel_cat']:
                ##########################################################################
                # make probes to select from
                probes += database.cat_probe_list_dict[sel_cat]
                probes.sort()
                sel_probes = probes[1:] # removes 'Select'
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
                print 'Making cat_cluster_img'
                fig = database.make_cat_cluster_fig(sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                cat_cluster_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make token_ba trajectories_fig
                print 'Making token_ba_trajs_img for probes in "{}"'.format(sel_cat)
                fig = trajdatabase.make_token_ba_trajs_fig(sel_probes, sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_ba_trajs_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make cfreq_traj_fig
                print 'Making cfreq_traj_img for probes in "{}"'.format(sel_cat)
                fig = trajdatabase.make_cfreq_traj_fig(sel_probes, sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                cfreq_traj_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            if sel_probe != DEFAULTS['sel_probe']:
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
                           bokeh_head=bokeh_head,
                           log_entries=log_entries,
                           headers=headers,
                           log_mtime=get_log_mtime(),
                           block_names=block_names,
                           probes=probes,
                           cats=sorted(DEFAULTS['cats']),
                           sel_block_name=sel_block_name,
                           sel_model_name=sel_model_name,
                           sel_probe=sel_probe,
                           sel_cat=sel_cat,
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
                           ba_pp_mw_corr_img=ba_pp_mw_corr_img)


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
    app.run(port=5000, debug=True)

def start():
    app.run(port=5000, debug=True) # can set this to false for production