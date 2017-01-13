import os, datetime, pytz, csv, socket, base64
from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
import pandas as pd
import numpy as np
import StringIO
from database import DataBase
from rnnhelper import load_rc
##########################################################################
app = Flask(__name__)
##########################################################################
DEFAULTS = {'block_names': ['Select', '0000','0050', '1000', '2000', '2800'],
            'cats': ['Select','BODY', 'KITCHEN', 'MAMMAL', 'FAMILY', 'TOYS', 'BIRD',
                       'CLOTHING', 'NUMBERS','FURNITURE', 'MUSIC', 'INSECT', 'DAYS',
                       'TIMES', 'HOUSEHOLD', 'BATHROOM','DESSERT', 'TOOLS', 'ELECTRONICS',
                       'MONTHS', 'DRINK', 'SPACE', 'FRUIT','VEHICLES', 'WEATHER',
                       'VEGETABLE', 'MEAT', 'GAMES', 'PLANTS', 'SHAPE'],
            'sel_block_name': 'Select',
            'sel_cat': 'Select',
            'sel_probe': 'Select',
            'headers': ['model_name', 'num_iterations', 'num_hidden_units', 'best_token_ba'],
            'allow_incomplete' : True}
##########################################################################
log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
##########################################################################


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


def load_database(model_name, block_name):
    ##########################################################################
    print 'Loading database for {} {}...'.format(model_name, block_name)
    # load df
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    file_name = 'df_block_{}.h5'.format(block_name)
    print
    print os.path.join(path, file_name)
    print
    df = pd.read_hdf(os.path.join(path, file_name), 'df')
    # load configs_dict
    path = os.path.join(runs_dir, model_name, 'Configs')
    file_name = 'configs_dict.npy'
    configs_dict = dict(np.load(os.path.join(path, file_name)).item())
    # make database
    database = DataBase(configs_dict, df, block_name)
    ##########################################################################
    return database


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
    acts_2d_img = None
    token_acts_dh_img = None
    token_corcoeff_hist_img = None
    acts_dh_img = None
    ba_breakdown_scatter_img = None
    ba_breakdow_img = None
    cat_cluster_img = None
    probes = [DEFAULTS['sel_probe']]
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
        if sel_block_name != DEFAULTS['sel_block_name']:
            ##########################################################################
            # load database
            database = load_database(sel_model_name, sel_block_name) # TODO how to cache this?
            ##########################################################################
            # make ba_breakdown_scatter_img
            print 'Making ba_breakdown_scatter_img for: {} {}...'.format(sel_model_name, sel_block_name)
            fig = database.make_ba_breakdown_scatter_fig()
            figfile = StringIO.StringIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)
            ba_breakdown_scatter_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            # make ba_breakdown_img
            print 'Making ba_breakdown_img for: {} {}...'.format(sel_model_name, sel_block_name)
            fig = database.make_ba_breakdown_fig()
            figfile = StringIO.StringIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)
            ba_breakdow_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            # make_acts_2d_fig
            print 'Making acts_2d_img for: {} {}...'.format(sel_model_name, sel_block_name)
            fig = database.make_acts_2d_fig()
            figfile = StringIO.StringIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)
            acts_2d_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            if sel_cat != DEFAULTS['sel_cat']:
                ##########################################################################
                # make probes to select from
                probes += database.get_probes_from_cat(sel_cat)
                probes.sort()
                ##########################################################################
                # make cat_cluster_img
                fig = database.make_cat_cluster_fig(sel_cat)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                cat_cluster_img = base64.b64encode(figfile.getvalue())
            ##########################################################################
            if sel_probe != DEFAULTS['sel_probe']:
                ##########################################################################
                # make acts_dh_fig for a single token
                print 'Making acts_dh_img (single probe) for: {} {}...'.format(sel_model_name, sel_block_name)
                fig = database.make_acts_dh_fig(sel_probe)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_acts_dh_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make acts_dh_fig for all tokens
                print 'Making acts_dh_img (all probes) for: {} {}...'.format(sel_model_name, sel_block_name)
                fig = database.make_acts_dh_fig(probe=None)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                acts_dh_img = base64.b64encode(figfile.getvalue())
                ##########################################################################
                # make_token_corcoeff_hist_fig
                print 'Making token_corcoeff_hist_img for: {} {}...'.format(sel_model_name, sel_block_name)
                fig = database.make_token_corcoeff_hist_fig(sel_probe)
                figfile = StringIO.StringIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)
                token_corcoeff_hist_img = base64.b64encode(figfile.getvalue())
    ##########################################################################
    else:
        print 'WARNING : No log entries found'
        headers = 'Log does not contain data for completed RNNs'.split()
    ##########################################################################
    # render to html
    return render_template('home.html',
                           log_entries=log_entries,
                           headers=headers,
                           log_mtime=get_log_mtime(),
                           block_names=DEFAULTS['block_names'],
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
                           cat_cluster_img=cat_cluster_img)


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
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(port=5000, debug=True)

