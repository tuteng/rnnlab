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
import numpy as np
from bokeh.embed import components
from sklearn.feature_extraction.text import TfidfVectorizer



from dbutils import load_rnnlabrc
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


def make_tf_idf_mat(corpus, save_ev_block):
    ##########################################################################
    print 'Making tf-idf mat using train docs...'
    ##########################################################################
    # fit tf-idf model
    doc_str_list = [] # scikit requires a list of space-separated words
    tfidf = TfidfVectorizer(vocabulary=corpus.token_list)
    for block_name, doc_id in corpus.gen_train_block_name_and_id():
        doc_str = ' '.join(corpus.corpus_content[doc_id])
        doc_str_list.append(doc_str)
    tfidf.fit(doc_str_list)
    ##########################################################################
    # get values from save_ev_block
    doc_str_list = []
    for block_name, doc_id in corpus.gen_train_block_name_and_id():
        if int(block_name) % save_ev_block == 0:
            doc_str = ' '.join(corpus.corpus_content[doc_id])
            doc_str_list.append(doc_str)
    tf_idf_mat = tfidf.fit_transform(doc_str_list).toarray()
    ##########################################################################
    return tf_idf_mat


def make_probe_cf_traj_dict(corpus, save_ev_block):
    ##########################################################################
    print 'Making probe_cf_traj using train docs...'
    ##########################################################################
    # make dict
    probe_cf_traj_dict = {probe: np.zeros(corpus.num_total_train_docs) for probe in corpus.probe_list}
    ##########################################################################
    # collect probe frequency
    for block_name, doc_id in corpus.gen_train_block_name_and_id():
        doc_probe_list = corpus.corpus_content[doc_id]
        traj_id = int(block_name) - 1
        for probe in doc_probe_list:
            if probe in corpus.probe_id_dict:  probe_cf_traj_dict[probe][traj_id] += 1
    ##########################################################################
    # calc cumulative sum
    for probe, probe_freq_traj in probe_cf_traj_dict.iteritems():
        probe_cf_traj_dict[probe] = np.cumsum(probe_freq_traj)
    ##########################################################################
    # get values from save_ev_block
    for probe, probe_freq_traj in probe_cf_traj_dict.iteritems():
        cfs =[]
        for n, cf in enumerate(probe_cf_traj_dict[probe]):
            if n % save_ev_block == 0:
                cfs.append(cf)
        probe_cf_traj_dict[probe] = cfs
    ##########################################################################
    return probe_cf_traj_dict


def make_lex_div_traj(corpus, save_ev_block):
    ##########################################################################
    print 'Making lex_div_traj using train docs...'
    ##########################################################################
    lex_div_traj =[]
    for block_name, doc_id in corpus.gen_train_block_name_and_id():
        if int(block_name) % save_ev_block == 0:
            doc_probe_list = corpus.corpus_content[doc_id]
            num_unique_tokens = len(list(set(doc_probe_list)))
            num_total_tokens = len(doc_probe_list)
            lex_div = float(num_unique_tokens) / num_total_tokens
            lex_div_traj.append(lex_div)
    ##########################################################################
    return lex_div_traj

