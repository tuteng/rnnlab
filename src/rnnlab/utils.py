import os
import csv
import time
import multiprocessing as mp
import shutil
import sys
import datetime
import StringIO
import subprocess
import base64
from operator import itemgetter
import pandas as pd
import numpy as np
from itertools import groupby
from bokeh.embed import components
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


from database import DataBase
from database import load_corpus_data
from database import load_rnnlabrc


runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))


def gen_neighbor_name_and_sim(neighbors_for_probe):  # TODO get rid of this
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


def gen_user_configs():
    ##########################################################################
    # define directories
    user_configs_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_user_configs.csv'))
    if not os.path.isfile(user_configs_path):
        raise AttributeError('rnnlab: {} not found'.format(user_configs_path))
    ##########################################################################
    # check that there are no duplicated configs
    reader = csv.reader(open(user_configs_path, 'r'))
    rows = []
    configs_names = []
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


def to_mb_name(num_mbs):
    ##########################################################################
    return ('0000000' + str(num_mbs))[-6:]


def mb_name_exists(model_name, mb_name):
    ##########################################################################
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    if os.path.isfile(os.path.join(path, 'df_mb_{}.h5'.format(mb_name))):
        return True
    else:
        return False


def get_log_mtime():
    ##########################################################################
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    log_mtime_unformatted = datetime.datetime.fromtimestamp(os.path.getmtime(log_path))
    log_mtime = log_mtime_unformatted.strftime("%Y-%m-%d %H:%M")
    ##########################################################################
    return log_mtime


def load_custom_fig_input(btn_name):
    ##########################################################################
    tuples = []
    with open(os.path.join('static', 'custom_fig_input.txt'), 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                probe = line.split()[0]
                probe_class = line.split()[1]
                tuples.append((probe, probe_class))
    ##########################################################################
    fig_input = [tuple[0] for tuple in tuples if tuple[1] == btn_name]
    ##########################################################################
    return fig_input


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
    if used > 90: sys.exit('rnnlab: Disk space usage > 90%.')


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


def load_database(model_name, mb_name=None):
    ##########################################################################
    # load configs
    configs_dict = load_configs_dict(model_name)
    ##########################################################################
    # make database
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    if mb_name is None:
        file_name = 'df_mb_{}.h5'.format(to_mb_name(0))
    else:
        file_name = 'df_mb_{}.h5'.format(mb_name)
    df = pd.read_hdf(os.path.join(path, file_name), 'df')
    database = DataBase(configs_dict, df, mb_name)
    ##########################################################################
    return database





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
    log_entries_list, headers = load_log()
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
    log_entries_list, headers = load_log()
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


def load_log():
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


def load_filtered_log_entries(app_headers, allow_incomplete=True):
    ##########################################################################
    # inits
    log_entries_list, log_headers = load_log()
    num_app_headers = len(app_headers)
    c = [0, 1] if allow_incomplete else [1]
    ##########################################################################
    # make filtered_headers
    col_ids = [log_headers.index(app_header) for app_header in app_headers
               if app_header in log_headers]
    filtered_headers_unsorted = [i for n, i in enumerate(log_headers) if n in col_ids]
    sort_dict = {header: i for header, i in zip(app_headers, range(num_app_headers))}
    filtered_headers = sorted(filtered_headers_unsorted, key=sort_dict.get)
    ##########################################################################
    # filter log entries
    sort_list = [app_headers.index(filtered_header) for filtered_header in filtered_headers_unsorted]
    filtered_log_entries = [[tuple[1] for tuple in
                             sorted(zip(sort_list, [i for n, i in enumerate(log_entry)
                                                    if n in col_ids]))]
                            for log_entry in log_entries_list
                            if int(log_entry[-2]) in c
                            and float(log_entry[-1]) >= 50.0]
    filtered_log_entries = filtered_log_entries[::-1]
    ##########################################################################
    return filtered_log_entries, filtered_headers


def make_mb_names1(model_name, num_display=5):
    ##########################################################################
    # saved mb names
    saved_mb_names = get_saved_mb_names_fast(model_name)
    ##########################################################################
    # inits
    saved_mbs = [int(mb_name) for mb_name in saved_mb_names]
    save_ev_mb = int(load_configs_dict(model_name)['save_ev_mb'])
    stop_mb = int(load_corpus_data(model_name, 'stop_mb'))
    ##########################################################################
    # make candidates
    mb1_candidates = np.arange(0, stop_mb, save_ev_mb)
    ########################################################################## # TODO does this work?
    # make mb_names1
    step = len(mb1_candidates) / num_display
    sliced = mb1_candidates[::step]
    filtered = filter(lambda x: x in saved_mbs, sliced)
    mb_names1 = [to_mb_name(mb) for mb in filtered]
    ##########################################################################
    # if empty, get last
    if not mb_names1: mb_names1 = [saved_mb_names[-1]]
    ##########################################################################
    return mb_names1


def make_comparison_dict_list(model_name1, mb_names1, log_headers, limit_to_same_flavor=False):
    ##########################################################################
    # make model_names2_tuples
    num_reps1 = load_configs_dict(model_name1)['num_reps']
    if limit_to_same_flavor: flavors = [model_name1.split('_')[1]]
    else: flavors = ['srn', 'lstm', 'irnn', 'scrn']
    headers = ['num_reps'] + log_headers  # num reps must be first in list
    filtered_log_entries, filtered_headers = load_filtered_log_entries(headers)
    model_names2_tuples = [(log_entry[1], '+'.join(log_entry[2:-2])) for log_entry in filtered_log_entries
                           if log_entry[1].split('_')[1] in flavors
                           and int(log_entry[0]) == num_reps1]  # compare only models with same num_reps
    model_names2_tuples = sorted(model_names2_tuples, key=itemgetter(1))  # required for groupby
    ##########################################################################
    # make comparison_dict_list
    mbs1_set = set(mb_names1)
    comparison_dict_list = []
    for mclass, group in groupby(model_names2_tuples, itemgetter(1)):
        model_names2 = [tuple[0] for tuple in list(group)]
        ##########################################################################
        # make comparison_dict
        saved_mb_names2 = get_saved_mb_names_fast(model_names2[0])
        mbs2_set = set(saved_mb_names2)
        mb_names2 = sorted(list(mbs1_set.intersection(mbs2_set)))
        is_self = True if model_name1 in model_names2 else False
        comparison_dict = {'mb_names2': mb_names2,
                           'model_names2': model_names2,
                           'self': is_self,
                           'mclass': mclass}
        ##########################################################################
        # add dict to list
        comparison_dict_list.append(comparison_dict)
    ##########################################################################
    # sort by time
    comparison_dict_list = sorted(comparison_dict_list,
                                  key=lambda x: x['model_names2'][0], reverse=True)  # this is working
    ##########################################################################
    # get self_id
    self_id = None
    for n, comparison_dict in enumerate(comparison_dict_list):
        if comparison_dict['self']:
            self_id = n
            break
    ##########################################################################
    return comparison_dict_list, self_id


def get_saved_mb_names_fast(model_name):
    ##########################################################################
    ba_trajdf_path = os.path.join(runs_dir, model_name, 'Data_Frame', 'ba_trajdf.h5')
    with pd.HDFStore(ba_trajdf_path, mode='r') as store:
        saved_mb_names = store.select_column('trajdf', 'index').values
    ##########################################################################
    # sort
    saved_mb_names = sorted(saved_mb_names)
    ##########################################################################
    return saved_mb_names


def calc_neighbors_rbo(neighbors1, neighbors2, p=0.98):
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


def make_tf_idf_mat(corpus, save_ev_mb=None):
    ##########################################################################
    print 'Making tf-idf mat using train docs...'
    ##########################################################################
    # fit tf-idf model
    if save_ev_mb is None: save_ev_mb = 1
    doc_str_list = [] # scikit requires a list of space-separated words
    tfidf = TfidfVectorizer(vocabulary=corpus.token_list)
    for doc_id in corpus.gen_train_doc_id(num_epochs=1):
        doc_str = ' '.join(corpus.train_doc_token_lists[doc_id])
        doc_str_list.append(doc_str)
    tfidf.fit(doc_str_list)
    ##########################################################################
    # get values at saved timepoints
    doc_str_list = []
    for n, doc_id in enumerate(corpus.gen_train_doc_id()):
        if (n * corpus.num_mbs_in_doc) % save_ev_mb == 0:
            doc_str = ' '.join(corpus.train_doc_token_lists[doc_id])
            doc_str_list.append(doc_str)
    tf_idf_mat = tfidf.fit_transform(doc_str_list).toarray()
    ##########################################################################
    return tf_idf_mat


def make_probe_cf_traj_dict(corpus, save_ev_mb, min_probe_freq=10,
                            verbose=False):  # TODO link this number to num_ba_samples
    ##########################################################################
    print 'Making probe_cf_traj using train docs...'
    ##########################################################################
    # make dict
    probe_cf_traj_dict = {probe: np.zeros(corpus.num_blocks) for probe in corpus.probe_list}
    ##########################################################################
    # collect probe frequency
    for n, doc_id in enumerate(corpus.gen_train_doc_id()):
        doc_probe_list = corpus.train_doc_token_lists[doc_id]
        for probe in doc_probe_list:
            if probe in corpus.probe_id_dict:  probe_cf_traj_dict[probe][n] += 1
    ##########################################################################
    # copy
    probe_doc_freq_dict = probe_cf_traj_dict.copy()
    ##########################################################################
    # calc cumulative sum
    for probe, probe_freq_traj in probe_cf_traj_dict.iteritems():
        probe_cf_traj_dict[probe] = np.cumsum(probe_freq_traj).astype(np.int)
    ##########################################################################
    # check that probes occur at least min_probe_freq
    num_probe_occurences = 0
    if min_probe_freq is not None:
        for probe in corpus.probe_list:
            probe_freq = probe_cf_traj_dict[probe][-1] / corpus.num_epochs
            num_probe_occurences += probe_freq
            if verbose: print probe, probe_freq
            if probe_freq < min_probe_freq:
                print 'rnnlab WARNING: {} occurs less than {} times in training docs'.format(probe, min_probe_freq)
    ##########################################################################
    # get values at saved timepoints
    for probe, probe_freq_traj in probe_cf_traj_dict.iteritems():
        cfs =[]
        for n, cf in enumerate(probe_cf_traj_dict[probe]):
            if (n * corpus.num_mbs_in_doc) % save_ev_mb == 0:
                cfs.append(cf)
        probe_cf_traj_dict[probe] = cfs
    ##########################################################################
    return probe_cf_traj_dict, num_probe_occurences, probe_doc_freq_dict


def make_lex_div_traj(corpus, save_ev_mb=None):
    ##########################################################################
    print 'Making lex_div_traj using train docs...'
    ##########################################################################
    if save_ev_mb is None: save_ev_mb = 1
    lex_div_traj =[]
    for n, doc_id in enumerate(corpus.gen_train_doc_id()):
        if (n * corpus.num_mbs_in_doc) % save_ev_mb == 0:
            doc_probe_list = corpus.train_doc_token_lists[doc_id]
            num_unique_tokens = len(list(set(doc_probe_list)))
            num_total_tokens = len(doc_probe_list)
            lex_div = float(num_unique_tokens) / num_total_tokens
            lex_div_traj.append(lex_div)
    ##########################################################################
    return lex_div_traj


def create_rnn_graph(num_input_units, configs_dict):
    ##########################################################################
    flavor = configs_dict['flavor']
    ##########################################################################
    # init model
    if flavor == 'lstm':
        from lstm import LSTM
        rnn = LSTM(num_input_units, configs_dict)
    elif flavor == 'irnn':
        from irnn import IRNN
        rnn = IRNN(num_input_units, configs_dict)
    elif flavor == 'srn':
        from srn import SRN
        rnn = SRN(num_input_units, configs_dict)
    elif flavor == 'scrn':
        from scrn import SCRN
        rnn = SCRN(num_input_units, configs_dict)
    else:
        raise AttributeError('rnnlab: RNN flavor not recognized.')
    ##########################################################################
    return rnn


def make_cat_conf_mat(database):
    ##########################################################################
    path = os.path.join(runs_dir, database.model_name, 'Balanced_Accuracy')
    file_name = 'cat_confusion_mat_data_mb_{}.npz'.format(database.mb_name)
    npzfile = np.load(os.path.join(path, file_name))
    hits_by_cat_dict = npzfile['hits_by_cat_dict'].item()
    fas_by_cat_dict = npzfile['fas_by_cat_dict'].item()
    num_cats = len(fas_by_cat_dict)
    cat_list = database.cat_list
    ##########################################################################
    # make confusion mat
    cat_conf_mat = np.zeros((num_cats, num_cats), dtype=float)
    for row_id, row_cat in enumerate(cat_list):
        for col_id, col_cat in enumerate(cat_list):
            num_probes_row_cat = len(database.cat_probe_list_dict[row_cat])
            num_probes_col_cat = len(database.cat_probe_list_dict[col_cat])
            n = num_probes_row_cat * num_probes_col_cat - num_probes_row_cat
            if row_id == col_id:  # hits
                hits = float(hits_by_cat_dict[row_cat][col_cat])
                cat_conf_mat[row_id, col_id] = hits / n * 100
            else:  # fas
                fas = float(fas_by_cat_dict[row_cat][col_cat])
                cat_conf_mat[row_id, col_id] = fas / n * 100
    ##########################################################################
    # mask
    mask = np.zeros_like(cat_conf_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    ##########################################################################
    return cat_conf_mat, mask


def get_excluded_tokens(token_id_dict, exc_criteria):
    ##########################################################################
    if exc_criteria == 1:
        return [token_id_dict.get('PERIOD'),
                token_id_dict.get('UNKNOWN'),
                token_id_dict.get('COMMA'),
                token_id_dict.get('EXCLAIM'),
                token_id_dict.get('QUESTION'),
                token_id_dict.get('xxx')]
    elif exc_criteria == 2:
        return [token_id_dict.get('UNKNOWN'),
                token_id_dict.get('COMMA'),
                token_id_dict.get('EXCLAIM'),
                token_id_dict.get('QUESTION'),
                token_id_dict.get('xxx')]
    elif exc_criteria == 0:
        return [token_id_dict.get('xxx'),
                token_id_dict.get('UNKNOWN')]


def complete_phrase(database, phrase, num_words=4, num_outputs=5,
                    num_samples=50, sort_column=None, exc_criteria=0):
    ##########################################################################
    import tensorflow as tf
    ##########################################################################
    # restore rnn
    rnn_graph = create_rnn_graph(database.num_input_units, database.configs_dict)
    rnn_graph.saver.restore(rnn_graph.sess, os.path.join(runs_dir, database.model_name, 'Weights',
                                                         'weights_at_mb_{}.ckpt'.format(database.mb_name)))
    ##########################################################################
    # inits
    bptt_steps = database.configs_dict['bptt_steps']
    excluded_words = get_excluded_tokens(database.token_id_dict, exc_criteria)
    ##########################################################################
    # calc multiple phrases
    output_list = []
    for i in range(num_outputs):
        ##########################################################################
        # calc single phrase
        tokens_in_phrase = phrase.split()
        num_tokens_in_phrase = len(tokens_in_phrase)
        new_phrase = [[database.token_id_dict[x]] for x in tokens_in_phrase]
        while not len(new_phrase) == num_words + len(tokens_in_phrase):
            ##########################################################################
            # get softmax probs
            X = np.asarray(new_phrase)[-bptt_steps:].transpose()
            [softmax_probs] = rnn_graph.sess.run(rnn_graph.softmax_probs, feed_dict={rnn_graph.x: X})
            ##########################################################################
            # calc new token and add to phrase
            samples = np.zeros([database.num_input_units], float)
            total_samples = 0
            while total_samples < num_samples:
                softmax_probs[0] -= sum(softmax_probs[:]) - 1.0  # need to ompensate for float arithmetic
                new_sample = np.random.multinomial(1, softmax_probs)
                current_index = np.argmax(new_sample)
                if not current_index in excluded_words:
                    samples += new_sample
                    total_samples += 1
            sampled_probe_id = np.argmax(samples)
            new_phrase.append(np.asarray([sampled_probe_id]))
        ##########################################################################
        # convert phrase to string and add to output_list
        phrase_str = ' '.join([database.token_list[token_id[0]] for token_id in new_phrase[num_tokens_in_phrase:]])
        output_list.append(phrase_str)
    ##########################################################################
    # sort
    if sort_column is not None: output_list.sort(key=lambda x: x[sort_column])
    ##########################################################################
    rnn_graph.sess.close()
    tf.reset_default_graph()
    ##########################################################################
    return output_list


def calc_hca(database, acts_cols, cat_col, epochs=30):
    ########################################################################################
    print 'Calculating classifier accuracy...'
    ########################################################################################
    # make data for classifier
    x_data = acts_cols
    y_data = np.zeros(len(cat_col))
    for n, cat in enumerate(cat_col): y_data[n] = database.cat_list.index(cat)
    assert len(x_data) == len(y_data)
    ########################################################################################
    # classifier # TODO make sure classifier works
    from classifier import calc_hca
    train_hca, test_hca = calc_hca(database.model_name, x_data, y_data, epochs)
    ########################################################################################
    return train_hca, test_hca


def calc_ba_list(database, num_ba_samples, ba_method='exemplar', thr_step=0.001):
    ##########################################################################
    # make thr_ranges
    num_cpus = 3
    thr_start, thr_end = 0.7, 1.0  # TODO how flexible is this?
    thr_num_steps = round(((thr_end - thr_start) / thr_step) / num_cpus, 2)
    thr_lists = []
    while True:
        thr_list = np.arange(thr_start, thr_start + thr_num_steps * thr_step, thr_step)
        thr_lists.append(thr_list)
        thr_start = float(thr_list[-1])
        time.sleep(0.5)
        if round(thr_end, 2) == round(thr_start, 2):  # TODO can i use numpy equal approximation?
            break
    ##########################################################################
    start = time.time()
    pool = mp.Pool(processes=num_cpus)
    ##########################################################################
    # exemplar
    if ba_method == 'exemplar' and num_ba_samples == 0:
        probes_acts_df = database.get_probes_acts_df(num_ba_samples)
        probe_simmat = calc_probe_sim_mat(probes_acts_df)
        sampled_probes_list = database.probe_list
        async_results = [pool.apply_async(calc_ba_mat_exemplar,
                                          args=(database, thr_list, probe_simmat))
                         for thr_list in thr_lists]
    ##########################################################################
    # prototype
    elif ba_method == 'prototype':
        cat_prototypes_mat = database.get_cat_prototypes_df().values
        probes_acts_df = database.get_probes_acts_df(num_ba_samples)
        probe_acts_mat = probes_acts_df.values
        sampled_probes_list = probes_acts_df.index.tolist()
        async_results = [pool.apply_async(calc_ba_mat_prototype,
                                          args=(database, thr_list, cat_prototypes_mat,
                                                probe_acts_mat, sampled_probes_list))
                         for thr_list in thr_lists]
    else:
        raise AttributeError(
            'rnnlab: Use either "prototype" or "exemplar" as method for calculating balanced accuracy.'
            'Only "prototype" supports using num_ba_samples > 0')
    ##########################################################################
    print 'Calculating balanced accuracy...'
    ##########################################################################
    # get async results
    ba_mats = [result.get()[0] for result in async_results]
    ba_mat = np.hstack((mat for mat in ba_mats))
    cat_confusion_mat_data_list_of_lists = [result.get()[1] for result in async_results]
    cat_confusion_mat_data_list = [item for sublist in cat_confusion_mat_data_list_of_lists for item in sublist]
    pool.close()
    ##########################################################################
    print 'Took {} minutes to calc ba'.format(abs(time.time() - start) / 60.)
    ##########################################################################
    # make probe_ba_list
    token_ba_mat_col_means = np.nanmean(ba_mat, 0)
    best_token_ba_mat_col_id = np.argmax(token_ba_mat_col_means)
    probe_ba_list = np.multiply(ba_mat[:, best_token_ba_mat_col_id], 100).tolist()
    ##########################################################################
    # make avg_probe_ba_list
    avg_probe_ba_list = pd.DataFrame(
        data={'probe': sampled_probes_list,
              'probe_ba': probe_ba_list}).groupby('probe').mean()['probe_ba'].values.tolist()
    ##########################################################################
    # save confusion data
    cat_confusion_mat_data = cat_confusion_mat_data_list[best_token_ba_mat_col_id]
    runs_dir = load_rnnlabrc('runs_dir')
    path = os.path.join(runs_dir, database.model_name, 'Balanced_Accuracy')
    file_name = 'cat_confusion_mat_data_mb_{}.npz'.format(database.mb_name)
    np.savez(os.path.join(path, file_name),
             hits_by_cat_dict=cat_confusion_mat_data[0],
             fas_by_cat_dict=cat_confusion_mat_data[1])
    ##########################################################################
    return probe_ba_list, avg_probe_ba_list, sampled_probes_list


def calc_probe_sim_mat(probes_acts_df):
    ##########################################################################
    print 'Calculating probe simmat...'
    ##########################################################################
    # calc sim mat
    probe_simmat = probes_acts_df.T.corr().values
    nan_ids = np.where(np.isnan(probe_simmat).all(axis=1))[0]
    assert len(nan_ids) == 0
    ##########################################################################
    return probe_simmat


def calc_cat_sim_mat(database):
    ##########################################################################
    # probe simmat
    probes_acts_df = database.get_probes_acts_df()
    probe_simmat = calc_probe_sim_mat(probes_acts_df)
    ##########################################################################
    # inits
    num_probes = len(database.probe_list)
    num_cats = len(database.cat_list)
    ##########################################################################
    # make category sim dict
    cat_sim_dict = {cat_outer: {cat_inner: [] for cat_inner in database.cat_list}
                    for cat_outer in database.cat_list}
    for i in range(num_probes):
        probe1 = database.probe_list[i]
        cat1 = database.probe_cat_dict[probe1]
        for j in range(num_probes):
            if i != j:
                probe2 = database.probe_list[j]
                cat2 = database.probe_cat_dict[probe2]
                sim = probe_simmat[i, j]
                cat_sim_dict[cat1][cat2].append(sim)
    ##########################################################################
    # make category simmat
    cat_simmat = np.zeros([num_cats, num_cats], float)
    for i in range(num_cats):
        cat1 = database.cat_list[i]
        for j in range(num_cats):
            cat2 = database.cat_list[j]
            sims = np.array(cat_sim_dict[cat1][cat2])  # this contains a list of sims
            sim_mean = sims.mean()
            cat_simmat[database.cat_list.index(cat1), database.cat_list.index(cat2)] = sim_mean
    ##########################################################################
    return cat_simmat


def calc_ba_mat_exemplar(database, thr_list, probe_simmat):
    ##########################################################################
    # inits
    import pyprind
    pbar = pyprind.ProgBar(len(thr_list))
    cat_confusion_mat_data_list = []
    num_probes = len(database.probe_list)
    num_cats = len(database.cat_list)
    num_thrs = len(thr_list)
    ##########################################################################
    # initialisations
    item_hits = np.zeros([num_probes, num_thrs], float)
    item_misses = np.zeros([num_probes, num_thrs], float)
    item_false_alarms = np.zeros([num_probes, num_thrs], float)
    item_correct_rejections = np.zeros([num_probes, num_thrs], float)
    ba_mat = np.zeros([num_probes, num_thrs], float)
    ##########################################################################
    for n, thr in enumerate(thr_list):
        pbar.update()
        ##########################################################################
        hits_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in database.cat_list} for cat_1 in database.cat_list}
        fas_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in database.cat_list} for cat_1 in database.cat_list}
        ##########################################################################
        # calc hits, misses, false alarms, correct rejections
        for i in range(num_probes):
            token_1 = database.probe_list[i]
            cat_1 = database.probe_cat_dict[token_1]

            for j in range(num_probes):
                if i != j:
                    token_2 = database.probe_list[j]
                    cat_2 = database.probe_cat_dict[token_2]
                    sim = probe_simmat[i, j]

                    if sim != 'nan':
                        if cat_1 == cat_2:
                            if sim > thr:
                                item_hits[i, n] += 1
                                hits_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                            else:
                                item_misses[i, n] += 1
                        else:
                            if sim > thr:
                                item_false_alarms[i, n] += 1
                                fas_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                            else:
                                item_correct_rejections[i, n] += 1
        ##########################################################################
        # calc token balanced accuracy
        for i in range(num_probes):

            current_hits = item_hits[i, n]
            current_misses = item_misses[i, n]
            current_false_alarms = item_false_alarms[i, n]
            current_correct_rejections = item_correct_rejections[i, n]

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

            ba_mat[i, n] = current_item_BA
        ##########################################################################
        # save hits and false alarms to cat_confusion_mat_data_list
        cat_confusion_mat_data_list.append((hits_by_cat_dict, fas_by_cat_dict))
    ##########################################################################
    return ba_mat, cat_confusion_mat_data_list


def calc_ba_mat_prototype(database, thr_list, cat_prototypes_mat, probe_acts_mat, sampled_probes_list):
    ##########################################################################
    # inits
    import pyprind
    pbar = pyprind.ProgBar(len(thr_list))
    cat_confusion_mat_data_list = []
    num_probes = len(sampled_probes_list)
    num_cats = len(database.cat_list)
    num_thrs = len(thr_list)
    ##########################################################################
    # initialisations
    item_hits = np.zeros([num_probes, num_thrs], float)
    item_misses = np.zeros([num_probes, num_thrs], float)
    item_false_alarms = np.zeros([num_probes, num_thrs], float)
    item_correct_rejections = np.zeros([num_probes, num_thrs], float)
    ba_mat = np.zeros([num_probes, num_thrs], float)
    ##########################################################################
    for n, thr in enumerate(thr_list):
        pbar.update()
        ##########################################################################
        hits_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in database.cat_list} for cat_1 in database.cat_list}
        fas_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in database.cat_list} for cat_1 in database.cat_list}
        ##########################################################################
        # calc hits, misses, false alarms, correct rejections
        for i in range(num_probes):
            probe = sampled_probes_list[i]
            cat_1 = database.probe_cat_dict[probe]
            probe_act = probe_acts_mat[i]

            for j in range(num_cats):
                cat_2 = database.cat_list[j]
                cat_prototype_act = cat_prototypes_mat[j]
                sim = np.corrcoef(cat_prototype_act, probe_act)[1, 0]

                if cat_1 == cat_2:
                    if sim > thr:
                        item_hits[i, n] += 1
                        hits_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                    else:
                        item_misses[i, n] += 1
                else:
                    if sim > thr:
                        item_false_alarms[i, n] += 1
                        fas_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                    else:
                        item_correct_rejections[i, n] += 1
        ##########################################################################
        # calc token balanced accuracy
        for i in range(num_probes):

            current_hits = item_hits[i, n]
            current_misses = item_misses[i, n]
            current_false_alarms = item_false_alarms[i, n]
            current_correct_rejections = item_correct_rejections[i, n]

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

            ba_mat[i, n] = current_item_BA
        ##########################################################################
        # collect hits and false alarms
        cat_confusion_mat_data_list.append((hits_by_cat_dict, fas_by_cat_dict))
    ##########################################################################
    return ba_mat, cat_confusion_mat_data_list


def calculate_dprime(hits, misses, fas, crs):
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


def get_mb_name_from_request(request, arg, mb_names1):
    ##########################################################################
    if request.args.get(arg) == 'traj':
        mb_name = mb_names1[-1]
    else:
        mb_name = request.args.get(arg)
    ##########################################################################
    return mb_name


def make_hc_fig(database):  # TODO
    ##########################################################################
    # calc and add hca train and test values to new entry
    acts_cols = database.df.filter(regex='H').values
    cat_col = database.df['cat'].values
    train_hca, test_hca = calc_hca(database, acts_cols, cat_col)
    ##########################################################################


def load_num_synsets(probe, verbose=False):
    ##########################################################################
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(probe)
    num_synsets = len(synsets)
    if verbose:
        for i, j in enumerate(synsets):
            print "Meaning", i, "NLTK ID:", j.name()
            print "Definition:", j.definition()
    ##########################################################################
    return num_synsets


def calc_num_probe_acts_clusters(database, probe, min_num_acts=100, method='elbow'):
    ##########################################################################
    print 'Calculating number of clusters for {}...'.format(probe)
    ##########################################################################
    acts_mat = database.get_probe_acts_df(probe).values
    if method == 'elbow':
        if len(acts_mat) > min_num_acts:
            lnk0 = linkage(pdist(acts_mat))
            acceleration = np.diff(lnk0[-10:, :], 2)  # 2nd derivative of the distances
            acceleration_rev = acceleration[::-1]
            num_probe_acts_clusters = acceleration_rev.argmax() + 2
        else:
            num_probe_acts_clusters = 2
    else:
        raise NotImplementedError
    ##########################################################################
    return num_probe_acts_clusters


def plot_best_fit_line(ax, xys, fontsize, color='red', linewidth=2.0, zorder=3, x_pos=0.05, y_pos=0.9):
    ##########################################################################
    x, y = zip(*xys)
    best_fit_fxn = np.polyfit(x, y, 1, full=True)
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]
    ax.plot(xl, yl, linewidth=linewidth, c=color, zorder=zorder)
    ##########################################################################
    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=2)
    if Rsqr > 0.5: fontsize += 5
    ax.text(x_pos, y_pos, '$R^2$ = {}'.format(Rsqr), transform=ax.transAxes, fontsize=fontsize)


def make_btn_name_desc_dict():
    ##########################################################################
    # parse buttons.txt
    btn_name_desc_tuple = []
    btn_name_list = []
    with open(os.path.join('static', 'buttons.txt'), 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                btn_name = line.split()[0]
                desc = ' '.join(line.split()[1:-1]).strip("'")
                btn_name_list.append(btn_name)
                btn_name_desc_tuple.append((btn_name, desc))
    ##########################################################################
    # make dict
    btn_name_desc_dict = {btn: desc for btn, desc in btn_name_desc_tuple}
    ##########################################################################
    # top and bottom
    btn_names_top = btn_name_list[:8]
    btn_names_bottom = btn_name_list[8:]
    ##########################################################################
    return btn_name_desc_dict, btn_names_top, btn_names_bottom


def load_database_and_img_desc(model_name, request, btn_name):
    ##########################################################################
    start = time.time()
    btn_name_desc_dict, btn_names_top, btn_names_bottom = make_btn_name_desc_dict()
    mb_names1 = make_mb_names1(model_name)
    mb_name = get_mb_name_from_request(request, btn_name, mb_names1)
    database = load_database(model_name, mb_name)
    desc = btn_name_desc_dict[btn_name]
    imgs_desc = '{} {}'.format(desc, mb_name)
    print 'Loaded database and img_desc in {} secs'.format(time.time() - start)
    ##########################################################################
    return database, imgs_desc


def load_log_headers():
    ##########################################################################
    app_headers = []
    with open(os.path.join('static', 'app_headers.txt'), 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                app_header = line.split()[0]
                app_headers.append(app_header)
    ##########################################################################
    return app_headers


def load_compare_probes_tuples():
    ##########################################################################
    compare_probes_tuples = []
    with open(os.path.join('static', 'compare_probes.txt'), 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                probe = line.split()[0]
                group = line.split()[1]
                compare_probes_tuples.append((probe, group))
    ##########################################################################
    return compare_probes_tuples


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    ##########################################################################
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    ##########################################################################
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def get_neighbors_from_probe_simmat(database, probe_simmat, probe, num_neighbors):  # TODO use this in neighbors figures
    ##########################################################################
    probe_id = database.probe_id_dict[probe]
    neighbor_tuples = [(target, sim) for target, sim in zip(database.probe_list, probe_simmat[probe_id])]
    neighbor_tuples_sorted = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[1:num_neighbors]
    neighbors = [tuple[0] for tuple in neighbor_tuples_sorted]
    ##########################################################################
    return neighbors
