import os
import csv
import random
import time
import multiprocessing as mp
import shutil
import sys
import datetime
import StringIO
import subprocess
import base64
import pandas as pd
import numpy as np
from bokeh.embed import components
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


from database import DataBase
from database import load_rnnlabrc


runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))


def gen_neighbor_name_and_sim(neighbors_for_probe):
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
    if not os.path.isfile(user_configs_path): sys.exit('rnnlab: {} not found'.format(user_configs_path))
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


def get_log_mtime():  # TODO test this
    ##########################################################################
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    log_mtime_unformatted = datetime.datetime.fromtimestamp(os.path.getmtime(log_path))
    log_mtime = log_mtime_unformatted.strftime("%Y-%m-%d %H:%M")
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


def load_database(model_name, block_name=None):
    ##########################################################################
    # load configs
    configs_dict = load_configs_dict(model_name)
    ##########################################################################
    # make database
    path = os.path.join(runs_dir, model_name, 'Data_Frame')
    if block_name is None:
        file_name = 'df_block_{}.h5'.format(to_block_name(0))
    else:
        file_name = 'df_block_{}.h5'.format(block_name)
    df = pd.read_hdf(os.path.join(path, file_name), 'df')
    database = DataBase(configs_dict, df, block_name)
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


def make_block_names1(database, step=7):
    ##########################################################################
    block_names_to_display = database.get_saved_block_names()[0::step]
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
        database = load_database(model_name2)
        saved_block_names2 = database.get_saved_block_names()
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


def make_probe_cf_traj_dict(corpus, save_ev_block, min_probe_freq=10):
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
    # check that probes occur at least min_probe_freq
    if min_probe_freq is not None:
        for probe in corpus.probe_list:
            if probe_cf_traj_dict[probe][-1] < min_probe_freq:
                print 'rnnlab WARNING: {} occurs less than {} times in training docs'.format(probe, min_probe_freq)
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
        sys.exit('RNN flavor not recognized.')
    ##########################################################################
    return rnn


def make_cat_conf_mat(database):
    ##########################################################################
    path = os.path.join(runs_dir, database.model_name, 'Balanced_Accuracy')
    file_name = 'cat_confusion_mat_data_block_{}.npz'.format(database.block_name)
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
    """
    :param model_name: model_name
    :param block_name: block_name
    :param phrase: string containing tokens (validated by vocab_validator)
    :param num_words: number of downstream words
    :param num_outputs: number of output predictions
    :param num_samples: number of random samples before selection
    :param sort_column: sort output by this column
    :param exc_criteria: exclusion criteria. 0=xxx and UNKNOWN. 1 adds punctuation. 2 adds PERIOD
    :return: output_list: list containing num_outputs phrases (string)
    """
    ##########################################################################
    # restore rnn
    rnn_graph = create_rnn_graph(database.num_input_units, database.configs_dict)
    rnn_graph.saver.restore(rnn_graph.sess, os.path.join(runs_dir, database.model_name, 'Weights',
                                                         'weights_at_block_{}.ckpt'.format(database.block_name)))
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
    tf.reset_default_graph()  # TODO can i reset variables without importing tensorlfow ?
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


def calc_token_ba_list(database, num_samples=None, thr_step=0.001, verbose=False):
    ##########################################################################
    # calc simmat
    probe_simmat, probe_simmat_labels = calc_probe_sim_mat(database, num_samples)
    ##########################################################################
    print 'Calculating balanced accuracy...'
    ##########################################################################
    # make thr_ranges
    num_cpus = 6
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
    if mp.cpu_count() < num_cpus:
        print 'rnnlab WARNING: CPU Count is < 6. Parallel calculation of token_ba may not work'
    if verbose:
        print 'Calculating token ba using {} processes with {} steps...'.format(num_cpus, thr_step)
        print 'Threshold Ranges:'
        for i in thr_lists: print i, '\n'
    ##########################################################################
    # calc ba and get cat confusion mat data
    start = time.time()
    pool = mp.Pool(processes=num_cpus)
    async_results = [pool.apply_async(calc_ba_mats, args=(probe_simmat_labels,
                                                          database.cat_list,
                                                          database.probe_cat_dict,
                                                          probe_simmat,
                                                          thr_list,
                                                          'token')) for thr_list in thr_lists]
    ba_mats = [result.get()[0] for result in async_results]
    ba_mat = np.hstack((mat for mat in ba_mats))
    cat_confusion_mat_data_list_of_lists = [result.get()[1] for result in async_results]
    cat_confusion_mat_data_list = [item for sublist in cat_confusion_mat_data_list_of_lists for item in sublist]
    pool.close()
    ##########################################################################
    print 'Took {} minutes to calc ba'.format(abs(time.time() - start) / 60.)
    ##########################################################################
    # make token_ba_list
    token_ba_mat_col_means = np.nanmean(ba_mat, 0)
    best_token_ba_mat_col_id = np.argmax(token_ba_mat_col_means)
    token_ba_list = np.multiply(ba_mat[:, best_token_ba_mat_col_id], 100).tolist()
    tmp_df = pd.DataFrame(data={'probe': probe_simmat_labels,
                                'token_ba': token_ba_list})
    token_ba_list = tmp_df.groupby('probe').mean()['token_ba'].values
    ##########################################################################
    # save confusion data
    cat_confusion_mat_data = cat_confusion_mat_data_list[best_token_ba_mat_col_id]
    runs_dir = load_rnnlabrc('runs_dir')
    path = os.path.join(runs_dir, database.model_name, 'Balanced_Accuracy')
    file_name = 'cat_confusion_mat_data_block_{}.npz'.format(database.block_name)
    np.savez(os.path.join(path, file_name),
             hits_by_cat_dict=cat_confusion_mat_data[0],
             fas_by_cat_dict=cat_confusion_mat_data[1])
    ##########################################################################
    return token_ba_list  # TODO change this to return non-averaged ba, AND averaged ba


def calc_probe_sim_mat(database, num_samples=None, method='pearson'):  # TODO num_samples should be in configs
    ##########################################################################
    print 'Calculating probe simmat...'
    ##########################################################################
    # calc sim mat
    all_acts_df, all_acts_labels = database.get_all_acts_df(num_samples)
    probe_simmat = all_acts_df.T.corr(method=method).values
    probe_simmat_labels = all_acts_labels
    nan_ids = np.where(np.isnan(probe_simmat).all(axis=1))[0]
    assert len(nan_ids) == 0
    ##########################################################################
    return probe_simmat, probe_simmat_labels


def calc_cat_sim_mat(database):
    ##########################################################################
    # probe simmat
    probe_simmat, _ = calc_probe_sim_mat(database)
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


def calc_ba_mats(probe_list, cat_list, probe_cat_dict, probe_simmat, thr_list, output, verbose=False):
    ##########################################################################
    # inits
    assert len(probe_list) == len(probe_simmat)
    import pyprind
    pbar = pyprind.ProgBar(len(thr_list))
    ba_mat = None
    cat_confusion_mat_data_list = []
    num_probes = len(probe_list)
    num_cats = len(cat_list)
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
        pbar.update()
        ##########################################################################
        hits_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in cat_list} for cat_1 in cat_list}
        fas_by_cat_dict = {cat_1: {cat_2: 0 for cat_2 in cat_list} for cat_1 in cat_list}
        ##########################################################################
        # calc hits, misses, false alarms, correct rejections
        for i in range(num_probes):
            token_1 = probe_list[i]
            cat_1 = probe_cat_dict[token_1]
            cat_id_1 = cat_list.index(cat_1)

            for j in range(num_probes):
                if i != j:
                    token_2 = probe_list[j]
                    cat_2 = probe_cat_dict[token_2]
                    sim = probe_simmat[i, j]

                    if sim != 'nan':
                        if cat_1 == cat_2:
                            if sim > thr:
                                category_hits[cat_id_1, n] += 1
                                item_hits[i, n] += 1
                                hits_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                            else:
                                category_misses[cat_id_1, n] += 1
                                item_misses[i, n] += 1
                        else:
                            if sim > thr:
                                category_false_alarms[cat_id_1, n] += 1
                                item_false_alarms[i, n] += 1
                                fas_by_cat_dict[cat_1][cat_2] += 1  # for confusion matrix data
                            else:
                                category_correct_rejections[cat_id_1, n] += 1
                                item_correct_rejections[i, n] += 1
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

                token_ba_mat[i, n] = current_item_BA
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
            d_prime, beta, c, ad = calculate_dprime(
                current_hit_sums, current_miss_sums, current_false_alarm_sums, current_correct_rejection_sums)
            current_threshold_BA_mean = np.nanmean(ba_mat[:, n])
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
        # save hits and false alarms to cat_confusion_mat_data_list
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


def get_block_name_from_request(request, arg, block_names1):
    ##########################################################################
    if request.args.get('complete') == 'traj':
        block_name1 = block_names1[-1]
    else:
        block_name1 = request.args.get(arg)
    ##########################################################################
    return block_name1


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


def calc_num_probe_acts_clusters(database, probe, method='gap'):
    ##########################################################################
    acts_mat = database.get_token_acts_df(probe).values
    if method == 'elbow':
        lnk0 = linkage(pdist(acts_mat))
        acceleration = np.diff(lnk0[-10:, :], 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        num_probe_acts_clusters = acceleration_rev.argmax() + 2
    else:
        raise NotImplementedError
    ##########################################################################
    return num_probe_acts_clusters


def plot_best_fit_line(ax, xys, fontsize):
    ##########################################################################
    x, y = zip(*xys)
    best_fit_fxn = np.polyfit(x, y, 1, full=True)
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]
    ax.plot(xl, yl, linewidth=2.0, c='red')
    ##########################################################################
    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=4)
    ax.text(0.01, 0.9, '$R^2$ = {}'.format(Rsqr), fontsize=fontsize)
