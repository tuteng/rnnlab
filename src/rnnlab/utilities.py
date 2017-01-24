import csv
import time
import sys
import subprocess
import shutil
import numpy as np
import os


def remove_log_entry(model_name):
    ##########################################################################
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    log_content = csv.reader(open(log_path, 'r'))
    ##########################################################################
    # get all entries except for that belonging to model_name
    runs_log_content_new = []
    for row in log_content:
        if not 'model_name' in row:
            model_name_ = row[0]
            if model_name_ != model_name:
                runs_log_content_new.append(row)
        elif 'model_name' in row:
            runs_log_content_new.append(row)
    ##########################################################################
    time.sleep(1)
    with open(log_path, 'w') as f:
        writer = csv.writer(f)
        for row in runs_log_content_new:
            writer.writerow(row)



def remove_model_data(model_name):
    ##########################################################################
    runs_dir = os.path.abspath(load_rc('runs_dir'))
    path_to_delete = os.path.join(runs_dir, model_name)
    if '' == raw_input('Press ENTER to delete {}'.format(path_to_delete)):
        shutil.rmtree(path_to_delete)



def is_completed(model_name):
    ##########################################################################
    log_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_log.csv'))
    ##########################################################################
    log_content = csv.reader(open(log_path, 'r'))
    for row in log_content:
        if row[0] == model_name:
            is_completed = bool(row[-2])
            ##########################################################################
            return is_completed


def check_disk_space(runs_dir):
    ##########################################################################
    df = subprocess.Popen(["df", "{}".format(runs_dir)], stdout=subprocess.PIPE)
    df_str = df.communicate()[0]
    used = int(df_str.split('\n')[1].split()[4].strip('%'))
    ##########################################################################
    print 'Checking disk space of filesystem containing runs_dir:'
    print df_str
    ##########################################################################
    if used > 90: sys.exit('rnnlab: Disk space usage > 90%')


def load_token_data(runs_dir, model_name):
    ##########################################################################
    path = os.path.join(runs_dir, model_name, 'Token_Data')
    file_name = 'token_data.npz'.format(model_name)
    npzfile = np.load(os.path.join(path, file_name))
    ##########################################################################
    # load
    token_list, token_id_dict = npzfile['token_list'].tolist(), npzfile['token_id_dict'].item()
    probe_list, probe_id_dict = npzfile['probe_list'].tolist(), npzfile['probe_id_dict'].item()
    probe_cat_dict = npzfile['probe_cat_dict'].item()
    cat_list = npzfile['cat_list'].tolist()
    cat_probe_list_dict = {cat: [probe for probe in probe_list if probe_cat_dict[probe] == cat]
                           for cat in cat_list}
    probe_cf_traj_dict = npzfile['probe_cf_traj_dict'].item()
    ##########################################################################
    return token_list, token_id_dict, probe_list, probe_id_dict,\
           probe_cat_dict, cat_list, cat_probe_list_dict, probe_cf_traj_dict



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





def calc_probe_sim_mat(all_acts_df, probe_list):
    ##########################################################################
    print 'Calculating probe simmat...'
    ##########################################################################
    # calc sim mat
    probe_simmat = np.asarray(all_acts_df.T.corr(method='pearson'))
    assert probe_simmat.shape == (len(probe_list), len(probe_list))
    nan_ids = np.where(np.isnan(probe_simmat).all(axis=1))[0]
    assert len(nan_ids) == 0
    ##########################################################################
    return probe_simmat


def calc_ba_mats(probe_list, cat_list, probe_cat_dict, probe_simmat, thr_list, output, verbose=False):
    ##########################################################################
    ba_mat = None
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
        ##########################################################################
        # calc hits, misses, false alarms, correct rejections
        for token_id_1 in range(num_probes):
            token_1 = probe_list[token_id_1]
            cat_1 = probe_cat_dict[token_1]
            cat_id_1 = cat_list.index(cat_1)

            for token_id_2 in range(num_probes):
                if token_id_1 != token_id_2:
                    token_2 = probe_list[token_id_2]
                    cat_2 = probe_cat_dict[token_2]
                    sim = probe_simmat[token_id_1, token_id_2]

                    if sim != 'nan':
                        if cat_1 == cat_2:
                            if sim > thr:
                                category_hits[cat_id_1, n] += 1
                                item_hits[token_id_1, n] += 1
                            else:
                                category_misses[cat_id_1, n] += 1
                                item_misses[token_id_1, n] += 1
                        else:
                            if sim > thr:
                                category_false_alarms[cat_id_1, n] += 1
                                item_false_alarms[token_id_1, n] += 1
                            else:
                                category_correct_rejections[cat_id_1, n] += 1
                                item_correct_rejections[token_id_1, n] += 1
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
            for token_id in range(num_probes):

                current_hits = item_hits[token_id, n]
                current_misses = item_misses[token_id, n]
                current_false_alarms = item_false_alarms[token_id, n]
                current_correct_rejections = item_correct_rejections[token_id, n]

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

                token_ba_mat[token_id, n] = current_item_BA
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
            current_threshold_BA_mean = np.nanmean(cat_ba_mat[:, n])
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
    return ba_mat



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


def gen_user_configs():
    ##########################################################################
    # define directories
    working_dir = os.path.dirname(os.path.abspath(__file__))
    rnn_dir = os.path.abspath(working_dir + os.sep + '..')
    user_configs_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'rnnlab_user_configs.csv'))
    if not os.path.isfile(user_configs_path): sys.exit('rnnlab: {} not found'.format(user_configs_path))
    ##########################################################################
    # check if model has already been trained for given user_config
    # TODO it would be cool if given a set of configurations, a unique id could be assigned
    # TODO which would alert the user anytime they run a configuration that has been run in the past
    ##########################################################################
    # check that there are no duplicated configs
    reader = csv.reader(open(os.path.join(rnn_dir, user_configs_path), 'r'))
    rows = []
    for n, row in enumerate(reader):
        if n != 0: rows.append(tuple(row))
    if len(set(rows)) != len(rows): print 'rnnlab WARNING: Duplicate configs detected in {}'.format(user_configs_path)
    ##########################################################################
    # gen user_configs (tuple)
    reader = csv.reader(open(os.path.join(rnn_dir, user_configs_path), 'r'))
    for n, row in enumerate(reader):
        if n == 0:
            configs_names = row
        else:
            user_configs = [(name, config) for name, config in zip(configs_names, row)]
            ##########################################################################
            yield user_configs


def load_rc(string): # .rnnlabrc file should specify gpu/cpu and runs_dir path
    ##########################################################################
    # load rc from file
    rc = None
    with open(os.path.join(os.path.expanduser('~'),'.rnnlabrc'), 'r') as f:
        for line in f.readlines():
            if line.startswith(string):
                rc = line.split()[1]
    if rc is None:
        sys.exit('rnnlab: Did not find "{}" in .rnnlabrc'.format(rc))
    ##########################################################################
    return rc


def get_childes_data(): # downloads childes data from github - not used
    ##########################################################################
    print 'Downloading childes data to {}...'.format(os.getcwd())
    ##########################################################################
    import requests
    if not os.path.isdir('data'): os.mkdir('data')
    os.chdir('data')
    ##########################################################################
    for dir, file_names in [('childes2_3YO', ['vocab_3YO_4238.txt', 'corpus.txt']), ('probes',['semantic.txt'])]:
        if not os.path.isdir(dir): os.mkdir(dir)
        os.chdir(dir)
        print 'Donwloading {}'.format(','.join(file_names))
        for file_name in file_names:
            r = requests.get('https://raw.githubusercontent.com/phueb/rnnlab/master/src/rnnlab/data/{}/{}'
                             .format(dir, file_name))
            with open(file_name,'w') as f:
                f.write(r.text)
        os.chdir('..')
    os.chdir('..')