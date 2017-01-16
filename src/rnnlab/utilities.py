import numpy as np
import os


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