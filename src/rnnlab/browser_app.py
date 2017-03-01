import socket
from wtforms import Form, TextAreaField, validators
from wtforms.validators import ValidationError
from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from bokeh.palettes import Category10
from itertools import cycle

from figs import *

runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))
app = Flask(__name__)
btn_name_desc_dict, btn_names_top, btn_names_bottom = make_btn_name_desc_dict()
app_headers = load_log_headers()
version = 'dev'
hostname = socket.gethostname()

@app.route('/', methods=['GET', 'POST'])
def log():
    ##########################################################################
    # get log entries
    log_entries, headers = load_filtered_log_entries(app_headers)
    if not log_entries:
        headers = 'Log is empty'.split()
    ##########################################################################
    template_dict = {key:value for key,value in
                     zip(['version','hostname','log_mtime'],
                         ['dev', socket.gethostname(), get_log_mtime()])}
    ##########################################################################
    # render to html
    return render_template('log.html',
                           template_dict=template_dict,
                           log_entries=log_entries,
                           headers=headers)


@app.route('/model/<string:model_name1>/<block_name1>/complete/', methods=['GET', 'POST'])
def complete(model_name1, block_name1):
    ##########################################################################
    template_dict = {key: value for key, value in
                     zip(['version', 'hostname', 'log_mtime'],
                         ['dev', socket.gethostname(), get_log_mtime()])}
    ##########################################################################
    # inits
    phrase = None
    output_dict = {}
    num_samples_list = [1, 10, 50]
    ##########################################################################
    # make form
    database = load_database(model_name1, block_name1)
    def vocab_validator(form, field):
        input_token_list = field.data.split()
        if any(map(lambda x: x not in database.token_list, input_token_list)):
            raise ValidationError('Not in vocabulary')

    class PhraseForm(Form):
        phrase = TextAreaField('Type your phrase here:', [validators.InputRequired(),
                                                          vocab_validator])
    form = PhraseForm(request.args)
    ##########################################################################
    # calc predicted_token
    if form.validate():
        phrase = request.args.get('phrase')
        for num_samples in num_samples_list:
            output_dict[num_samples] = complete_phrase(database, phrase, num_samples=num_samples)
    ##########################################################################
    # render to html
    return render_template('complete.html',
                           model_name1=model_name1,
                           block_name1=block_name1,
                           template_dict=template_dict,
                           form=form,
                           phrase=phrase,
                           output_dict=output_dict,
                           num_samples_list=num_samples_list)


@app.route('/model/<string:model_name1>/', methods=['GET', 'POST'])
def model(model_name1):
    ##########################################################################
    start = time.time()
    ##########################################################################
    print '-----------------------------'
    print model_name1
    print '-----------------------------'
    ##########################################################################
    template_dict = {key: value for key, value in
                     zip(['version', 'hostname', 'log_mtime'],
                         [version, hostname, get_log_mtime()])}
    ##########################################################################
    # get model_names2 and block_names2_dict
    mb_names1 = make_mb_names1(model_name1)
    print
    print 'mb_names1'
    print mb_names1
    comparison_dict_list, self_id = make_comparison_dict_list(model_name1, mb_names1, app_headers)
    print 'comparison_dict_list'
    for d in comparison_dict_list:
        print d
    #########################################################################
    if request.args.get('delete') is not None:
        if 'yes' == raw_input('Enter yes to delete {} :\n'.format(model_name1)):
            delete_model(model_name1)
        else:
            print 'Aborted deletion'
        return redirect(url_for('log'))
    #########################################################################
    elif request.args.get('complete') is not None:
        mb_name1 = get_mb_name_from_request(request, 'complete', mb_names1)
        return redirect(url_for('complete', model_name1=model_name1, block_name1=mb_name1))
    ##########################################################################
    elif request.args.get('avg_trajs') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'avg_trajs')
        palette = cycle(['green', 'orange'])
        model_group1, model_group2 = [database], [database]
        fig_tuple1 = (
        make_probes_ba_traj_fig(model_group1, model_group2, palette), 'mpl')  # TODO make model_group2 optional
        fig_tuple2 = (make_test_pp_traj_fig(model_group1, model_group2, palette), 'mpl')
        fig_tuple3 = (make_probe_pp_traj_fig([database], palette), 'mpl')
        fig_tuple4 = (make_ba_pp_window_corr_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3, fig_tuple4)
    ##########################################################################
    elif request.args.get('dim_red') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'dim_red')
        fig_tuple1 = (make_acts_2d_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('cat_sim') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'cat_sim')
        fig_tuple1 = (make_cat_sim_dh_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('ba_by_probe') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'ba_by_probe')
        fig_tuple1 = (make_ba_breakdown_annotated_fig(database), 'mpl')
        fig_tuple2 = (make_ba_breakdown_fig(database), 'mpl')
        fig_tuple3 = (make_ba_vs_pp_fig(database), 'mpl')
        fig_tuple4 = (make_pairplot_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3, fig_tuple4)
    ##########################################################################
    elif request.args.get('cluster') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'cluster')
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append((make_cat_cluster_fig(database, cat), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('neighbors') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'neighbors')
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append((make_neighbors_table_fig(database, cat), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('probe_trajs') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'probe_trajs')
        fig_tuples = []
        for cat in database.cat_list:
            cat_probes = database.cat_probe_list_dict[cat]
            fig_tuples.append((make_avg_probe_ba_trajs_fig(database, cat_probes), 'bokeh'))
            fig_tuples.append((make_avg_probe_pp_trajs_fig(database, cat_probes), 'bokeh'))
            fig_tuples.append((make_cfreq_traj_fig(database, cat_probes), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('cat_conf_mat') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'cat_conf_mat')
        fig_tuple1 = (make_cat_confusion_mat_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('c_probe_dh') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'c_probe_dh')
        fig_input = load_custom_fig_input('c_probe_dh')
        fig_tuples = []
        for probe in fig_input:
            fig_tuples.append((make_acts_dh_fig(database, probe), 'mpl'))
            databases = [load_database(model_name1, block_name) for block_name in database.get_saved_mb_names()]
            fig_tuples.append((make_token_acts_avg_act_corr_fig(databases, probe), 'mpl'))
            fig_tuples.append((make_probe_ba_vs_pp_fig(database, probe), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('loop_counters') is not None:
        model_name_id, block_name_id = [int(loop_counter) for loop_counter
                                        in request.args.get('loop_counters').split('_')]
        if model_name_id != self_id:
            model_names1 = comparison_dict_list[self_id]['model_names2']
            model_names2 = comparison_dict_list[model_name_id]['model_names2']
            mb_name1 = comparison_dict_list[self_id]['mb_names2'][block_name_id]
            mb_name2 = comparison_dict_list[model_name_id]['mb_names2'][block_name_id]
            assert mb_name1 == mb_name2
            # make model_groups
            model_group1, model_group2 = [], []
            for model_name1 in model_names1: model_group1.append(load_database(model_name1, mb_name1))
            for model_name2 in model_names2: model_group2.append(load_database(model_name2, mb_name2))
            # figs
            palette = cycle(['green', 'orange'])
            fig_tuple1 = (make_compare_ba_by_cat_fig(model_group1, model_group2, palette), 'mpl')
            fig_tuple2 = (make_probes_ba_traj_fig(model_group1, model_group2, palette), 'mpl')
            fig_tuple3 = (
            make_test_pp_traj_fig(model_group1, model_group2, palette), 'mpl')  # TODO test if sem works with more runs
            fig_tuple4 = (make_probe_sim_comp_fig(model_group1, model_group2, palette), 'mpl')
            fig_tuple5 = (make_neighbors_rbo_fig(model_group1, model_group2), 'mpl')
            fig_tuple6 = (make_cat_conf_diff_fig(model_group1, model_group2), 'mpl')
            fig_tuple7 = (make_pp_timing_ba_diff_corr_fig(model_group1, model_group2), 'mpl')
            fig_tuple8 = (make_probe_freq_ba_diff_corr_fig(model_group1, model_group2), 'mpl')
            fig_tuple9 = (make_avg_probe_pp_ba_diff_corr_fig(model_group1, model_group2), 'mpl')
            fig_tuple10 = (make_probe_doc_freq_ba_diff_corr_fig(model_group1, model_group2), 'mpl')

            # fig_tuples = []
            # for cat in model_group1[0].cat_list:
            #     fig_tuples.append((make_cat_probe_ba_comp_fig(model_group1, model_group2, cat), 'mpl'))


            imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3,
                             fig_tuple4, fig_tuple5, fig_tuple6,
                             fig_tuple7, fig_tuple8, fig_tuple9, fig_tuple10)  # *fig_tuples)
            imgs_desc = 'Model Class Comparison at Minibatch {:,}'.format(int(mb_name1))

        else:  # self model_group comparison
            mb_name1 = mb_names1[block_name_id]
            imgs_desc = 'Self Model Class Comparison at Minibatch {:,}'.format(int(mb_name1))
            imgs = None
    ##########################################################################
    elif request.args.get('comp_trajs') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'comp_trajs')
        compare_probes_tuples = load_compare_probes_tuples()  # TODO test this
        fig_tuple1 = (make_comp_probes_ba_fig(database, compare_probes_tuples), 'mpl')
        fig_tuple2 = (make_comp_binned_freqs_fig(database, compare_probes_tuples), 'mpl')
        imgs = make_imgs(fig_tuple1, fig_tuple2)
    ##########################################################################
    elif request.args.get('c_neighbors') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'c_neighbors')
        probes = load_custom_fig_input('c_neighbors')
        fig_tuple1 = (make_custom_neighbors_table_fig(database, probes), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('c_freq_hist') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'c_freq_hist')
        probes = load_custom_fig_input('c_freq_hist')
        fig_tuple1 = (make_probe_freq_hist_fig(database, probes), 'mpl')
        fig_tuple2 = (make_cat_count_pie_chart_fig(database), 'mpl')
        fig_tuple3 = (make_corpus_traj_fig(database), 'bokeh')
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3)
    ##########################################################################
    elif request.args.get('c_cluster') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'c_cluster')
        cats = load_custom_fig_input('c_cluster')
        custom_cats = map(lambda x: x.upper(), cats)
        fig_tuple1 = (make_custom_cat_clust_fig(database, custom_cats), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    # in case no img requested
    else:
        imgs_desc = None
        imgs = False
    ##########################################################################
    print 'Built page in {} secs'.format(time.time() - start)
    ##########################################################################
    return render_template('model.html',
                           template_dict=template_dict,
                           btn_names_top=btn_names_top,
                           btn_names_bottom=btn_names_bottom,
                           model_name1=model_name1,
                           mb_names1=mb_names1,
                           comparison_dict_list=comparison_dict_list,
                           imgs=imgs,
                           imgs_desc=imgs_desc)



##########################################################################
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT' # for sessions
    app.run(port=5000, debug=True, host='0.0.0.0')
