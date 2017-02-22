import time, socket, os
from wtforms import Form, TextAreaField, validators
from wtforms.validators import ValidationError
from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from bokeh.palettes import Category10
from itertools import cycle

from utils import get_log_mtime
from utils import make_block_names1
from utils import make_imgs
from utils import delete_model
from utils import load_custom_fig_input
from utils import load_filtered_log_entries
from utils import make_block_names2_dict
from utils import load_database
from utils import block_to_mb
from utils import load_rnnlabrc
from utils import complete_phrase
from utils import get_block_name_from_request
from utils import make_btn_name_desc_dict
from utils import load_database_and_img_desc
from utils import load_app_headers

from figs import make_neighbors_rbo_fig
from figs import make_custom_neighbors_table_fig
from figs import make_ba_bds_fig
from figs import make_probes_ba_traj_fig
from figs import make_test_pp_traj_fig
from figs import make_probe_sim_comp_fig
from figs import make_probe_freq_hist_fig
from figs import make_cat_count_pie_chart_fig
from figs import make_ba_pp_window_corr_fig
from figs import make_ba_breakdown_annotated_fig
from figs import make_ba_breakdown_fig
from figs import make_cat_sim_dh_fig
from figs import make_neighbors_table_fig
from figs import make_cat_cluster_fig
from figs import make_avg_probe_ba_trajs_fig
from figs import make_cfreq_traj_fig
from figs import make_cat_confusion_mat_fig
from figs import make_corpus_traj_fig
from figs import make_comp_probes_ba_fig
from figs import make_acts_dh_fig
from figs import make_token_acts_avg_act_corr_fig
from figs import make_acts_2d_fig
from figs import make_custom_cat_clust_fig
from figs import make_cat_conf_diff_fig
from figs import make_comp_binned_freqs_fig
from figs import make_cat_token_ba_comp_fig
from figs import make_num_clusters_ba_diff_corr_fig
from figs import make_pairplot_fig
from figs import make_probe_freq_ba_diff_corr_fig
from figs import make_avg_probe_pp_ba_diff_corr_fig
from figs import make_avg_probe_pp_trajs_fig
from figs import make_probe_pp_traj_fig
from figs import make_ba_vs_pp_fig
from figs import make_probe_ba_vs_pp_fig

runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))
app = Flask(__name__)
btn_name_desc_dict, btn_names_top, btn_names_bottom = make_btn_name_desc_dict()
app_headers = load_app_headers()
version = 'dev'

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
                         [version, socket.gethostname(), get_log_mtime()])}
    ##########################################################################
    # get model_names2 and block_names2_dict
    block_names1 = make_block_names1(model_name1)
    start = time.time()
    model_names2, block_names2_dict = make_block_names2_dict(model_name1, block_names1)  # TODO
    print 'Made blockname2dict in {}secs'.format(int(time.time() - start))
    #########################################################################
    if request.args.get('delete') is not None:
        if 'yes' == raw_input('Enter yes to delete {} :\n'.format(model_name1)):
            delete_model(model_name1)
        else:
            print 'Aborted deletion'
        return redirect(url_for('log'))
    #########################################################################
    elif request.args.get('complete') is not None:
        block_name1 = get_block_name_from_request(request, 'complete', block_names1)
        return redirect(url_for('complete', model_name1=model_name1, block_name1=block_name1))
    ##########################################################################
    elif request.args.get('avg_trajs') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'avg_trajs')
        num_comparisons = 1
        palette = cycle(Category10[max(3, num_comparisons)][:num_comparisons])
        fig_tuple1 = (make_probes_ba_traj_fig([database], palette), 'mpl')
        fig_tuple2 = (make_test_pp_traj_fig([database], palette), 'mpl')
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
            databases = [load_database(model_name1, block_name) for block_name in database.get_saved_block_names()]
            fig_tuples.append((make_token_acts_avg_act_corr_fig(databases, probe), 'mpl'))
            fig_tuples.append((make_probe_ba_vs_pp_fig(database, probe), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('comp2models') is not None:
        comp2models_request = request.args.get('comp2models').split('+')  # TODO use hidden html buttons
        block_name1_id = int(comp2models_request[0])
        block_name1 = block_names1[block_name1_id]
        model_name2, block_name2 = comp2models_request[1], comp2models_request[2]
        database1 = load_database(model_name1, block_name1)
        database2 = load_database(model_name2, block_name2)
        databases = [database1, database2]
        mb = block_to_mb(model_name1, block_name1)
        imgs_desc = 'Model Comparison at Minibatch {:,}'.format(mb)
        num_comparisons = 2
        palette = cycle(Category10[max(3,num_comparisons)][:num_comparisons])
        fig_tuple1 = (make_ba_bds_fig(databases, palette), 'mpl')
        fig_tuple2 = (make_probes_ba_traj_fig(databases, palette), 'mpl')
        fig_tuple3 = (make_test_pp_traj_fig(databases, palette), 'mpl')
        fig_tuple4 = (make_probe_sim_comp_fig(databases, palette), 'mpl')
        probes = load_custom_fig_input('comp2models')
        # fig_tuple5 = (make_neighbors_rbo_fig(databases, probes), 'bokeh')
        # fig_tuple6 = (make_cat_conf_diff_fig(databases), 'mpl')
        # fig_tuple7 = (make_num_clusters_ba_diff_corr_fig(databases), 'mpl')
        # fig_tuple8 = (make_probe_freq_ba_diff_corr_fig(databases), 'mpl')
        # fig_tuple9 = (make_avg_probe_pp_ba_diff_corr_fig(databases), 'mpl')
        fig_tuples = []
        # for cat in database1.cat_list:
        # fig_tuples.append((make_cat_token_ba_comp_fig(databases, cat), 'mpl'))
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3,
                         fig_tuple4)  # , fig_tuple5, fig_tuple6, fig_tuple7,
        # fig_tuple8, fig_tuple9) #, *fig_tuples)
    ##########################################################################
    elif request.args.get('comp_trajs') is not None:
        database, imgs_desc = load_database_and_img_desc(model_name1, request, 'comp_trajs')
        compare_probes_tuples = load_compare_probes_tuples()  # TODO
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
                           block_names1=block_names1,
                           model_names2=model_names2,
                           block_names2_dict=block_names2_dict,
                           imgs=imgs,
                           imgs_desc=imgs_desc)



##########################################################################
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT' # for sessions
    app.run(port=5000, debug=True, host='0.0.0.0')
