import time, socket, os
from wtforms import Form, TextAreaField, validators
from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from bokeh.palettes import Category10
from itertools import cycle
import numpy as np
import tensorflow as tf

from utils import get_log_mtime
from utils import get_block_names_to_display
from utils import make_imgs
from utils import delete_model
from utils import load_custom_probes_tuples
from utils import load_filtered_log_entries
from utils import make_block_names2_dict
from utils import load_database
from utils import load_trajdatabase
from utils import block_to_iteration
from utils import create_rnn_graph
from utils import load_rnnlabrc
from utils import load_configs_dict
from dbutils import load_token_data
from dbutils import load_corpus_data

from figs import make_neighbors_rbo_fig
from figs import make_custom_neighbors_table_fig
from figs import make_ba_bds_fig
from figs import make_avg_ba_traj_fig
from figs import make_test_pp_trajs_fig
from figs import make_probe_sim_comp_fig
from figs import make_probe_freq_hist_fig
from figs import make_cat_count_pie_chart_fig
from figs import make_ba_pp_window_corr_fig
from figs import make_ba_breakdown_scatter_fig
from figs import make_ba_breakdown_fig
from figs import make_cat_sim_dh_fig
from figs import make_neighbors_table_fig
from figs import make_cat_cluster_fig
from figs import make_token_ba_trajs_fig
from figs import make_cfreq_traj_fig
from figs import make_cat_confusion_mat_fig
from figs import make_corpus_traj_fig
from figs import make_compprobes_fig
from figs import make_acts_dh_fig
from figs import make_token_corcoeff_hist_fig
from figs import make_acts_2d_fig
from figs import make_custom_cat_clust_fig
from figs import make_cat_conf_diff_fig
from figs import make_comp_binned_freqs_fig

runs_dir = os.path.abspath(load_rnnlabrc('runs_dir'))

app = Flask(__name__)

btn_names_top = ['avgtrajBtn', 'dimredBtn', 'catsimBtn', 'bdBtn',
                 'clustBtn', 'neighborsBtn', 'batrajBtn', 'catconfBtn']

btn_names_bottom = ['compprobes', 'customn', 'custompdh', 'freqhist',
                    'customclust', 'delete', 'complete']

headers_to_display = ['model_name', 'block_order',
                      'num_reps', 'num_iterations', 'completed', 'best_token_ba']


class PhraseForm(Form):
    ##########################################################################
    phrase = TextAreaField('Type your phrase here:', [validators.DataRequired()])


@app.route('/', methods=['GET', 'POST'])
def log():
    ##########################################################################
    # get log entries
    log_entries, headers = load_filtered_log_entries(headers_to_display)
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
    # make form
    phrase = None
    predicted_token = None
    form = PhraseForm(request.args)
    if form.validate():
        ##########################################################################
        # calc predicted_token
        phrase = request.args.get('phrase')
        token_id_dict = load_token_data(model_name1)[1]
        token_list = load_token_data(model_name1)[0]
        X = np.asarray([[token_id_dict[probe] for probe in phrase.split()]])
        configs_dict = load_configs_dict(model_name1)
        num_input_units = load_corpus_data(model_name1)[4]
        rnn_graph = create_rnn_graph(num_input_units, configs_dict)
        rnn_graph.saver.restore(rnn_graph.sess, os.path.join(runs_dir, model_name1, 'Weights',
                                                             'weights_at_block_{}.ckpt'.format(block_name1)))
        softmax_probs = rnn_graph.sess.run(rnn_graph.softmax_probs,
                                           feed_dict={rnn_graph.x: X}).tolist()
        token_id = np.argmax(softmax_probs)
        predicted_token = token_list[token_id]
        rnn_graph.sess.close()
        tf.reset_default_graph()
    ##########################################################################
    # render to html
    return render_template('complete.html',
                           model_name1=model_name1,
                           block_name1=block_name1,
                           template_dict=template_dict,
                           form=form,
                           phrase=phrase,
                           predicted_token=predicted_token)


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
                         ['dev', socket.gethostname(), get_log_mtime()])}
    ##########################################################################
    # get model_names2 and block_names2_dict
    block_names1 = get_block_names_to_display(model_name1)
    model_names2, block_names2_dict = make_block_names2_dict(model_name1, block_names1)
    #########################################################################
    if request.args.get('delete') is not None:
        if 'yes' == raw_input('Enter yes to delete {} :\n'.format(model_name1)):
            delete_model(model_name1)
        else:
            print 'Aborted deletion'
        return redirect(url_for('log'))
    #########################################################################
    elif request.args.get('complete') is not None:
        block_name1 = request.args.get('complete')
        return redirect(url_for('complete', model_name1=model_name1, block_name1=block_name1))
    ##########################################################################
    elif request.args.get('avgtrajBtn') == 'traj':
        imgs_desc = 'Average Trajectories'
        trajdatabase = load_trajdatabase(model_name1)
        num_comparisons = 1
        palette = cycle(Category10[max(3, num_comparisons)][:num_comparisons])
        fig_tuple1 = (make_avg_ba_traj_fig([trajdatabase], palette), 'mpl')
        fig_tuple2 = (make_test_pp_trajs_fig([trajdatabase], palette), 'mpl')
        fig_tuple3 = (make_ba_pp_window_corr_fig(trajdatabase), 'mpl')
        trajdatabase.trajstore.close()
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3)
    ##########################################################################
    elif request.args.get('dimredBtn') not in ['traj', None]:
        block_name1 = request.args.get('dimredBtn')
        imgs_desc = 'Dimensionality Reduction Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuple1 = (make_acts_2d_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('catsimBtn') not in ['traj', None]:
        block_name1 = request.args.get('catsimBtn')
        imgs_desc = 'Category Similarity Heatmap-Dendrogram Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuple1 = (make_cat_sim_dh_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('bdBtn') not in ['traj', None]:
        block_name1 = request.args.get('bdBtn')
        imgs_desc = 'Balanced Accuracy Breakdown by Probe Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuple1 = (make_ba_breakdown_scatter_fig(database), 'mpl')
        fig_tuple2 = (make_ba_breakdown_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1, fig_tuple2)
    ##########################################################################
    elif request.args.get('clustBtn') not in ['traj', None]:
        block_name1 = request.args.get('clustBtn')
        imgs_desc = 'Activation States Clustering Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append((make_cat_cluster_fig(database, cat), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('neighborsBtn') not in ['traj', None]:
        block_name1 = request.args.get('neighborsBtn')
        imgs_desc = 'Nearest Neighbors Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append((make_neighbors_table_fig(database, cat), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('batrajBtn') == 'traj':
        imgs_desc = 'Token Balanced Accuracy Trajectories'
        trajdatabase = load_trajdatabase(model_name1)
        fig_tuples = []
        for cat in trajdatabase.cat_list:
            cat_probes = trajdatabase.cat_probe_list_dict[cat]
            fig_tuples.append((make_token_ba_trajs_fig(trajdatabase, cat_probes), 'bokeh'))
            fig_tuples.append((make_cfreq_traj_fig(trajdatabase, cat_probes), 'mpl'))
        trajdatabase.trajstore.close()
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('catconfBtn') not in ['traj', None]:
        block_name1 = request.args.get('catconfBtn')
        imgs_desc = 'Category Confusion Matrix Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        fig_tuple1 = (make_cat_confusion_mat_fig(database), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('custompdh') not in ['traj', None]:
        block_name1 = request.args.get('custompdh')
        imgs_desc = 'Activations Dendrogram-Heatmap Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        custom_probes_tuples = load_custom_probes_tuples()
        custom_tuples = [tuple[0] for tuple in custom_probes_tuples if tuple[1] == 'custompdh']
        fig_tuples = []
        for custom_probe in custom_tuples:
            fig_tuples.append((make_acts_dh_fig(database, custom_probe), 'mpl'))
            fig_tuples.append((make_token_corcoeff_hist_fig(database, custom_probe), 'mpl'))
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    elif request.args.get('comp2models') not in ['traj', None]:
        comp2models_request = request.args.get('comp2models').split('+')
        block_name1_id = int(comp2models_request[0])
        block_name1 = block_names1[block_name1_id]
        model_name2, block_name2 = comp2models_request[1], comp2models_request[2]
        databases = [load_database(model_name1, block_name1), load_database(model_name2, block_name2)]
        trajdatabases = [load_trajdatabase(model_name1), load_trajdatabase(model_name2)]
        iteration = block_to_iteration(model_name1, block_name1)
        imgs_desc = 'Model Comparison Iteration {}'.format(iteration)
        num_comparisons = 2
        palette = cycle(Category10[max(3,num_comparisons)][:num_comparisons])
        fig_tuple1 = (make_ba_bds_fig(databases, palette), 'mpl')
        fig_tuple2 = (make_avg_ba_traj_fig(trajdatabases, palette), 'mpl')
        fig_tuple3 = (make_test_pp_trajs_fig(trajdatabases, palette),'mpl')
        fig_tuple4 = (make_probe_sim_comp_fig(databases, palette), 'mpl')
        custom_probes_tuples = load_custom_probes_tuples()
        custom_probes = [tuple[0] for tuple in custom_probes_tuples if tuple[1] == 'customn']
        fig_tuple5 = (make_neighbors_rbo_fig(databases, custom_probes), 'bokeh')
        fig_tuple6 = (make_cat_conf_diff_fig(databases), 'mpl')
        for trajdatabase in trajdatabases: trajdatabase.trajstore.close()
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3, fig_tuple4, fig_tuple5, fig_tuple6)
    ##########################################################################
    elif request.args.get('compprobes') == 'traj':
        imgs_desc = 'Probe Comparison'
        trajdatabase = load_trajdatabase(model_name1)
        custom_probes_tuples = load_custom_probes_tuples()
        comp_probe_tuples = [tuple for tuple in custom_probes_tuples
                             if not tuple[1] in ['custompdh','customn','freqhist','customclust']]
        fig_tuple1 = (make_compprobes_fig(trajdatabase, comp_probe_tuples), 'mpl')
        fig_tuple2 = (make_comp_binned_freqs_fig(trajdatabase, comp_probe_tuples), 'mpl')
        trajdatabase.trajstore.close()
        imgs = make_imgs(fig_tuple1, fig_tuple2)
    ##########################################################################
    elif request.args.get('customn') not in ['traj', None]:
        block_name1 = request.args.get('customn')
        imgs_desc = 'Nearest Neighbors Block {}'.format(block_name1)
        database = load_database(model_name1, block_name1)
        custom_probes_tuples = load_custom_probes_tuples()
        customn_probes = [tuple[0] for tuple in custom_probes_tuples if tuple[1] == 'customn']
        fig_tuple1 = (make_custom_neighbors_table_fig(database, customn_probes), 'mpl')
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    elif request.args.get('freqhist') == 'traj':
        trajdatabase = load_trajdatabase(model_name1)
        last_block_name = block_names1[-1]
        database = load_database(model_name1, last_block_name)
        imgs_desc = 'Token & Corpus Data'
        custom_probes_tuples = load_custom_probes_tuples()
        freqhist_probes = [tuple[0] for tuple in custom_probes_tuples if tuple[1] == 'freqhist']
        fig_tuple1 = (make_probe_freq_hist_fig(trajdatabase, freqhist_probes), 'mpl')
        fig_tuple2 = (make_cat_count_pie_chart_fig(database), 'mpl')
        # fig_tuple3 = (make_corpus_traj_fig(trajdatabase), 'bokeh')
        trajdatabase.trajstore.close()
        imgs = make_imgs(fig_tuple1, fig_tuple2) #, fig_tuple3)
    ##########################################################################
    elif request.args.get('customclust') not in ['traj', None]:
        block_name1 = request.args.get('customclust')
        imgs_desc = 'Custom Clustering'
        database = load_database(model_name1, block_name1)
        custom_probes_tuples = load_custom_probes_tuples()
        custom_cats = [tuple[0] for tuple in custom_probes_tuples if tuple[1] == 'customclust']
        custom_cats = map(lambda x: x.upper(), custom_cats)
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
