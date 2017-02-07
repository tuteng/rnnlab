import time, socket
from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from bokeh import mpl
from bokeh.models import Range1d
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import Line
from bokeh.palettes import Category10
from itertools import cycle

from utils import get_log_mtime
from utils import get_block_names_to_display
from utils import make_imgs
from utils import delete_model
from utils import load_custom_probes_tuples
from utils import load_filtered_log_entries
from utils import make_block_names2_dict
from utils import load_database
from utils import load_trajdatabase
from utils import make_neighbors_rbo_fig
from utils import make_ba_bds_fig
from utils import make_avg_token_ba_trajs_fig
from utils import make_test_pp_trajs_fig
from utils import make_probe_sim_comp_fig
from utils import block_to_iteration
from utils import make_probe_freq_hist_fig
from utils import make_cat_count_pie_chart_fig
from utils import make_corpus_traj_fig
from utils import load_configs_dict


##########################################################################
app = Flask(__name__)
##########################################################################
# defaults
btn_names_top = ['avgtrajBtn', 'dimredBtn', 'catsimBtn', 'bdBtn',
             'clustBtn', 'neighborsBtn', 'batrajBtn', 'catconfBtn']

btn_names_bottom = ['compprobes', 'customn', 'custompdh', 'freqhist', 'customclust', 'delete']

headers_to_display = ['model_name', 'block_order',
    'num_reps', 'num_iterations', 'completed', 'best_token_ba']
##########################################################################


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
    ##########################################################################
    # make avgtraj imgs
    elif request.args.get('avgtrajBtn') == 'clicked':
        trajdatabase = load_trajdatabase(model_name1)
        num_comparisons = 1
        palette = cycle(Category10[max(3, num_comparisons)][:num_comparisons])
        # fig 1
        sel_model_names = [model_name1]
        fig = make_avg_token_ba_trajs_fig(sel_model_names, palette)
        fig1 = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, save')
        fig1.y_range = Range1d(50, 80)
        fig1.plot_width = 600
        fig1.plot_height = 300
        fig_tuple1 = (fig1, 'bokeh')
        # fig 2
        fig = make_test_pp_trajs_fig(sel_model_names, palette)
        fig2 = mpl.to_bokeh(fig, tools='pan, wheel_zoom, crosshair, save')
        fig2.y_range = Range1d(0, 500)
        fig2.plot_width = 600
        fig2.plot_height = 300
        fig_tuple2 = (fig2, 'bokeh')
        # fig 3
        fig3 = trajdatabase.make_ba_pp_window_corr_fig()
        fig_tuple3 = (fig3, 'mpl')
        # format fig_tuples to html
        imgs_desc = 'Average Trajectories'
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3)
        # close
        trajdatabase.trajstore.close()
    ##########################################################################
    # make dimred imgs
    elif request.args.get('dimredBtn'):
        sel_block_name = request.args.get('dimredBtn')
        database = load_database(model_name1, sel_block_name)
        # fig 1
        fig1 = database.make_acts_2d_fig()
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Dimensionality Reduction Block {}'.format(sel_block_name)
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    # make catsim imgs
    elif request.args.get('catsimBtn'):
        sel_block_name = request.args.get('catsimBtn')
        database = load_database(model_name1, sel_block_name)
        # fig 1
        fig1 = database.make_cat_sim_dh_fig()
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Category Similarity Heatmap-Dendrogram Block {}'.format(sel_block_name)
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    # make breakdown imgs
    elif request.args.get('bdBtn'):
        sel_block_name = request.args.get('bdBtn')
        database = load_database(model_name1, sel_block_name)
        # fig 1
        fig1 = database.make_ba_breakdown_scatter_fig()
        fig_tuple1 = (fig1, 'mpl')
        # fig2
        fig2 = database.make_ba_breakdown_fig()
        fig_tuple2 = (fig2, 'mpl')
        imgs_desc = 'Balanced Accuracy Breakdown by Probe Block {}'.format(sel_block_name)
        imgs = make_imgs(fig_tuple1, fig_tuple2)
    ##########################################################################
    # make cluster imgs
    elif request.args.get('clustBtn'):
        sel_block_name = request.args.get('clustBtn')
        database = load_database(model_name1, sel_block_name)
        # fig_tuples
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append((database.make_cat_cluster_fig(cat), 'mpl'))
        imgs_desc = 'Activation States Clustering Block {}'.format(sel_block_name)
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    # make neighbors imgs
    elif request.args.get('neighborsBtn'):
        sel_block_name = request.args.get('neighborsBtn')
        database = load_database(model_name1, sel_block_name)
        # fig_tuples
        fig_tuples = []
        for cat in database.cat_list:
            fig_tuples.append(database.make_neighbors_table_fig(cat))
        imgs_desc = 'Nearest Neighbors Block {}'.format(sel_block_name)
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    # make batrajs imgs
    elif request.args.get('batrajBtn'):
        sel_block_name = request.args.get('batrajBtn')
        trajdatabase = load_trajdatabase(model_name1)
        # fig_tuples
        fig_tuples = []
        for cat in trajdatabase.cat_list:
            custom_probes = trajdatabase.cat_probe_list_dict[cat]
            fig, x, ys, palette = trajdatabase.make_token_ba_trajs_fig(custom_probes, cat)
            hover = HoverTool(
                tooltips=[('block', '@block'), ('probe', '@probe'), ('balAcc', '$y')])
            fig1 = mpl.to_bokeh(fig, tools=[hover, 'pan, wheel_zoom, crosshair, save'])
            for n, y in enumerate(ys):
                source = ColumnDataSource(
                    data=dict(block=x, balAcc=y, probe=[custom_probes[n]] * len(x)))
                line = Line(x='block', y='balAcc', line_color=next(palette), line_width=2)
                fig1.add_glyph(source, line)
            fig1.y_range = Range1d(0, 100)
            fig_tuples.append((fig1, 'bokeh'))
            fig_tuples.append((trajdatabase.make_cfreq_traj_fig(custom_probes, cat), 'mpl'))
        imgs_desc = 'Token Balanced Accuracy Trajectories Block {}'.format(sel_block_name)
        imgs = make_imgs(*fig_tuples)
        # close
        trajdatabase.trajstore.close()
    ##########################################################################
    # make catconfusion img
    elif request.args.get('catconfBtn'):
        sel_block_name = request.args.get('catconfBtn')
        database = load_database(model_name1, sel_block_name)
        # fig1
        fig1 = database.make_cat_confusion_mat_fig()
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Category Confusion Matrix Block {}'.format(sel_block_name)
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    # make dh imgs
    elif request.args.get('custompdh'):
        sel_block_name = request.args.get('custompdh')
        database = load_database(model_name1, sel_block_name)
        # fig_tuples
        act_function = load_configs_dict(model_name1)['act_function']
        vmin = -1.0 if act_function == 'tanh' else 0.0
        custom_data_tuples = load_custom_probes_tuples()
        custom_probes = [tuple[0] for tuple in custom_data_tuples if tuple[1] == 'custompdh']
        fig_tuples = []
        for custom_probe in custom_probes:
            fig_tuples.append((database.make_acts_dh_fig(custom_probe, vmin=vmin), 'mpl'))
            fig_tuples.append((database.make_token_corcoeff_hist_fig(custom_probe), 'mpl'))
        imgs_desc = 'Activations Dendrogram-Heatmap Block {}'.format(sel_block_name)
        imgs = make_imgs(*fig_tuples)
    ##########################################################################
    # make comp2models imgs
    elif request.args.get('comp2models') is not None:
        comp2models_request = request.args.get('comp2models').split('+')
        block_name1_id = int(comp2models_request[0])
        block_name1 = block_names1[block_name1_id]
        model_name2, block_name2 = comp2models_request[1], comp2models_request[2]
        sel_block_names_to_compare = [block_name1, block_name2]
        sel_model_names_to_compare = [model_name1, model_name2]
        # reuse same palette
        num_comparisons = len(sel_model_names_to_compare)
        palette = cycle(Category10[max(3,num_comparisons)][:num_comparisons])
        # fig1
        fig1 = make_ba_bds_fig(sel_model_names_to_compare, sel_block_names_to_compare, palette)
        fig_tuple1 = (fig1, 'mpl')
        # fig2
        fig2 = make_avg_token_ba_trajs_fig(sel_model_names_to_compare, palette)
        fig2 = mpl.to_bokeh(fig2, tools='pan, wheel_zoom, crosshair, save')
        fig2.y_range = Range1d(50, 100)
        fig2.plot_width = 600
        fig2.plot_height = 300
        fig_tuple2 = (fig2, 'bokeh')
        # fig3
        fig3 = make_test_pp_trajs_fig(sel_model_names_to_compare, palette)
        fig3 = mpl.to_bokeh(fig3, tools='pan, wheel_zoom, crosshair, save')
        fig3.y_range = Range1d(0, 500)
        fig3.plot_width = 600
        fig3.plot_height = 300
        fig_tuple3 = (fig3,'bokeh')
        # fig4
        fig4 = make_probe_sim_comp_fig(sel_model_names_to_compare, sel_block_names_to_compare, palette)
        fig_tuple4 = (fig4, 'mpl')
        # fig5
        start = time.time()
        custom_data_tuples = load_custom_probes_tuples()
        custom_probes = [tuple[0] for tuple in custom_data_tuples if tuple[1] == 'customn']
        fig5 = make_neighbors_rbo_fig(sel_model_names_to_compare, sel_block_names_to_compare, custom_probes)
        fig5.y_range = Range1d(0, 1)
        fig_tuple5 = (fig5, 'bokeh')
        print 'Took {} secs to make fig'.format(time.time() - start)
        iteration = block_to_iteration(model_name1, block_name1)
        imgs_desc = 'Model Comparison Iteration {}'.format(iteration)
        imgs = make_imgs(fig_tuple1, fig_tuple2, fig_tuple3, fig_tuple4, fig_tuple5)
    ##########################################################################
    # make compprobes imgs
    elif request.args.get('compprobes'):
        trajdatabase = load_trajdatabase(model_name1)
        # fig1
        custom_data_tuples = load_custom_probes_tuples()
        sel_custom_prob_tuples = [tuple for tuple in custom_data_tuples
                                  if not tuple[1] in ['custompdh','customn','freqhist']]
        fig1 = trajdatabase.make_compprobes_fig(sel_custom_prob_tuples)
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Probe Comparison'
        imgs = make_imgs(fig_tuple1)
        # close
        trajdatabase.trajstore.close()
    ##########################################################################
    # make customn imgs
    elif request.args.get('customn'):
        sel_block_name = request.args.get('customn')
        database = load_database(model_name1, sel_block_name)
        # fig
        custom_data_tuples = load_custom_probes_tuples()
        custom_probes = [tuple[0] for tuple in custom_data_tuples if tuple[1] == 'customn']
        fig1 = database.make_custom_neighbors_table_fig(custom_probes)
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Nearest Neighbors Block {}'.format(sel_block_name)
        imgs = make_imgs(fig_tuple1)
    ##########################################################################
    # make freqhist imgs
    elif request.args.get('freqhist'):
        # fig1
        custom_data_tuples = load_custom_probes_tuples()
        custom_probes = [tuple[0] for tuple in custom_data_tuples if tuple[1] == 'freqhist']
        fig1 = make_probe_freq_hist_fig(model_name1, custom_probes)
        fig_tuple1 = (fig1, 'mpl')
        # fig2
        fig2 = make_cat_count_pie_chart_fig(model_name1)
        fig_tuple2 = (fig2, 'mpl')
        # fig3
        # fig3 = make_corpus_traj_fig(model_name1)
        # fig_tuple3 = (fig3, 'bokeh')
        imgs_desc = 'Token Data'
        imgs = make_imgs(fig_tuple1, fig_tuple2) #, fig_tuple3)
    ##########################################################################
    # make customclust img
    elif request.args.get('customclust'):
        sel_block_name = request.args.get('customclust')
        database = load_database(model_name1, sel_block_name)
        # fig1
        custom_data_tuples = load_custom_probes_tuples()
        custom_cats = [tuple[0] for tuple in custom_data_tuples if tuple[1] == 'customclust']
        custom_cats = map(lambda x: x.upper(), custom_cats)
        fig1 = database.make_custom_cat_clust_fig(custom_cats)
        fig_tuple1 = (fig1, 'mpl')
        imgs_desc = 'Token Data'
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
