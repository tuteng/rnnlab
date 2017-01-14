import os, socket, csv, sys, datetime
from corpus import Corpus
import numpy as np
from abc import ABCMeta


class RNNHelper(object):
    """
    Abstract class containing methods to create and restore an rnn.
    Inherited by RNN class.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        ##########################################################################
        # define directories 
        self.data_dir = os.path.join('data')  # this doesn't work from pycharm, but with pip yes
        self.runs_dir = load_rc('runs_dir')
        ##########################################################################
        # assign instance variables


    def restore_rnn(self):
        ##########################################################################
        # chose model_name and load configs_dict
        model_name, block_name = self.load_model_and_block_name()
        ##########################################################################
        # get configs
        configs_dict = self.load_configs_dict(model_name)
        ##########################################################################
        # make corpus
        corpus_args = [configs_dict[key] for key in ['vocab_file_name', 'freq_cutoff', 'probes_name']]
        corpus = self.make_corpus(configs_dict['corpus_name'], corpus_args)
        ##########################################################################
        # init rnn
        rnn = self.init_rnn(configs_dict, corpus)
        ##########################################################################
        # restore rnn
        rnn.saver.restore(rnn.sess, os.path.join(
            self.runs_dir, model_name, 'Weights', 'weights_at_block_{}.ckpt'.format(block_name)))
        print 'Initialized and restored from checkpoint saved at block {}'.format(block_name)
        ##########################################################################
        return rnn


    def create_rnn(self, user_configs, flavor):
        ##########################################################################
        # get configs
        configs_dict = self.make_configs_dict(user_configs, flavor)
        ##########################################################################
        # make corpus
        corpus_args = [configs_dict[key] for key in ['vocab_file_name', 'freq_cutoff', 'probes_name']]
        corpus = self.make_corpus(configs_dict['corpus_name'], corpus_args)
        ##########################################################################
        # init rnn
        rnn = self.init_rnn(configs_dict, corpus)
        ##########################################################################
        return rnn


    def make_corpus(self, corpus_name, corpus_args):
        ##########################################################################
        corpus = Corpus(corpus_name, *corpus_args)
        ##########################################################################
        return corpus


    def load_model_and_block_name(self):
        ##########################################################################
        # inits
        import re
        mf_filter = re.compile('.+_[0-9]+-[0-9]+-[0-9]+-[0-9]+')
        model_names = sorted([f for f in os.listdir(self.runs_dir) if mf_filter.search(f) is not None])
        num_mfs = len(model_names)
        last_block_names = []
        ##########################################################################
        # load all model_names
        for model_name in model_names:
            weights_file_names = os.listdir(os.path.join(self.runs_dir, model_name, 'Weights'))
            if weights_file_names:
                last_block_names.append(max([int(file_name.split('.')[0][-4:])
                                             for file_name in weights_file_names if file_name.endswith('ckpt')]))
        ##########################################################################
        # select model_name and block_name
        model_name = model_names[int(raw_input('Chose:\n{}\n'.format(
            ('\n').join(['[{}] {} (trained to block {})'.format(
                ('0' + str(i))[-2:], model_name, last_trained_block) for i, model_name, last_trained_block in
                         zip(range(num_mfs), model_names, last_block_names)]))))]
        block_names = [int(filter(str.isdigit, file_name)) for file_name in os.listdir(self.runs_dir)]
        last_block_name = block_names[int(raw_input('Chose:\n{}\n'.format(
            ('\n').join(['[{}] {}'.format(('0' + str(i))[-2:], last_trained_block) for i, last_trained_block in
                         zip(range(num_mfs), block_names)]))))]
        ##########################################################################
        return model_name, last_block_name


    def load_configs_dict(self, model_name):
        ##########################################################################
        path = os.path.join(self.runs_dir, model_name, 'Configs')
        file_name = 'configs.npy'
        configs_dict = dict(np.load(os.path.join(path, file_name)).item())
        ##########################################################################
        return configs_dict


    def make_model_name(self, flavor):
        ##########################################################################
        # make model_name (flavor in name is required for data removal to work)
        time_of_init = datetime.datetime.now().strftime('%m-%d-%H-%M')
        model_name = '{}_{}_{}'.format(socket.gethostname(), time_of_init, flavor)
        ##########################################################################
        # exit if model name dir already exists
        if os.path.isdir(os.path.join(self.runs_dir, model_name)):
            sys.exit('rnnlab : Model name already exists. This is typically caused by starting two models '
                     'within the same minute. Please try again.')
        ##########################################################################
        return model_name


    def make_configs_dict(self, user_configs, flavor):
        ##########################################################################
        # default configs
        configs_dict = {
            'bptt_steps': 7,
            'num_hidden_units': 512,
            'mb_size': 64,
            'learning_rate': 0.001,
            'weight_init': 'uus',
            'act_function': 'sigmoid',
            'bias': 1,
            'leakage': 0.95,
            'num_iterations': 20,
            'num_epochs': 1,
            'randomize_blocks': 0,
            'save_ev': 10,
            'model_name': self.make_model_name(flavor),
            'flavor': flavor,
            'corpus_name': None,
            'vocab_file_name': None,
            'freq_cutoff': None,
            'probes_name': None}
        ##########################################################################
        # overwrite default config dict
        overwritten_list = []
        print 'Overwriting default configs with user configs:'
        for user_config in user_configs:
            config_name = user_config[0]
            config_value = user_config[1]
            config_value = False if config_value == '0' else config_value
            if not config_value == 'default':
                if config_name in configs_dict:
                    print config_name, '{} -> {}'.format(configs_dict[config_name], config_value)
                    configs_dict[config_name] = config_value
                    overwritten_list.append(config_name)
        ##########################################################################
        # check configs
        for c in ['corpus_name']:
            if c not in overwritten_list: sys.exit('rnnlab WARNING: Did not find "{}" in user configs'.format(c))
        if 'freq_cutoff' not in overwritten_list and 'vocab_file_name' not in overwritten_list:
            sys.exit('rnnlab WARNING: Did not find "freq_cutoff" or "vocab_file_name" in user configs')
        if 'probes_name' not in overwritten_list: print 'rnnlab WARNING: "Did not find probes_name" in user_configs'
        ##########################################################################
        return configs_dict


    def init_rnn(self, configs_dict, corpus):
        ##########################################################################
        flavor = configs_dict['flavor']
        ##########################################################################
        # init model
        if flavor == 'lstm':
            from lstm import LSTM
            rnn = LSTM(configs_dict, corpus)
        elif flavor == 'irnn':
            from irnn import IRNN
            rnn = IRNN(configs_dict, corpus)
        elif flavor == 'srn':
            from srn import SRN
            rnn = SRN(configs_dict, corpus)
        elif flavor == 'scrn':
            from scrn import SCRN
            rnn = SCRN(configs_dict, corpus)
        else:
            sys.exit('RNN flavor not recognized.')
        ##########################################################################
        return rnn


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
    if len(set(rows)) != len(rows): sys.exit('rnnlab: Duplicate configs detected in {}'.format(user_configs_path))
    ##########################################################################
    # warn user
    if len(rows) > 1 : print 'WARNING: rnnlab does not remember which user_configurations may have been used ' \
                            'for training in the past. All configurations will be used.'
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




def get_childes_data():
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