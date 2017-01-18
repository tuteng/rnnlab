

import tensorflow as tf
from utilities import load_rc


class SRN(object):
    '''
    Simple recurrent neural network graph in tensorflow
    '''

    def __init__(self, configs_dict, corpus): # TODO i don't really need ot make this a class, and i don't need 'self'
        ##########################################################################
        # assign instance variables from configs_dict
        self.bptt_steps = int(configs_dict['bptt_steps'])
        self.num_hidden_units = int(configs_dict['num_hidden_units'])
        self.mb_size = int(configs_dict['mb_size'])
        self.learning_rate = float(configs_dict['learning_rate'])
        self.weight_init = str(configs_dict['weight_init'])
        self.act_function = configs_dict['act_function']
        self.bias = int(configs_dict['bias'])
        self.optimizer = int(configs_dict['optimizer'])
        self.leakage, configs_dict['leakage'] = 0, 0 # srn specific
        self.num_iterations = int(configs_dict['num_iterations'])
        self.num_epochs = int(configs_dict['num_epochs'])
        self.randomize_blocks = int(configs_dict['randomize_blocks'])
        self.save_ev = int(configs_dict['save_ev'])
        self.num_input_units = len(corpus.token_list)
        ##########################################################################
        # add dict and corpus
        self.configs_dict = configs_dict
        self.corpus = corpus
        ##########################################################################
        device = '/gpu:0' if (load_rc('gpu')) == 'True' else '/cpu:0'
        ##########################################################################
        # weights
        def weight_initializer(self, dim1):
            if self.weight_init == 'random_normal':
                return tf.random_normal_initializer(0.0, 1/dim1)
            elif self.weight_init == 'truncated_normal':
                return tf.truncated_normal_initializer(0.0, 1/dim1)
            elif self.weight_init == 'uus':
                return tf.uniform_unit_scaling_initializer(factor=1.0)
        def bias_initializer(self):
            if not self.bias: return tf.constant_initializer(0.0)
            else: return tf.constant_initializer(0.0)
        with tf.device('/cpu:0'): # always needs to be cpu
            self.Wx = tf.get_variable('Wx', [self.num_input_units, self.num_hidden_units], initializer=weight_initializer(self, self.num_input_units))
            self.Wy = tf.get_variable('Wy', [self.num_hidden_units, self.num_input_units], initializer=weight_initializer(self, self.num_hidden_units))
            self.by = tf.get_variable('by', [self.num_input_units], initializer=bias_initializer(self), trainable=self.bias)
        ##########################################################################
        # define training step
        self.sess = tf.InteractiveSession()
        with tf.device(device):
            ##########################################################################
            # placeholders
            self.x = tf.placeholder(tf.int32, [self.mb_size, None], name='input_placeholder')
            self.y = tf.placeholder(tf.int32, [None], name='labels_placeholder')
            ########################################################################
            # project x to hidden layer by indexing Wx
            x_projected_to_hidden = tf.nn.embedding_lookup(self.Wx, self.x)
            ########################################################################
            # rnn cell definition
            act = None
            if self.act_function == 'sigmoid': act = tf.sigmoid
            elif self.act_function == 'tanh': act = tf.tanh
            cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden_units, activation=act)
            ########################################################################
            # unfold rnn cell in time
            (hidden_states_tensor, last_hidden_state_tensor) = tf.nn.dynamic_rnn(cell, x_projected_to_hidden, dtype=tf.float32)
            ########################################################################
            # hidden state tensors
            self.last_hidden_state = last_hidden_state_tensor
            self.all_hidden_states = tf.reshape(hidden_states_tensor, [-1, self.num_hidden_units])
            ########################################################################
            # output layer
            last_logit = tf.matmul(self.last_hidden_state, self.Wy) + self.by
            self.softmax_probs = tf.nn.softmax(last_logit)
            ########################################################################
            # cost
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(last_logit, self.y)
            self.total_loss = tf.reduce_mean(losses)
            # perplexity
            self.pp_mat = tf.exp(losses)
            self.mean_pp = tf.exp(self.total_loss)
            ########################################################################
            # training step definition
            if self.optimizer == 'adagrad':
                print 'Using Adagrad optmizer'
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)
            else:
                print 'Using SGD optmizer'
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
        ########################################################################
        # saver
        self.saver = tf.train.Saver(max_to_keep=10)
        ########################################################################
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        print 'Compiled tensorflow graph and initialized all variables'
