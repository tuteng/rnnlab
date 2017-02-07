

import tensorflow as tf
from dbutils import load_rnnlabrc
import numpy as np # irnn specific

class IRNN(object):
    '''
    Identity recurrent neural network graph in tensorflow
    '''

    def __init__(self, num_input_units, configs_dict):
        ##########################################################################
        # assign instance variables that will be called outside of class definition
        self.mb_size = int(configs_dict['mb_size'])
        self.bptt_steps = int(configs_dict['bptt_steps'])
        self.num_hidden_units = int(configs_dict['num_hidden_units'])
        ##########################################################################
        # unpack remaining configs
        learning_rate = float(configs_dict['learning_rate'])
        weight_init = configs_dict['weight_init']
        act_function = configs_dict['act_function']
        bias = int(configs_dict['bias'])
        optimizer = configs_dict['optimizer']
        ##########################################################################
        device = '/gpu:0' if (load_rnnlabrc('gpu')) == 'True' else '/cpu:0'
        ##########################################################################
        # weights
        def weight_initializer(dim1):
            if weight_init == 'random_normal':
                return tf.random_normal_initializer(0.0, 1/dim1)
            elif weight_init == 'truncated_normal':
                return tf.truncated_normal_initializer(0.0, 1/dim1)
            elif weight_init == 'uus':
                return tf.uniform_unit_scaling_initializer(factor=1.0)
        def bias_initializer():
            if not bias: return tf.constant_initializer(0.0)
            else: return tf.constant_initializer(0.0)
        with tf.device(device):
            Wy = tf.get_variable('Wy', [self.num_hidden_units, num_input_units], initializer=weight_initializer(self.num_hidden_units))
            by = tf.get_variable('by', [num_input_units], initializer=bias_initializer(), trainable=bias)
        ##########################################################################
        # define training step
        self.sess = tf.InteractiveSession()
        with tf.device(device):
            ##########################################################################
            # placeholders
            self.x = tf.placeholder(tf.int32, [self.mb_size, self.bptt_steps], name='input_placeholder') # can set bppt to none
            self.y = tf.placeholder(tf.int32, [self.mb_size], name='labels_placeholder')
            ########################################################################
            # convert integer input to one-hot
            x_one_hot = tf.one_hot(self.x, self.num_input_units)
            rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, self.bptt_steps, x_one_hot)]
            ########################################################################
            # rnn cell definition
            def cell(rnn_input, state):
                self.Wx = tf.get_variable('Wx', [self.num_input_units, self.num_hidden_units], initializer=weight_initializer(self, self.num_input_units))
                self.Wh = tf.get_variable('Wh', [self.num_hidden_units, self.num_hidden_units], initializer=tf.constant_initializer(np.identity(self.num_hidden_units)), trainable=True)
                self.bh = tf.get_variable('bh', [self.num_hidden_units], initializer=bias_initializer(self))
                if self.leakage:
                    leakage_vector = np.zeros(self.num_hidden_units, dtype=np.float32)#np.arange(start=0, stop=1, step=1./self.num_hidden_units, dtype=np.float32)
                    # leakage_vector[:self.num_hidden_units/2] = 0
                    # leakage_vector[self.num_hidden_units/2:] = self.leakage
                    leakage_vector[:] = self.leakage
                    leakage_vector = tf.constant(leakage_vector)
                    one_minus_leakage_vector = 1-leakage_vector
                    return tf.nn.relu(tf.mul(one_minus_leakage_vector, tf.matmul(rnn_input, self.Wx)) + tf.mul(leakage_vector, tf.matmul(state, self.Wh) + self.bh))
                else:
                    return tf.nn.relu(tf.matmul(rnn_input, self.Wx) + tf.matmul(state, self.Wh) + self.bh)

            # try constraining learning the diagonal weights between 0 and 1, using sigmoid by making tau_vector trainable

            ########################################################################
            # unfold rnn cell in time
            hidden_states_list = []
            hidden_state = tf.zeros([self.mb_size, self.num_hidden_units])
            with tf.variable_scope('rnn_cell'):
                for bptt_step in range(self.bptt_steps):
                    if bptt_step > 0: tf.get_variable_scope().reuse_variables()
                    hidden_state = cell(rnn_inputs[bptt_step], hidden_state)
                    hidden_states_list.append(hidden_state)
            last_hidden_state_tensor = hidden_states_list[-1]
            hidden_states_tensor = tf.concat(1, hidden_states_list)
            ########################################################################
            # hidden state tensors
            self.last_hidden_state = last_hidden_state_tensor
            self.all_hidden_states = tf.reshape(hidden_states_tensor, [-1, num_hidden_units])
            ########################################################################
            # output layer
            last_logit = tf.matmul(self.last_hidden_state, Wy) + by
            self.softmax_probs = tf.nn.softmax(last_logit)
            ########################################################################
            # cost
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(last_logit, self.y)
            total_loss = tf.reduce_mean(losses)
            # perplexity
            self.pp_vec = tf.exp(losses)
            self.mean_pp = tf.exp(total_loss) # used too calc test docs pp
            ########################################################################
            # training step definition
            if optimizer == 'adagrad':
                print 'Using Adagrad optmizer'
                self.train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
            else:
                print 'Using SGD optmizer'
                self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
        ########################################################################
        # saver
        self.saver = tf.train.Saver(max_to_keep=10)
        ########################################################################
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        print 'Compiled tensorflow graph and initialized all variables'