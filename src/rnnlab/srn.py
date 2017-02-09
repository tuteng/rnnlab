

import tensorflow as tf
from dbutils import load_rnnlabrc


class SRN(object):
    '''
    Simple recurrent neural network graph in tensorflow
    '''

    def __init__(self, num_input_units, configs_dict):
        ##########################################################################
        # unpack configs
        mb_size = int(configs_dict['mb_size'])
        bptt_steps = int(configs_dict['bptt_steps'])
        num_hidden_units = int(configs_dict['num_hidden_units'])
        learning_rate = float(configs_dict['learning_rate'])
        weight_init = configs_dict['weight_init']
        act_function = configs_dict['act_function']
        optimizer = configs_dict['optimizer']
        ##########################################################################
        device = '/gpu:0' if (load_rnnlabrc('gpu')) == 'True' else '/cpu:0'
        ##########################################################################
        # weights
        with tf.device('/cpu:0'): # always needs to be cpu
            Wx = tf.get_variable('Wx', [num_input_units, num_hidden_units],
                                 initializer=self.weight_initializer(weight_init, num_input_units))
            Wy = tf.get_variable('Wy', [num_hidden_units, num_input_units],
                                 initializer=self.weight_initializer(weight_init, num_hidden_units))
            by = tf.get_variable('by', [num_input_units],
                                 initializer=tf.constant_initializer(0.0))
        ##########################################################################
        # define training step
        self.sess = tf.InteractiveSession()
        with tf.device(device):
            ##########################################################################
            # placeholders
            self.x = tf.placeholder(tf.int32, [None, None], name='input_placeholder')  # [mb_size, bptt_steps]
            self.y = tf.placeholder(tf.int32, [None], name='labels_placeholder')  # [mb_size]
            ########################################################################
            # project x to hidden layer by indexing Wx
            x_projected_to_hidden = tf.nn.embedding_lookup(Wx, self.x)
            ########################################################################
            # rnn cell definition
            act = None
            if act_function == 'sigmoid': act = tf.sigmoid
            elif act_function == 'tanh': act = tf.tanh
            cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden_units, activation=act)
            ########################################################################
            # unfold rnn cell in time (outputs and final state should be the same for basicrnn)
            (outputs, final_state) = tf.nn.dynamic_rnn(cell, x_projected_to_hidden, dtype=tf.float32)
            ########################################################################
            # hidden state tensors
            self.last_hidden_state = final_state
            self.all_hidden_states = tf.reshape(outputs, [-1, num_hidden_units])
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



    def weight_initializer(self, weight_init, dim1):
        ########################################################################
        if weight_init == 'random_normal':
            return tf.random_normal_initializer(0.0, 1 / dim1)
        elif weight_init == 'truncated_normal':
            return tf.truncated_normal_initializer(0.0, 1 / dim1)
        elif weight_init == 'uus':
            return tf.uniform_unit_scaling_initializer(factor=1.0)