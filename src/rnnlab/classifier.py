import tensorflow as tf
import numpy as np
import math, os
from sklearn.model_selection import train_test_split



def calc_hca(x_data=None, y_data=None, num_h1=50, num_h2=30, num_epochs=50, mb_size=64, lr=0.001):
    ##########################################################################
    # load data
    if x_data:
        X = x_data
        Y = y_data
        input_dim = X.shape[1]
        num_classes = len(list(set(Y)))
    else:
        X = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data', 'toyxy', 'toy_X.csv'), delimiter=',')
        X = np.repeat(X, 100, axis=0)
        Y = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data', 'toyxy', 'toy_Y.csv'), delimiter=',')
        Y = np.repeat(Y, 100, axis=0)
        input_dim = X.shape[1]
        num_classes = 3
    ##########################################################################
    # shuffle datasets in unison
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    X_all_shuffled = X[p]
    Y_all_shuffled = Y[p]
    ##########################################################################
    # train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X_all_shuffled, Y_all_shuffled, test_size=0.1)
    num_train_examples, num_test_examples = len(X_train), len(X_test)
    num_test_examples = len(X_test)
    print 'Num train data: {} | Num test data: {}'.format(num_train_examples, num_test_examples)
    ##########################################################################
    # random batch fetching function
    def get_random_batch(x_data, y_data, batch_size, num_examples):
        assert len(x_data) == len(y_data)
        num_batches = num_examples // batch_size
        random_batch_id = np.random.random_integers(num_batches)
        x_batch = x_data[random_batch_id: random_batch_id + batch_size]
        y_batch = y_data[random_batch_id: random_batch_id + batch_size]
        return x_batch, y_batch
    ##########################################################################
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # placeholders
            x = tf.placeholder(tf.float32, shape=(None, input_dim))
            y = tf.placeholder(tf.int32, shape=(None))
            # hidden 1 ops
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.truncated_normal([input_dim, num_h1],
                                                          stddev=1.0 / math.sqrt(input_dim)), name='weights')
                biases = tf.Variable(tf.zeros([num_h1]))
                hidden_1 = tf.nn.relu(tf.matmul(x, weights) + biases)
            # hidden 2 ops
            with tf.name_scope('hidden2'):
                weights = tf.Variable(tf.truncated_normal([num_h1, num_h2],
                                                          stddev=1.0 / math.sqrt(num_h1)), name='weights')
                biases = tf.Variable(tf.zeros([num_h2]))
                hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights) + biases)
            # logits ops
            with tf.name_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([num_h2, num_classes],
                                                          stddev=1.0 / math.sqrt(num_h2)), name='weights')
                biases = tf.Variable(tf.zeros([num_classes]))
                logits = tf.nn.relu(tf.matmul(hidden_2, weights) + biases)
            # loss ops
            labels = tf.to_int64(y)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            # training ops
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss)
            # evaluation ops
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct_sum_ = tf.reduce_sum(tf.cast(correct, tf.int32))
    ##########################################################################
    # session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    ##########################################################################
    # test precision before training
    correct_sum = sess.run(correct_sum_, feed_dict={x: X_test, y: Y_test})
    precision = correct_sum / float(num_test_examples) *100
    print 'Before Training Test Set Num Correct: {:>10}/{:>10}   Precision {}%'.format(correct_sum, num_test_examples, int(precision))
    ##########################################################################
    max_steps = num_train_examples * num_epochs
    train_hca_traj, test_hca_traj = [], []
    for step in xrange(max_steps):
        ##########################################################################
        # training
        X_train_batch, Y_train_batch = get_random_batch(X_train, Y_train, mb_size, num_train_examples)
        _, loss_value = sess.run([train_op, loss], feed_dict={x: X_train_batch, y: Y_train_batch})
        ##########################################################################
        # evaluate after every epoch
        if step % (max_steps/num_epochs) == 0:
            # training data
            correct_sum = sess.run(correct_sum_, feed_dict={x: X_train, y: Y_train})
            train_hca = correct_sum / float(num_train_examples) * 100
            train_hca_traj.append(train_hca)
            # test data
            correct_sum = sess.run(correct_sum_, feed_dict={x: X_test, y: Y_test})
            test_hca = correct_sum / float(num_test_examples) *100
            test_hca_traj.append(test_hca)
            print 'Test Num Correct: {:>10}/{:>10}   Precision {}%'.format(
                correct_sum, num_test_examples, int(test_hca))
    ##########################################################################
    return train_hca_traj, test_hca_traj




