from rnnlab import load_rnnlabrc
import tensorflow as tf
import numpy as np
import math, time, os, socket


####################################################################################################
def calc_hca(model_name, x_data, y_data, hc_epochs):
    ####################################################################################################
    # datasets params
    VALIDATION_PERC_OF_TOTAL = 10
    TEST_PERC_OF_TOTAL = 10
    USE_TOY_DATA = False
    # graph params
    HIDDEN_1_DIM = 50
    HIDDEN_2_DIM = 30
    # training params
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    USE_TENSORBOARD = False
    ####################################################################################################
    runs_dir = load_rnnlabrc('runs_dir')
    ####################################################################################################
    # make run_name for tensorboard
    run_num = 1
    while True:
        run_name = '{}_block_run_{}'.format(model_name, run_num)
        if os.path.isdir(os.path.join(runs_dir, model_name, 'Classifier', run_name)):
            run_num += 1
        else:
            break
    ####################################################################################################
    # load datasets into X and Y
    if not USE_TOY_DATA:
        X = x_data
        Y = y_data
        input_dim = X.shape[1]
        num_classes = np.max(Y) # TODO does this work?
    ####################################################################################################
    # load toy datasets into X and Y, if specified
    else:
        X = np.genfromtxt(os.path.join(rnn_dir, 'toy_X.csv'), delimiter=',')
        X = np.repeat(X, 100, axis=0)
        Y = np.genfromtxt(os.path.join(rnn_dir, 'toy_Y.csv'), delimiter=',')
        Y = np.repeat(Y, 100, axis=0)
        input_dim = X.shape[1]
        num_classes = 3
    ####################################################################################################
    # shuffle datasets in unison
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    X_all_shuffled = X[p]
    Y_all_shuffled = Y[p]
    ####################################################################################################
    # assign portions of datasets to train, val, and test sets
    num_examples = len(X)
    train_end = int(num_examples * (1 - (VALIDATION_PERC_OF_TOTAL + TEST_PERC_OF_TOTAL) /100.0))
    val_end = int(train_end + num_examples * VALIDATION_PERC_OF_TOTAL /100.0)
    X_train = X_all_shuffled[0 : train_end]
    Y_train = Y_all_shuffled[0 : train_end]
    num_train_examples = len(X_train)
    X_valid = X_all_shuffled[train_end : val_end]
    Y_valid = Y_all_shuffled[train_end : val_end]
    num_val_examples = len(X_valid)
    X_test = X_all_shuffled[val_end : ]
    Y_test = Y_all_shuffled[val_end : ]
    num_test_examples = len(X_test)
    print 'Examples in train datasets: {} Examples in val datasets: {} Examples in test datasets: {}'.\
        format(num_train_examples, num_val_examples, num_test_examples)
    ####################################################################################################
    # random batch fetching function
    def get_random_batch(x_data, y_data, batch_size, num_examples):
        assert len(x_data) == len(y_data)
        num_batches = num_examples // batch_size
        random_batch_id = np.random.random_integers(num_batches)
        x_batch = x_data[random_batch_id: random_batch_id + batch_size]
        y_batch = y_data[random_batch_id: random_batch_id + batch_size]
        return x_batch, y_batch
    ####################################################################################################
    # GRAPH START
    ####################################################################################################
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):

            # placeholders
            x = tf.placeholder(tf.float32, shape=(None, input_dim)) # to make it be bale to hold either batch or full datasets set
            y = tf.placeholder(tf.int32, shape=(None))

            # hidden 1 ops
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.truncated_normal([input_dim, HIDDEN_1_DIM], stddev=1.0 / math.sqrt(input_dim)), name='weights')
                biases = tf.Variable(tf.zeros([HIDDEN_1_DIM]))
                hidden_1 = tf.nn.relu(tf.matmul(x, weights) + biases)

            # hidden 2 ops
            with tf.name_scope('hidden2'):
                weights = tf.Variable(tf.truncated_normal([HIDDEN_1_DIM, HIDDEN_2_DIM], stddev=1.0 / math.sqrt(HIDDEN_1_DIM)), name='weights')
                biases = tf.Variable(tf.zeros([HIDDEN_2_DIM]))
                hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights) + biases)

            # logits ops
            with tf.name_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([HIDDEN_2_DIM, num_classes], stddev=1.0 / math.sqrt(HIDDEN_2_DIM)), name='weights')
                biases = tf.Variable(tf.zeros([num_classes]))
                logits = tf.nn.relu(tf.matmul(hidden_2, weights) + biases)

            # loss ops
            labels = tf.to_int64(y)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            tf.scalar_summary(loss.op.name, loss)

            # training ops
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            global_step = tf.Variable(0, name='global_step', trainable=False) # for tracking steps
            train_op = optimizer.minimize(loss, global_step=global_step)

            # evaluation ops
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))

            # summary, saver, initializer, session ops
            init = tf.initialize_all_variables()
            saver = tf.train.Saver()
            sess = tf.Session()
            if USE_TENSORBOARD:
                summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(os.path.join(sum_dir, 'classifier', run_name), sess.graph)
    ####################################################################################################
    # GRAPH END
    ####################################################################################################
    # run net
    sess.run(init)
    start_time = time.time()

    # test precision before training
    true_count = 0
    true_count += sess.run(correct_sum, feed_dict={x: X_test, y: Y_test})
    precision = true_count / float(num_test_examples) *100
    print 'Before Training Test Set Num Correct: {:>10}/{:>10}   Precision {}%'.format(true_count, num_test_examples, int(precision))
    print '--------------------------------------'




    max_steps = num_train_examples * hc_epochs
    for step in xrange(max_steps):

        # get random training batch
        X_train_batch, Y_train_batch = get_random_batch(X_train, Y_train, BATCH_SIZE, num_train_examples)

        # training
        _, loss_value = sess.run([train_op, loss], feed_dict={x: X_train_batch, y: Y_train_batch})
        elapsed = (time.time() - start_time)/60.

        # evaluate against complete train set, print to console and tensorboard
        if step % (max_steps/hc_epochs) == 0:
            true_count = 0
            true_count += sess.run(correct_sum, feed_dict={x: X_train, y: Y_train})
            train_hca = true_count / float(num_train_examples) * 100
            print 'Step {:>10}/{:>10} Loss of Last Batch: {:<20} Elapsed: {:<5} mins Run {}'.format(step, max_steps, loss_value, int(elapsed), run_num, int(precision))
            print '    Training   Set Num Correct: {:>10}/{:>10}   Precision {}%'.format(true_count, num_train_examples, int(train_hca))
            if USE_TENSORBOARD:
                summary_str = sess.run(summary_op, feed_dict={x: X_train_batch, y: Y_train_batch})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


        # evaluate against complete validation set
        if step % (max_steps/hc_epochs) == 0:
            true_count = 0
            true_count += sess.run(correct_sum, feed_dict={x: X_valid, y: Y_valid})
            val_precision = true_count / float(num_val_examples) *100
            print '    Validation Set Num Correct: {:>10}/{:>10}   Precision {}%'.format(true_count, num_val_examples, int(val_precision))



        # evaluate against complete test set
        if step % (max_steps/hc_epochs) == 0:
            true_count = 0
            true_count += sess.run(correct_sum, feed_dict={x: X_test, y: Y_test})
            test_hca = true_count / float(num_test_examples) *100
            print '    Test       Set Num Correct: {:>10}/{:>10}   Precision {}%'.format(true_count, num_test_examples, int(test_hca))

        # save last accuracies
        if step == max_steps -1:
            hc_dict = {'train_hca' : train_hca, 'test_hca' : test_hca, 'hc_epochs' : hc_epochs, 'learning_rate' : LEARNING_RATE}
            path = os.path.join(runs_dir, model_name, 'Classifier')
            file_name = 'hc_data_block_{}.npy'
            np.save(os.path.join(path, file_name), hc_dict)



    return train_hca, test_hca




