import matplotlib.pyplot as plt
import pandas
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from sklearn.datasets import load_diabetes, load_boston, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = None


def train():
    np.random.seed(0)

    X, y = make_moons(n_samples=400, shuffle=True, noise=0.1, random_state=0)
    X = np.array(X)
    y = np.array(y)

    reshaped_y = np.array(y).reshape(len(y), 1)

    enc = OneHotEncoder()
    enc.fit(reshaped_y)
    encoded_y = enc.transform(reshaped_y).toarray()

    X_train, x_test, Y_train, y_test = train_test_split(X, encoded_y, test_size=0.33, random_state=42)

    n_dim = X_train.shape[1]
    n_out = Y_train.shape[1]


    learning_rate = 0.01
    training_epochs = 1000
    loss_history = np.empty(shape=[1], dtype=float)

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_dim], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, n_out], name="y-input")

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def wx(input_tensor, input_dim, output_dim):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('Wx'):
            wx_result = tf.matmul(input_tensor, weights)
        return wx_result

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = wx(input_tensor, input_dim, output_dim) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def skip_out_layer(input_tensor1, input_tensor2, output_dim, layer_name, act):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.add(input_tensor1, input_tensor2) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, n_dim, 6, 'layer1', act=tf.nn.relu)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = nn_layer(dropped, 6, 6, 'layer2', act=tf.nn.tanh)
    hidden3 = nn_layer(hidden2, 6, 6, 'layer3', act=tf.nn.tanh)

    with tf.name_scope('a_out1'):
        a_out1 = wx(hidden3, 6, n_out)
    with tf.name_scope('a_out2'):
        a_out2 = wx(x, n_dim, n_out)

    y = skip_out_layer(a_out1, a_out2, n_out, 'output_layer', act=tf.nn.tanh)

    init = tf.global_variables_initializer()

    with tf.name_scope('cross_entropy'):
        lloss = tf.losses.softmax_cross_entropy(
            onehot_labels=y_, logits=y)

    tf.summary.scalar('log_loss', lloss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            lloss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = X_train, Y_train
            k = FLAGS.dropout
        else:
            xs, ys = x_test, y_test
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, mq = sess.run([merged, lloss], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Error at step %s: %s' % (i, mq))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                loss_history = np.append(loss_history, sess.run(lloss, feed_dict=feed_dict(True)))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=6000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.08,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/dl_hw2/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/dl_hw2/logs/dl_hw2_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
