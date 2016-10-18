import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim import layers
from tensorflow.contrib.slim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

def BiLSTM(x, config):

    # x = shape (batch_size, num_time_steps, features)

    num_hidden = 256
    num_classes = config["num_classes"]

    batch_size = int(x._shape[0])                   # 32 see config.yaml
    max_length = num_time_steps = int(x._shape[1])  # 35 see Conv7 layer
    num_input = int(x._shape[2])                    # 512 See Conv7 layer

    # weights_hidden = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(2.0 / (2 * num_hidden))))
    # weights_out = tf.Variable(tf.truncated_normal([2 * num_hidden, num_classes], stddev=np.sqrt(2.0 / (2 * num_hidden))))

    # bias_hidden = tf.Variable(tf.zeros([num_hidden]))
    # bias_out = tf.Variable(tf.zeros([num_classes]))

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, num_time_steps, num_input)
    # Required shape: 'num_time_steps' tensors list of shape (batch_size, num_input)

    # Permuting batch_size and num_time_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (num_time_steps * batch_size, num_input)
    x = tf.reshape(x, [-1, num_input])
    # Split to get a list of 'num_time_steps' tensors of shape (batch_size, num_input)
    x = tf.split(0, num_time_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)

    # Get lstm cell output: [(bs, 2*num_hidden)] * num_time_steps
    outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, scope='BiRNN')
    #
    # # Reshape output state to split forward and backward pass: [(bs, 2, num_hidden)] * num_time_steps
    # fb_hidden = [tf.reshape(t, [batch_size, 2, num_hidden]) for t in outputs]
    #
    # # Combine forward and backward state by summation: [(bs, num_hidden)] * num_time_steps
    # out = [tf.reduce_sum(tf.mul(t, weights_hidden), reduction_indices=1) + bias_hidden for t in fb_hidden]
    #
    # # Reduce hidden states to num_classes: [(bs, num_classes)] * num_time_steps
    # logits = [tf.matmul(t, weights_out) + bias_out for t in out]


    combined_final_states = tf.concat(1, [output_state_fw.c, output_state_bw.c])
    # logits = tf.matmul(a, weights_out) + bias_out
    logits = layers.fully_connected(combined_final_states, num_classes, activation_fn=None)

    return logits


def create_model(inputs, config, is_training=True):

    weight_decay = 0.0005

    batch_norm_params = {
        "is_training": is_training,
        # Decay for the moving averages
        "decay": 0.9997,
        # epsilon to prevent 0s in variance.
        "epsilon": 0.001,
    }

    with arg_scope([layers.conv2d],
                   trainable=True,
                   activation_fn=tf.nn.relu,
                   normalizer_params=batch_norm_params,
                   weights_regularizer=slim.l2_regularizer(weight_decay),
                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                   normalizer_fn=layers.batch_norm,
    ):
        end_points = OrderedDict()
        end_points['conv1'] = layers.conv2d(inputs, 64, [3, 3], scope='conv1')
        end_points['pool1'] = layers.max_pool2d(end_points['conv1'], [2, 2], scope='pool1')
        end_points['conv2'] = layers.conv2d(end_points['pool1'], 128, [3, 3], scope='conv2')
        end_points['pool2'] = layers.max_pool2d(end_points['conv2'], [2, 2], scope='pool2')
        end_points['conv3'] = layers.conv2d(end_points['pool2'], 256, [3, 3], scope='conv3')
        end_points['conv4'] = layers.conv2d(end_points['conv3'], 25, [3, 3], scope='conv4')
        end_points['pool4'] = layers.max_pool2d(end_points['conv4'], [1, 2], scope='pool4')  # TODO Correct kernel?
        #end_points['dropout4'] = layers.dropout(end_points['pool4'], 0.5, is_training=is_training, scope='dropout4')
        end_points['conv5'] = layers.conv2d(end_points['pool4'], 512, [3, 3], scope='conv5')
        end_points['conv6'] = layers.conv2d(end_points['conv5'], 512, [3, 3], padding='VALID', scope='conv6')
        end_points['pool6'] = layers.max_pool2d(end_points['conv6'], [1, 2], scope='pool6')  # TODO Correct kernel?
        end_points['conv7'] = layers.conv2d(end_points['pool6'], 512, [2, 2], padding='VALID', scope='conv7')  # (batch_size, 1, 73, 512)

        # (32, 1, 73, 512) -> (32, 73, 512)
        map_to_sequence = tf.squeeze(end_points['conv7'])

        assert len(map_to_sequence._shape) == 3

        # Bidirectional LSTM
        logits = BiLSTM(map_to_sequence, config)

        for key, endpoint in end_points.iteritems():
            print "{0}: {1}".format(key, endpoint._shape)

        return logits, end_points


def loss(logits, labels):

    # Reshape the labels into a dense Tensor of shape [FLAGS.batch_size, num_classes].
    num_classes = logits.get_shape()[-1]
    one_hot_labels = layers.one_hot_encoding(labels, num_classes)

    # Note: label smoothing regularization LSR
    # https://arxiv.org/pdf/1512.00567.pdf
    loss = losses.softmax_cross_entropy(logits, one_hot_labels, label_smoothing=0.1)

    return loss


def ctc_loss():

    pass
    # # Create a label for every sequence based on the true label of the whole file
    # sequence_length = 35 # logits.get_shape(0)
    # sequence_label = tf.mul(np.ones([batch_size, sequence_length], dtype=np.int32), tf.reshape(labels, [batch_size, 1]))
    # # sequence_label = tf.constant(np.random.randint(4, size=(batch_size, sequence_length)), tf.int32)
    # seq_labels = tf.Print(sequence_label, [sequence_label], message="Sequence Labels", summarize=10000)
    #
    # # Since technically our array is not sparse, create an index for every single entry [batch_size * sequence_length, 2]
    # # [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], ...
    # indices = tf.constant([[j, i] for j in range(batch_size) for i in range(sequence_length)], tf.int64)
    # _idx = tf.Print(indices, [indices], message="Indicies", summarize=10000)
    #
    # # Cross entropy loss for the main softmax prediction.
    # sparse_labels = tf.SparseTensor(_idx, tf.reshape(seq_labels, [-1]), tf.constant([batch_size, sequence_length], tf.int64))
    # # tf.Print(sparse_labels, [sparse_labels], message="sparse labels")
    #
    #
    # _logits = tf.Print(logits, [logits], message="logits", summarize=10000)
    # loss = ctc_loss(_logits , sparse_labels, batch_size * [sequence_length])