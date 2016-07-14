import tensorflow as tf
import numpy as np
from tensorflow.contrib.ctc import ctc_loss
from tensorflowslim.scopes import arg_scope
from tensorflowslim import ops
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

def BiLSTM(x, config):

    # x = [batch_size, num_steps, features]

    num_hidden = 256
    num_classes = config["num_classes"]

    batch_size = int(x._shape[0])
    max_length = num_steps = int(x._shape[1]) # = 35 see Conv7 layer
    num_input = int(x._shape[2]) # = 512 See Conv7 layer

    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    # weights_hidden = tf.Variable(tf.truncated_normal([num_input, 2 * num_hidden]))
    # weights_out = tf.Variable(tf.truncated_normal([2 * num_hidden, num_classes]))
    #
    # bias_hidden = tf.Variable(tf.truncated_normal([2 * num_hidden]))
    # bias_out = tf.Variable(tf.truncated_normal([num_classes]))

    weights_hidden = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(2.0 / (2 * num_hidden))))
    weights_out = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=np.sqrt(2.0 / num_hidden)))

    bias_hidden = tf.Variable(tf.truncated_normal([num_hidden]))
    bias_out = tf.Variable(tf.truncated_normal([num_classes]))

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, num_steps, num_input)
    # Required shape: 'num_steps' tensors list of shape (batch_size, num_input)

    # Permuting batch_size and num_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (num_steps * batch_size, num_input)
    x = tf.reshape(x, [-1, num_input])
    # Split to get a list of 'num_steps' tensors of shape (batch_size, num_input)
    x = tf.split(0, num_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True, forget_bias=1.0)

    # Get lstm cell output
    outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output

    fbH1rs = [tf.reshape(t, [batch_size, 2, num_hidden]) for t in outputs]
    outH1 = [tf.reduce_sum(tf.mul(t, weights_hidden), reduction_indices=1) + bias_hidden for t in fbH1rs]

    logits = [tf.matmul(t, weights_out) + bias_out for t in outH1]

    # x_fc1 = tf.reshape(outputs, [-1, 2 * num_hidden])
    # logits = tf.matmul(x_fc1, weights_out) + bias_out
    #
    # #  Reshaping to share weights accross timesteps
    # logits = tf.reshape(logits, [-1, max_length, num_classes])
    # logits = tf.transpose(logits, (1, 0, 2))

    return logits


def create_model(inputs, config):

    batch_norm_params = {
        # Decay for the moving averages
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    end_points = OrderedDict()

    with arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.0005, ):
        with arg_scope([ops.conv2d], stddev=0.1, activation=tf.nn.relu, batch_norm_params=batch_norm_params):
            end_points['conv1'] = ops.conv2d(inputs, 64, [3, 3], scope='conv1')
            end_points['pool1'] = ops.max_pool(end_points['conv1'], [2, 2], scope='pool1')
            end_points['conv2'] = ops.conv2d(end_points['pool1'], 128, [3, 3], scope='conv2')
            end_points['pool2'] = ops.max_pool(end_points['conv2'], [2, 2], scope='pool2')
            end_points['conv3'] = ops.conv2d(end_points['pool2'], 256, [3, 3], scope='conv3')
            end_points['conv4'] = ops.conv2d(end_points['conv3'], 25, [3, 3], scope='conv4')
            end_points['pool4'] = ops.max_pool(end_points['conv4'], [1, 2], scope='pool4') # TODO Correct kernel?
            end_points['conv5'] = ops.conv2d(end_points['pool4'], 512, [3, 3], scope='conv5')
            end_points['batch_norm5'] = ops.batch_norm(end_points['conv5'], scope='batch_norm5')
            end_points['conv6'] = ops.conv2d(end_points['batch_norm5'], 512, [3, 3], padding='VALID', scope='conv6')
            end_points['batch_norm6'] = ops.batch_norm(end_points['conv6'], scope='batch_norm6')
            end_points['pool6'] = ops.max_pool(end_points['batch_norm6'], [1, 2], scope='pool6') # TODO Correct kernel?
            end_points['conv7'] = ops.conv2d(end_points['pool6'], 512, [2, 2], padding='VALID', scope='conv7')

            #flatten = ops.flatten(end_points['conv7'], scope='flatten')
            # (32, 1, 35, 512) -> (32, 35, 512)
            map_to_sequence = tf.squeeze(end_points['conv7'])

            assert len(map_to_sequence._shape) == 3

            # Bidirectional LSTM
            logits = BiLSTM(map_to_sequence, config)

            for key, endpoint in end_points.iteritems():
                print "{0}: {1}".format(key, endpoint._shape)

            return logits, end_points


def inference(images, config):
    logits, endpoints = create_model(images, config)

    # Add summaries for viewing model statistics on TensorBoard.
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            tensor_name = act.op.name
            tf.histogram_summary(tensor_name + '/activations', act)
            tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(act))

    return logits


def loss(logits, labels,  sess, batch_size=None):
    # Adds all losses for the model.

    _logits = tf.Print(logits, [logits], message="Logits", summarize=10000)
    _labels = tf.Print(labels, [labels], message="Labels", summarize=10000)
    # Create a label for every sequence based on the true label of the whole file
    sequence_length = 35
    sequence_labels = tf.mul(np.ones([batch_size, sequence_length], dtype=np.int32), tf.reshape(_labels, [batch_size, 1]))
    sequence_labels = tf.reshape(sequence_labels, [batch_size * sequence_length])
    _seq_labels = tf.Print(sequence_labels, [sequence_labels], message="Seq Labels", summarize=10000)

    # Since technically our array is not sparse, create an index for every single entry [batch_size * sequence_length, 2]
    # [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], ...
    indices = tf.constant([[j, i] for j in range(batch_size) for i in range(sequence_length)], tf.int64)
    _idx = tf.Print(indices, [indices], message="Indicies", summarize=10000)

    # Cross entropy loss for the main softmax prediction.
    sparse_labels = tf.SparseTensor(_idx, _seq_labels, tf.constant([batch_size, sequence_length], tf.int64))
    #tf.Print(sparse_labels, [sparse_labels], message="sparse labels")

    logits3d = tf.pack(_logits)

    loss = ctc_loss(logits3d, sparse_labels, batch_size * [sequence_length])

    return tf.reduce_mean(loss)
