import tensorflow as tf
import numpy as np
from tensorflowslim.scopes import arg_scope
from tensorflowslim import ops
from tensorflowslim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

def BiLSTM(x, config):

    # x = shape (batch_size, num_time_steps, features)

    num_hidden = 256
    num_classes = config["num_classes"]

    batch_size = int(x._shape[0])                   # 32 see config.yaml
    max_length = num_time_steps = int(x._shape[1])  # 35 see Conv7 layer
    num_input = int(x._shape[2])                    # 512 See Conv7 layer

    weights_hidden = tf.Variable(tf.truncated_normal([2, num_hidden], stddev=np.sqrt(2.0 / (2 * num_hidden))))
    weights_out = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=np.sqrt(2.0 / num_hidden)))

    bias_hidden = tf.Variable(tf.zeros([num_hidden]))
    bias_out = tf.Variable(tf.zeros([num_classes]))

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

    # Reshape output state to split forward and backward pass: [(bs, 2, num_hidden)] * num_time_steps
    fb_hidden = [tf.reshape(t, [batch_size, 2, num_hidden]) for t in outputs]

    # Combine forward and backward state by summation: [(bs, num_hidden)] * num_time_steps
    out = [tf.reduce_sum(tf.mul(t, weights_hidden), reduction_indices=1) + bias_hidden for t in fb_hidden]

    # Reduce hidden states to num_classes: [(bs, num_classes)] * num_time_steps
    logits = [tf.matmul(t, weights_out) + bias_out for t in out]

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
            end_points['dropout4'] = ops.dropout(end_points['pool4'], 0.5, scope='dropout4')
            end_points['conv5'] = ops.conv2d(end_points['dropout4'], 512, [3, 3], scope='conv5')
            end_points['batch_norm5'] = ops.batch_norm(end_points['conv5'], scope='batch_norm5')
            end_points['conv6'] = ops.conv2d(end_points['batch_norm5'], 512, [3, 3], padding='VALID', scope='conv6')
            end_points['batch_norm6'] = ops.batch_norm(end_points['conv6'], scope='batch_norm6')
            end_points['pool6'] = ops.max_pool(end_points['batch_norm6'], [1, 2], scope='pool6') # TODO Correct kernel?
            end_points['conv7'] = ops.conv2d(end_points['pool6'], 512, [2, 2], padding='VALID', scope='conv7') # (batch_size, 1, 35, 512)

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


def loss(logits, labels, batch_size):

    # TODO Consider using first state and all states
    # Use the last state of the LSTM as output
    last_state = logits[-1]


    # Reshape the labels into a dense Tensor of shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = last_state.get_shape()[-1].value
    one_hot_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)


    # Cross entropy loss for the main softmax prediction.
    loss = losses.cross_entropy_loss(last_state, one_hot_labels, label_smoothing=0.1, weight=1.0)

    return tf.reduce_mean(loss)


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