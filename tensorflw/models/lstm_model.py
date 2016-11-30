import tensorflw as tf
import numpy as np

import tensorflw.contrib.slim as slim
from tensorflw.contrib.slim import arg_scope
from tensorflw.contrib.slim import layers
from tensorflw.contrib.slim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

def BiLSTM(x, config):

    # x = shape (batch_size, num_time_steps, features)

    num_hidden = 256
    num_classes = config["num_classes"]

    num_time_steps = int(x._shape[1])  # 35 see Conv7 layer
    num_input = int(x._shape[2])                    # 512 See Conv7 layer

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

    combined_final_states = tf.concat(1, [output_state_fw.c, output_state_bw.c])
    logits = layers.fully_connected(combined_final_states, num_classes, activation_fn=None)

    return logits


def create_model(inputs, config, is_training=True):

        # Bidirectional LSTM
        flattened_input = tf.squeeze(inputs)
        logits = BiLSTM(flattened_input, config)

        return logits, {}


def loss(logits, labels):

    # Reshape the labels into a dense Tensor of shape [FLAGS.batch_size, num_classes].
    num_classes = logits.get_shape()[-1]
    one_hot_labels = layers.one_hot_encoding(labels, num_classes)

    # Note: label smoothing regularization LSR
    # https://arxiv.org/pdf/1512.00567.pdf
    loss_ce = losses.softmax_cross_entropy(logits, one_hot_labels, label_smoothing=0.1)

    return losses.get_total_loss(add_regularization_losses=True, name="total_loss")
