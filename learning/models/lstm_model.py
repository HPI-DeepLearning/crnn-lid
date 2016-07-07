import tensorflow as tf
from tensorflowslim import losses

FLAGS = tf.app.flags.FLAGS


def create_model(inputs, config):

    num_hidden = 128
    weights = tf.Variable(tf.random_normal([num_hidden, config["num_classes"]]))
    biases = tf.Variable(tf.random_normal([config["num_classes"]]))

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_input, n_steps, channels)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(inputs, [0, 2, 1, 3])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, config["image_width"] * config["image_depth"]])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, config["image_height"], x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs[-1], weights) + biases
    return logits


def inference(images, config):
    logits, endpoints = create_model(images, config)

    # Add summaries for viewing model statistics on TensorBoard.
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            tensor_name = act.op.name
            tf.histogram_summary(tensor_name + '/activations', act)
            tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(act))

    return logits


def loss(logits, labels, batch_size=None):
    # Adds all losses for the model.

    # Reshape the labels into a dense Tensor of shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = logits.get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    # Cross entropy loss for the main softmax prediction.
    loss = losses.cross_entropy_loss(logits, dense_labels, label_smoothing=0.1, weight=1.0)
    return loss
