import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim import layers
from tensorflow.contrib.slim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS


def create_model(inputs, config, scope="letNet  ", is_training=True):

    weight_decay = 0.0005

    batch_norm_params = {
        "is_training": is_training,
        # Decay for the moving averages
        "decay": 0.9997,
        # epsilon to prevent 0s in variance.
        "epsilon": 0.001,
    }

    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'

        with arg_scope([layers.conv2d],
                       trainable=is_training,
                       activation_fn=tf.nn.relu,
                       normalizer_params=batch_norm_params,
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       normalizer_fn=layers.batch_norm,
                       outputs_collections=end_points_collection
                       ):
            end_points = OrderedDict()
            end_points['input'] = inputs
            end_points['conv1'] = layers.conv2d(inputs, 32, [5, 5], scope='conv1')
            end_points['pool1'] = layers.max_pool2d(end_points['conv1'], [2, 2], scope='pool1')
            end_points['conv2'] = layers.conv2d(end_points['pool1'], 32, [5, 5], scope='conv2')
            end_points['pool2'] = layers.max_pool2d(end_points['conv2'], [2, 2], scope='pool2')

            flattened = layers.flatten(end_points['pool2'])
            end_points['fc3'] = layers.fully_connected(flattened, 512, scope='fc3')
            end_points['dropout3'] = layers.dropout(end_points['fc3'], is_training=is_training, scope='dropout3')

            logits = end_points['fc4'] = layers.fully_connected(end_points['dropout3'], 10, activation_fn=None, scope='fc4')

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
