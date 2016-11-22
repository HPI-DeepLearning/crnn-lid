import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim import layers
from tensorflow.contrib.slim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS
NAME = "Topcoder_CNN"

def create_model(inputs, config, is_training=True, scope="default_scope"):

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
                       trainable=True,
                       activation_fn=tf.nn.relu,
                       normalizer_params=batch_norm_params,
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       normalizer_fn=layers.batch_norm,
                       outputs_collections=end_points_collection
                       ):
            end_points = OrderedDict()
            end_points['input'] = inputs
            end_points['conv1'] = layers.conv2d(inputs, 16, [7, 7], scope='conv1')
            end_points['pool1'] = layers.max_pool2d(end_points['conv1'], [3, 3], scope='pool1')
            end_points['conv2'] = layers.conv2d(end_points['pool1'], 32, [5, 5], scope='conv2')
            end_points['pool2'] = layers.max_pool2d(end_points['conv2'], [3, 3], scope='pool2')
            end_points['conv3'] = layers.conv2d(end_points['pool2'], 64, [3, 3], scope='conv3')
            end_points['pool3'] = layers.max_pool2d(end_points['conv3'], [3, 3], scope='pool3')
            end_points['conv4'] = layers.conv2d(end_points['pool3'], 128, [3, 3], scope='conv4')
            end_points['pool4'] = layers.max_pool2d(end_points['conv4'], [3, 3], scope='pool4')
            end_points['conv5'] = layers.conv2d(end_points['pool4'], 128, [3, 3], scope='conv5')
            end_points['pool5'] = layers.max_pool2d(end_points['conv5'], [3, 3], scope='pool5')
            end_points['conv6'] = layers.conv2d(end_points['pool5'], 256, [3, 3], scope='conv6')
            end_points['pool6'] = layers.max_pool2d(end_points['conv6'], [3, 3], scope='pool6')

            flatten = layers.flatten(end_points['pool6'])
            end_points['fc7'] = layers.fully_connected(flatten, 1024, scope='fc7')
            end_points['fc8'] = layers.fully_connected(end_points['fc7'], 4, activation_fn=None, scope='fc8')

            logits = tf.squeeze(end_points['fc8'])

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
