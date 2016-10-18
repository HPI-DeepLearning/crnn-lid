import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim import layers
from tensorflow.contrib.slim import losses
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS


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

        # (32, 1, 73, 512) -> (32, 73*512)
        flattened = layers.flatten(end_points['conv7'])
        logits = end_points['fc8'] = layers.fully_connected(flattened, 4, activation_fn=tf.identity, scope='fc8')

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
