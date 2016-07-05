import numpy as np
import tensorflow as tf
from tensorflowslim import losses
from tensorflowslim import ops
from tensorflowslim.scopes import arg_scope
from tensorflowslim import scopes
from tensorflowslim import variables

FLAGS = tf.app.flags.FLAGS


def create_model(inputs, num_classes, dropout_keep_prob=0.8):
    batch_norm_params = {
        # Decay for the moving averages
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    end_points = {}

    with arg_scope([ops.conv2d, ops.fc], stddev=0.01, weight_decay=0.0005, ):
        with arg_scope([ops.conv2d], stddev=0.1, activation=tf.nn.relu, batch_norm_params=batch_norm_params):
            end_points['conv1'] = ops.conv2d(inputs, 12, [6, 6], scope='conv1')
            end_points['pool1'] = ops.max_pool(end_points['conv1'], [2, 2], scope='pool1')
            end_points['conv2'] = ops.conv2d(inputs, 12, [6, 6], scope='conv2')
            end_points['pool2'] = ops.max_pool(end_points['conv2'], [2, 2], scope='pool2')
            end_points['conv3'] = ops.conv2d(inputs, 12, [6, 6], scope='conv3')

            end_points['pool3'] = ops.max_pool(end_points['conv2'], [141, 1], scope='pool3')

            flatten = ops.flatten(end_points['pool3'], scope='flatten4')
            end_points['logits'] = ops.fc(flatten, num_classes, activation=None, scope='logits')
            # Softmax is happening in loss function

            for endpoint in end_points:
                print tf.shape(endpoint)


            return end_points['logits'], end_points


def inference(images, num_classes):
    logits, endpoints = create_model(images, num_classes=num_classes)

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
