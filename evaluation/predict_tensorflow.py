from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf

from inception import inception_model as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', '/Users/therold/Programming/ML/deeptank/snapshots/model.ckpt-49999',
                           """Directory where to read model checkpoints.""")

#tf.app.flags.DEFINE_string('file_name', '../../dataset/training_data/tiger-i/tiger-i_0.png',
#tf.app.flags.DEFINE_string('file_name', '../../dataset/maps/background_41.png',
tf.app.flags.DEFINE_string('file_name', '/Users/therold/Programming/ML/deeptank/video_analysis/snapshots/snap_10.png',
                           """Path to prediction image.""")

tf.app.flags.DEFINE_integer('num_classes', '2',
                            """Number of classes.""")

def predict():
    """Do a prediction for a single image on the Inception model"""
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Get images and labels from the dataset.
            image_data = tf.gfile.FastGFile(FLAGS.file_name, 'r').read()
            image = sess.run(tf.image.decode_png(image_data, channels=3))
            assert len(image.shape) == 3
            assert image.shape[2] == 3

            # After this point, all image pixels reside in [0,1)
            # until the very end, when they're rescaled to (-1, 1).  The various
            # adjust_* ops all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            # Create a batch of 1. Final shape=[1, 299, 299, 3]
            image = tf.expand_dims(image, 0)

            # Number of classes in the Dataset label set plus 1.
            # Label 0 is reserved for an (unused) background class.
            num_classes = FLAGS.num_classes + 1

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits, _ = inception.inference(image, num_classes)

            if os.path.isabs(FLAGS.checkpoint):

                # Restores from checkpoint with absolute path.
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.checkpoint)

                pred_op = tf.nn.softmax(logits)
                label_op = tf.argmax(pred_op, dimension=1)

                prediction, label = sess.run([pred_op, label_op])
                print("Probabilities: ", prediction)
                print("Label: ", label)
                print(prediction.shape)

if __name__ == '__main__':

    predict()
