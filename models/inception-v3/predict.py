from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import numpy as np
import tensorflow as tf

from inception import inception_model as inception

lib_path = os.path.abspath(os.path.join('../../preprocessing'))
sys.path.append(lib_path)
from preprocessing_commons import wav_to_images

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', './snapshots/model.ckpt-49999',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('file_name', '../../evaluation/english.wav',
                           """Path to prediction sound file..""")

tf.app.flags.DEFINE_integer('num_classes', '4',
                            """Number of classes.""")

def predict(images):
    """Do a prediction for a single image on the Inception model"""
    with tf.Graph().as_default():

        with tf.Session() as sess:

            batch_size = len(images)
            image_batch = np.empty((batch_size, 299, 299, 3))


            # Decode images
            for i, image in enumerate(images):
                image_data = tf.gfile.FastGFile(image, 'r').read()
                image = sess.run(tf.image.decode_png(image_data, channels=3))

                assert len(image.shape) == 3
                assert image.shape[2] == 3

                # After this point, all image pixels reside in [0,1)
                # until the very end, when they're rescaled to (-1, 1).  The various
                # adjust_* ops all require this range for dtype float.
                image_batch[i] = tf.image.convert_image_dtype(image, dtype=tf.float32).eval()


            # Number of classes in the Dataset label set plus 1.
            # Label 0 is reserved for an (unused) background class.
            num_classes = FLAGS.num_classes + 1

            # Build a Graph that computes the logits predictions from the
            # inference model.
            batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
            print(image_batch)
            logits, _ = inception.inference(batch, num_classes)

            if os.path.isfile(FLAGS.checkpoint):
                print("dfdsf")

                # Restores from checkpoint with absolute path.
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.checkpoint)

                pred_op = tf.nn.softmax(logits)

                label_op = tf.reduce_mean(tf.argmax(pred_op, dimension=1))

                prediction, label = sess.run([pred_op, label_op])
                print("Probabilities: ", prediction)
                print("Label: ", label)


if __name__ == '__main__':

    file_names = wav_to_images(FLAGS.file_name, "/tmp")["spectros"]

    predict(file_names)
