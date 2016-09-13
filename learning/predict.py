from __future__ import division

import os.path
import time
from datetime import datetime
import subprocess
import glob

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from yaml import load

import sound_loader
import tensorflow as tf
from models import crnn_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("file", "", """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string("model_dir", "log", """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string("config", "config.yaml", """Path to config.yaml file""")

config = load(open(FLAGS.config, "rb"))
temp_directory = "temp_prediction"


def file_to_segments():

    file_name = FLAGS.file
    if os.path.isfile((file_name)):

        if not os.path.isdir(temp_directory):
            os.mkdir(temp_directory)

        output_file_name = os.path.join(
            temp_directory,
            os.path.basename(file_name)[:-4] + "_%03d.wav"
        )

        command = ["ffmpeg", "-y", "-i", file_name, "-map", "0", "-ac", "1", "-ar", "16000", "-f", "segment", "-segment_time", "10", output_file_name]
        print
        subprocess.call(command)

        file_pattern = os.path.join(temp_directory, "*.wav")
        file_list = glob.glob(file_pattern)

        return file_list

    else:
        raise Exception("-file argument is not a valid file: {}".format(file_name))


def predict():
    """Predict class for a new audio file using a trained model."""

    with tf.Graph().as_default():

        file_list = file_to_segments()
        config["batch_size"] = len(file_list)

        filename_queue = tf.train.string_input_producer(file_list)
        images = []

        for i in range(config["batch_size"]):

            single_segment_filename = filename_queue.dequeue()
            image = tf.py_func(sound_loader.wav_to_spectrogram, [single_segment_filename], [tf.double])[0]

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
            image.set_shape(image_shape)  # TODO Which way around is it?

            # Finally, rescale to [-1,1] instead of [0, 1)
            image = tf.sub(image, 0.5)
            image = tf.mul(image, 2.0)

            images.append(image)

        image_batch = tf.train.batch(
            images,
            batch_size=config["batch_size"],
            capacity=config["batch_size"],
        )

        # Init Model
        logits = crnn_model.inference(image_batch, config)
        # Use the last state of the LSTM as output
        predictions_op = tf.nn.softmax(logits[-1])
        label_op = tf.reduce_mean(tf.argmax(predictions_op, dimension=1))

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(tf.initialize_local_variables())

        # Restore model from checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.abspath(ckpt.model_checkpoint_path))

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            print("Succesfully loaded model from %s at step=%s." %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print("No checkpoint file found")
            return


        # Start the queue runners.
        coord = tf.train.Coordinator()
        with coord.stop_on_exception():
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                print("%s: starting evaluation on (%s)." % (datetime.now(), config["test_data_dir"],))
                step = 0
                predicted_labels = np.array([])


                probability, label = sess.run([predictions_op, label_op])
                print probability, label


            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            os.rmdir(temp_directory)


if __name__ == "__main__":
    predict()
