from __future__ import division

import glob
import os.path
import shutil
import subprocess

import numpy as np
import tensorflw as tf
from yaml import load

from loaders import sound_loader
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
        subprocess.call(command)

        file_pattern = os.path.join(temp_directory, "*.wav")
        file_list = glob.glob(file_pattern)

        return file_list

    else:
        raise Exception("-file argument is not a valid file: {}".format(file_name))


def audio_to_spectrogram(file_name):

    spectrogram = sound_loader.wav_to_spectrogram(file_name)

    image = np.divide(spectrogram, 255.0)
    image = image - 0.5
    image = image * 2.0

    return image


def predict():
    """Predict class for a new audio file using a trained model."""

    with tf.Graph().as_default():

        # Load file from disk and convert to spectrograms
        file_list = file_to_segments()
        config["batch_size"] = len(file_list)

        image_data = map(audio_to_spectrogram, file_list)

        image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
        image_batch = tf.placeholder(dtype=tf.float32, shape=[config["batch_size"]] + image_shape)

        # Init Model
        logits = crnn_model.inference(image_batch, config)
        # Use the last state of the LSTM as output
        predictions_op = tf.nn.softmax(logits)

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

        # Returns probabilities for every segment
        probability = sess.run(predictions_op, feed_dict={image_batch: image_data})

        # Use median, so outlies don't interfere as much
        median_probabilities = np.median(probability, axis=0)
        label_index = np.argmax(median_probabilities)
        label = config["label_names"][label_index]

        print("Probability raw", probability)
        print("Probability: ", median_probabilities)
        print("Label: ", label, label_index)

        shutil.rmtree(temp_directory, ignore_errors=True)

        return probability, label


if __name__ == "__main__":

    config["training_mode"] = False
    predict()
