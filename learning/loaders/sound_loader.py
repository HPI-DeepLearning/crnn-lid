import tensorflow as tf
import os
from math import ceil

import image_generators

def batch_inputs(csv_path, batch_size, data_shape, segment_length, image_generator, num_preprocess_threads=1):

    with tf.name_scope('batch_processing'):

        # load csv content
        file_path = tf.train.string_input_producer([csv_path])

        textReader = tf.TextLineReader()
        _, csv_content = textReader.read(file_path)
        sound_path_label = tf.decode_csv(csv_content, record_defaults=[[""], [0]])

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):

            if data_shape is None:
                raise ValueError("Please specify the image dimensions")

            # Load WAV files and convert them to a sequence of Mel-filtered spectrograms
            # TF needs static shape, so all images have same shape
            sound_path, label = sound_path_label

            [image_list, label_list] = tf.py_func(image_generator.wav_to_spectrogram, [sound_path, label, data_shape, segment_length], [tf.float32, tf.int32])
            images_and_labels.append([image_list, label_list])

        # Create batches
        images, labels = tf.train.shuffle_batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            shapes=[data_shape, []],
            enqueue_many=True,
            min_after_dequeue=32,
            # min_after_dequeue=1000 / segment_length
        )

        # Finally, rescale to [-1,1] instead of [0, 1)
        images_normalized = tf.div(images, 255.0)
        images_normalized = tf.sub(images_normalized, 0.5)
        images_normalized = tf.mul(images_normalized, 2.0)

        prefix = os.path.basename(csv_path)
        tf.image_summary("%s_raw" % prefix, images_normalized, max_images=10)

        return images_normalized, labels

def get(config):
    # Generates image, label batches

    if not os.path.isfile(config["train_data_dir"]):
        print('No file found for dataset %s' % config["train_data_dir"])
        exit(-1)

    image_type = config["image_type"]  # "mel" or "spectrogram" or "spectrogram2"
    image_generator = getattr(image_generators, image_type)

    data_shape = [
        config[image_type + "_image_height"],
        int(ceil(config[image_type + "_image_width"] * config["segment_length"])),
        config[image_type + "_image_depth"]
    ]

    with tf.device('/cpu:0'):
        images, labels = batch_inputs(config["train_data_dir"], config["batch_size"], data_shape, config["segment_length"], image_generator)

    return images, labels


