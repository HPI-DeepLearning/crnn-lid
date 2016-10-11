import tensorflow as tf
import numpy as np
import os
import sys

lib_dir = os.path.join(os.getcwd(), "..")
sys.path.append(lib_dir)

import scipy.io.wavfile as wav
from preprocessing import graphic, audio

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

def create_spectrogram(sample_rate, signal, num_filter):
    mel_image = audio.filterbank_energies = audio.melfilterbank.logfilter(sample_rate, signal, winlen=0.00833,
                                                                          winstep=0.00833, nfilt=num_filter,
                                                                          lowfreq=0, preemph=1.0)
    mel_image = graphic.colormapping.to_grayscale(mel_image, bytes=True)
    mel_image = graphic.histeq.histeq(mel_image)
    return np.expand_dims(mel_image, -1)


def wav_to_spectrogram(sound_file, data_shape):

    sample_rate, signal = wav.read(sound_file)
    image_height, image_width = data_shape[:2]

    # REMEMBER: Update config shape, when changing melfilter params

    # Augment spectrograms by creating various length ones
    mel_images = [
        create_spectrogram(sample_rate, signal, image_height),
        create_spectrogram(0.9 * sample_rate, signal, image_height),
        create_spectrogram(1.1 * sample_rate, signal, image_height)
    ]

    mel_images_normal = map(lambda image: graphic.windowing.cut_or_pad_window(image, image_width), mel_images)

    return np.array(mel_images_normal)


def reshape_image(image, data_shape):


    _image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    _image.set_shape(data_shape)

    # Finally, rescale to [-1,1] instead of [0, 1)
    _image = tf.div(_image, 255.0)
    _image = tf.sub(_image, 0.5)
    _image = tf.mul(_image, 2.0)

    return _image


def augment_image(image):

    return tf.image.flip_left_right(image)


def batch_inputs(csv_path, batch_size, data_shape, num_preprocess_threads=4, num_readers=1):
    with tf.name_scope('batch_processing'):

        # load csv content
        file_path = tf.train.string_input_producer([csv_path])


        # Approximate number of examples per shard.
        examples_per_shard = 512  # 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB

        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string, tf.int32])

        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TextLineReader()
                _, csv_content = reader.read(file_path)
                decode_op = tf.decode_csv(csv_content, record_defaults=[[""], [0]])
                enqueue_ops.append(examples_queue.enqueue(decode_op))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            sound_path_label = examples_queue.dequeue()

        else:

            textReader = tf.TextLineReader()
            _, csv_content = textReader.read(file_path)
            sound_path_label = tf.decode_csv(csv_content, record_defaults=[[""], [0]])



        images_and_labels = []
        for thread_id in range(num_preprocess_threads):

            if data_shape is None:
                raise ValueError("Please specify the image dimensions")

            height, width, depth = data_shape

            # Load WAV files and convert them to a sequence of Mel-filtered spectrograms
            # TF needs static shape, so all images have same shape
            sound_path, label = sound_path_label

            images = tf.py_func(wav_to_spectrogram, [sound_path, data_shape], [tf.double])[0]
            images = reshape_image(images, [3] + data_shape)

            for image in tf.unpack(images, axis=0):
                images_and_labels.append([image, label])


        # Create batches
        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            #shapes=[data_shape, []],
        )

        prefix = os.path.basename(csv_path)
        tf.image_summary("%s raw_images" % prefix, images, max_images=10)

        return images, tf.reshape(label_index_batch, [batch_size])


def get(csv_path, data_shape, batch_size=32):
    # Generates image, label batches

    if not os.path.isfile(csv_path):
        print('No file found for dataset %s' % csv_path)
        exit(-1)


    with tf.device('/cpu:0'):
        images, labels = batch_inputs(csv_path, batch_size, data_shape)

    return images, labels


