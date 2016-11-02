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


def wav_to_spectrogram(sound_file, label, data_shape, segment_length=1):

    sample_rate, signal = wav.read(sound_file)
    image_height, image_width = data_shape[:2]

    signal_duration = len(signal) / sample_rate  # seconds

    # Segment signal into smaller chunks
    mel_images = []
    for i in np.arange(0, signal_duration, segment_length):
        start_chunk = i * sample_rate
        end_chunk = start_chunk + sample_rate

        if end_chunk > len(signal):
            break

        signal_chunk = signal[start_chunk:end_chunk]

        # REMEMBER: Update config shape, when changing melfilter params
        img = create_spectrogram(sample_rate, signal_chunk, image_height)
        mel_images.append(img)

    # If this clip is too short for segmentation, create some dummy data
    if len(mel_images) == 0:
        mel_images.append(np.ones(data_shape))

    # Augment spectrograms by creating various length ones
    # mel_images = [
    #     create_spectrogram(sample_rate, signal, image_height),
    #     create_spectrogram(0.9 * sample_rate, signal, image_height),
    #     create_spectrogram(1.1 * sample_rate, signal, image_height)
    # ]

    mel_images_normal = map(lambda image: graphic.windowing.cut_or_pad_window(image, image_width).astype(np.float32), mel_images)
    labels = [label] * len(mel_images_normal)

    if len(mel_images) == 0:
        print(sound_file)

    return [mel_images_normal, labels]


def batch_inputs(csv_path, batch_size, data_shape, segment_length, num_preprocess_threads=4, num_readers=1):
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

        for thread_id in range(num_preprocess_threads):

            if data_shape is None:
                raise ValueError("Please specify the image dimensions")

            # Load WAV files and convert them to a sequence of Mel-filtered spectrograms
            # TF needs static shape, so all images have same shape
            sound_path, label = sound_path_label

            [image_list, label_list] = tf.py_func(wav_to_spectrogram, [sound_path, label, data_shape, segment_length], [tf.float32, tf.int32])

        # Create batches
        images = tf.train.batch_join(
            [[image_list]],
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            shapes=[data_shape],
            enqueue_many=True
        )

        labels = tf.train.batch_join(
            [[label_list]],
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            shapes=[[]],
            enqueue_many=True
        )

        # Finally, rescale to [-1,1] instead of [0, 1)
        images_normalized = tf.div(images, 255.0)
        images_normalized = tf.sub(images_normalized, 0.5)
        images_normalized = tf.mul(images_normalized, 2.0)

        prefix = os.path.basename(csv_path)
        tf.image_summary("%s_raw" % prefix, images_normalized, max_images=10)

        return images_normalized, labels

def get(csv_path, data_shape, batch_size=32, segment_length=10):
    # Generates image, label batches

    if not os.path.isfile(csv_path):
        print('No file found for dataset %s' % csv_path)
        exit(-1)


    with tf.device('/cpu:0'):
        images, labels = batch_inputs(csv_path, batch_size, data_shape, segment_length)

    return images, labels


