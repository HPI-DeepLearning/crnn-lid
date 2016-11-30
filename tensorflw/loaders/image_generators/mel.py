import tensorflw as tf
import numpy as np
import os
import sys

lib_dir = os.path.join(os.getcwd(), "..")
sys.path.append(lib_dir)

import scipy.io.wavfile as wav
from preprocessing import graphic, audio

FLAGS = tf.app.flags.FLAGS

def create_mel_spectrogram(sample_rate, signal, num_filter):
    mel_image = audio.filterbank_energies = audio.melfilterbank.logfilter(sample_rate, signal, winlen=0.00833,
                                                                          winstep=0.00833, nfilt=num_filter,
                                                                          lowfreq=0, preemph=1.0)
    mel_image = graphic.colormapping.to_grayscale(mel_image, bytes=True)
    mel_image = graphic.histeq.histeq(mel_image)
    # mel_image = graphic.colormapping.to_rgba(mel_image, colormap="magma", bytes=True)
    return np.expand_dims(mel_image, -1)


def wav_to_spectrogram(sound_file, label, data_shape, segment_length=1):

    sample_rate, signal = wav.read(sound_file)
    image_height, image_width = data_shape[:2]

    signal_duration = len(signal) / sample_rate  # seconds

    # Segment signal into smaller chunks
    images = []
    for i in np.arange(0, signal_duration, segment_length):
        chunk_duration = sample_rate * segment_length
        chunk_start = i * chunk_duration
        chunk_end = chunk_start + chunk_duration

        if chunk_end > len(signal):
            break

        signal_chunk = signal[chunk_start:chunk_end]

        # REMEMBER: Update config shape, when changing melfilter params
        img = create_mel_spectrogram(sample_rate, signal_chunk, image_height)
        images.append(img)

    # If this clip is too short for segmentation, create some dummy data
    if len(images) == 0:
        images.append(np.ones(data_shape))

    # Augment spectrograms by creating various length ones
    # images = [
    #     create_spectrogram(sample_rate, signal, image_height),
    #     create_spectrogram(0.9 * sample_rate, signal, image_height),
    #     create_spectrogram(1.1 * sample_rate, signal, image_height)
    # ]

    # images =  map(lambda image: np.squeeze(image[:,:,:3]), images)
    images_normal = map(lambda image: graphic.windowing.cut_or_pad_window(image, image_width).astype(np.float32), images)
    labels = [label] * len(images_normal)

    if len(images) == 0:
        print(sound_file)

    return [images_normal, labels]


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

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):

            if data_shape is None:
                raise ValueError("Please specify the image dimensions")

            # Load WAV files and convert them to a sequence of Mel-filtered spectrograms
            # TF needs static shape, so all images have same shape
            sound_path, label = sound_path_label

            [image_list, label_list] = tf.py_func(wav_to_spectrogram, [sound_path, label, data_shape, segment_length], [tf.float32, tf.int32])
            images_and_labels.append(image_list, label_list)

        # Create batches
        images, labels = tf.train.shuffle_batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=1000 / segment_length,
            shapes=[data_shape, []],
            enqueue_many=True,
            min_after_dequeue=1000 / segment_length
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


