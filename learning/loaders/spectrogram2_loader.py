import tensorflow as tf
import numpy as np
import os
import sys

lib_dir = os.path.join(os.getcwd(), "..")
sys.path.append(lib_dir)

from numpy.lib import stride_tricks
import scipy.io.wavfile as wav
from preprocessing import graphic

FLAGS = tf.app.flags.FLAGS

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:128]
    time_bins, freq_bins = np.shape(spec)
    scale = np.linspace(0, 1, freq_bins)  # ** factor

    # Voice Perturbation
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(
        map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale))
    scale *= (freq_bins - 1) / max(scale)

    newspec = np.complex128(np.zeros([time_bins, freq_bins]))
    all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
    freqs = [0.0] * freq_bins
    totw = [0.0] * freq_bins

    for i in range(0, freq_bins):
        if (i < 1 or i + 1 >= freq_bins):
            newspec[:, i] += spec[:, i]
            freqs[i] += all_freqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * all_freqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * all_freqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs


def create_spectrogram(sample_rate, samples, bin_size=1024, alpha=1):
    s = stft(samples, bin_size)

    sshow, freq = logscale_spec(s, factor=1, sr=sample_rate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    ims = np.transpose(ims)
    ims = ims[0:128, :]  # 0-5.5khz

    return np.expand_dims(ims, -1)


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
        img = create_spectrogram(sample_rate, signal_chunk)
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

    images = map(lambda image: graphic.windowing.cut_or_pad_window(image, image_width).astype(np.float32), images)
    labels = [label] * len(images)

    if len(images) == 0:
        print(sound_file)

    return [images, labels]


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


