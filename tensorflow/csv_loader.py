import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def decode_image(image_path, data_shape):

    try:
        image_buffer = tf.read_file(image_path)
        image = tf.image.decode_png(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image.set_shape(data_shape)


        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.sub(image, 0.5)
        image = tf.mul(image, 2.0)

    except Exception as e:
        print("Invalid PNG file {0}. ({1})".format(image_path, e))
        image = np.zeros(data_shape)

    finally:
        return image


def batch_inputs(csv_path, batch_size, data_shape, num_preprocess_threads=4, num_readers=2):
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
            image_path_label = examples_queue.dequeue()

        else:
            textReader = tf.TextLineReader()
            _, csv_content = textReader.read(file_path)
            image_path_label = tf.decode_csv(csv_content, record_defaults=[[""], [0]])



        images_and_labels = []
        for thread_id in range(num_preprocess_threads):

            if data_shape is None:
                raise ValueError('Please specify the image dimensions')

            # load images
            image_path, label_index = image_path_label
            image = decode_image(image_path, data_shape)

            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            shapes=[data_shape, []]
        )

        # Reshape images into these desired dimensions.
        height, width, depth = data_shape

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])
        tf.image_summary('raw_images', images)

        return images, tf.reshape(label_index_batch, [batch_size])


def get(csv_path, data_shape, batch_size=32):
    # Generates image, label batches

    if not os.path.isfile(csv_path):
        print('No file found for dataset %s' % csv_path)
        exit(-1)


    with tf.device('/cpu:0'):
        images, labels = batch_inputs(csv_path, batch_size, data_shape)

    return images, labels
