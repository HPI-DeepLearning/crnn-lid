import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def decode_image(image_buffer, data_shape):
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape(data_shape)


# Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label


def batch_inputs(data_files, batch_size, data_shape, num_preprocess_threads=4, num_readers=1):
    with tf.name_scope('batch_processing'):

        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=True,
                                                        capacity=16)

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
            dtypes=[tf.string])

        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            if data_shape is None:
                raise ValueError('Please specify the image dimensions')


            image_buffer, label_index = parse_example_proto(example_serialized)
            image = decode_image(image_buffer, data_shape)

            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height, width, depth = data_shape

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])
        tf.image_summary('raw_images', images)

        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])


def get(data_dir, data_shape, batch_size=32, subset="train"):
    # Generates image, label batches

    # Returns:
    # images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
    #                                    image_size, 3].
    # labels: 1-D integer Tensor of [FLAGS.batch_size].

    tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
        print('No files found for dataset %s/%s' % (data_dir, subset))
        exit(-1)

    with tf.device('/cpu:0'):
        images, labels = batch_inputs(data_files, batch_size, data_shape)

    return images, labels
