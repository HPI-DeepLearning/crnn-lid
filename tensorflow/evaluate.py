from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.contrib import metrics
import math
import os.path
import time
from datetime import datetime
from yaml import load
import model
import tfrecord_loader


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('config', 'dataset_config.yaml',
                           """Config file""")

dataset_config = load(open(FLAGS.config, "rb"))


def eval_once(sess, summary_writer, summary_op, global_step):
    # Runs Eval once.

    pass




def evaluate():
    """Evaluate model on Dataset for a number of steps."""

    with tf.Graph().as_default():

        #temp = set(tf.all_variables())
        #sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        image_shape = [dataset_config["image_width"], dataset_config["image_height"], dataset_config["image_depth"]]
        images, labels = tfrecord_loader.get(dataset_config["data_dir"], image_shape, dataset_config["batch_size"], "train")

        # Init Model
        logits = model.inference(images, dataset_config["num_classes"])
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)  # tf.nn.softmax(logits)


        # Calculate predictions.
        accuracy_op, update_accuracy_op = metrics.streaming_accuracy(predictions, labels)
        precision_op, update_precision_op = metrics.streaming_precision(predictions, labels)
        recall_op, update_recall_op = metrics.streaming_recall(predictions, labels)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(tf.initialize_local_variables())

        # Restore model from checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Restores from checkpoint with relative path.
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                                 ckpt.model_checkpoint_path))

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        #eval_once(sess, summary_writer, summary_op, global_step)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(dataset_config["num_classes"] / dataset_config["batch_size"]))

            # Counts the number of correct predictions.
            total_sample_count = num_iter * dataset_config["batch_size"]
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), dataset_config["data_dir"],))
            start_time = time.time()

            while step < num_iter and not coord.should_stop():

                sess.run([update_accuracy_op, update_precision_op, update_recall_op])
                print(sess.run(predictions))
                print(sess.run(labels))

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = dataset_config["batch_size"] / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            accuracy, precision_at_1, recall_at_1 = sess.run([accuracy_op, precision_op, recall_op])
            print('%s: accuracy = %.4f precision@1 = %.4f recall@1 = %.4f [%d examples]' %
                  (datetime.now(), accuracy, precision_at_1, recall_at_1, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision', simple_value=float(precision_at_1))
            summary.value.add(tag='Recall', simple_value=float(recall_at_1))
            summary.value.add(tag='Accuracy', simple_value=float(accuracy))
            summary_writer.add_summary(summary, global_step)


        except Exception as e:
            coord.request_stop(e)


        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    evaluate()
