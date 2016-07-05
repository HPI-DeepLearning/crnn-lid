from __future__ import division

import numpy as np
import tensorflow as tf
import math
import os.path
import time
from datetime import datetime
from yaml import load
import model
import csv_loader
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("log_dir", "log", """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string("checkpoint_dir", "log", """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string("config", "config.yaml", """Path to config.yaml file""")

config = load(open(FLAGS.config, "rb"))

def evaluation_metrics(true_labels, predicted_labels, summary_writer, global_step):

    try:

        available_labels = range(0, config["num_classes"])

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, labels=available_labels)

        print("Accuracy %s" % (accuracy))
        print(classification_report(true_labels, predicted_labels, labels=available_labels, target_names=config["label_names"]))
        print(confusion_matrix(true_labels, predicted_labels, labels=available_labels))

        summary = tf.Summary()
        summary.value.add(tag="Accuracy", simple_value=accuracy)

        for i, label_name in enumerate(config["label_names"]):
            summary.value.add(tag="Precision  %s" % label_name, simple_value=precision[i])
            summary.value.add(tag="Recall %s" % label_name, simple_value=recall[i])
            summary.value.add(tag="F1 %s" % label_name, simple_value=f1[i])

        summary_writer.add_summary(summary, global_step)

    except Exception as e:
        "UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples."
        pass


def evaluate():
    """Evaluate model on Dataset for a number of steps."""

    with tf.Graph().as_default():

        image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
        images, labels = csv_loader.get(config["test_data_dir"], image_shape, config["batch_size"])

        # Init Model
        logits = model.inference(images, config["num_classes"])
        predictions_op = tf.cast(tf.argmax(logits, 1), tf.int32)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(tf.initialize_local_variables())

        # Restore model from checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint with relative path.
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            print("Succesfully loaded model from %s at step=%s." %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print("No checkpoint file found")
            return

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)


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
                true_labels = np.array([])
                start_time = time.time()
                num_iter = config["num_test_files"] / config["batch_size"]

                while step < num_iter and not coord.should_stop():

                    predicted, labels = sess.run([predictions_op, labels])

                    # Keep a running tally for evaluating on the whole test set
                    predicted_labels = np.append(predicted_labels, predicted)
                    true_labels = np.append(true_labels, labels)

                    step += 1

                    # Periodically report the progress
                    if step % 100 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 100.0
                        examples_per_sec = config["batch_size"] / sec_per_batch
                        print("%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)" %
                             (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                        start_time = time.time()

                        # Temporary evals
                        evaluation_metrics(true_labels, predicted_labels, summary_writer, global_step)

                        # Logging
                        sess.run(summary_op)

                # Print overall statistics
                print("Done with evaluation.")
                evaluation_metrics(true_labels, predicted_labels, summary_writer, global_step)

            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == "__main__":
    evaluate()
