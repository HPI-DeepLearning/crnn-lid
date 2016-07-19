import os.path
import time
from datetime import datetime

import numpy as np
from yaml import load

import tensorflow as tf
# from models import cnn_model
from models import crnn_model
# from models import lstm_model

# import tfrecord_loader
import csv_loader
# import sound_loader
from evaluate import evaluation_metrics

FLAGS = tf.app.flags.FLAGS

# Defines are in 'evaluation.py'
# tf.app.flags.DEFINE_string("log_dir", "log", """Directory where to write event logs and checkpoint.""")
# tf.app.flags.DEFINE_string("config", "config.yaml", """Path to config.yaml file""")


def train():
    if FLAGS.config is None:
        print("Please provide a config.")

    config = load(open(FLAGS.config, "rb"))

    with tf.Graph().as_default():

        sess = tf.InteractiveSession()
        with sess.as_default():

            # Init Data Loader
            image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
            images, labels = csv_loader.get(config["train_data_dir"], image_shape, config["batch_size"])
            #images, labels, sequence_lengths = sound_loader.get(config["train_data_dir"], image_shape, config["batch_size"])

            # Init Model
            model = crnn_model
            logits = model.inference(images, config)
            loss_op = model.loss(logits, labels, config["batch_size"])
            prediction_op = tf.cast(tf.argmax(logits, 1), tf.int32) # For evaluation
            tf.scalar_summary("loss", loss_op)

            # Adam optimizer already does LR decay
            train_op = tf.train.AdamOptimizer(learning_rate=config["learning_rate"], beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                               name="AdamOptimizer").minimize(loss_op)

            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()


            sess.run(init)

            # Build the summary operation from all summaries.
            # Learning Rate is created by AdamOptimizer
            # lr = tf.get_variable("_lr_t")
            # tf.scalar_summary("learning_rate", lr)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

            summary_op = tf.merge_all_summaries()

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

            # Learning Loop
            for step in range(config["max_train_steps"]):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss_op])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), "Model diverged with loss = NaN"

                # Print the loss & examples/sec periodically
                if step % 10 == 0:
                    examples_per_sec = config["batch_size"] / float(duration)
                    format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

                # Evaluate a test batch periodically
                if step % 100 == 0:
                    predicted_labels, true_labels = sess.run([prediction_op, labels])
                    evaluation_metrics(true_labels, predicted_labels, summary_writer, step)

                # Save the summary periodically
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == config["max_train_steps"]:
                    checkpoint_path = os.path.join(FLAGS.log_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    train()
