import os.path
import subprocess
import time
import shutil
from datetime import datetime
from math import ceil

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from yaml import load

from models import crnn_model
from models import cnn_model
from models import lstm_model
from models import topcoder_crnn

import loaders
from evaluate import evaluation_metrics

FLAGS = tf.app.flags.FLAGS
config = load(open(FLAGS.config, "rb"))

# Defines are in 'evaluation.py'
# tf.app.flags.DEFINE_string("log_dir", "log", """Directory where to write event logs and checkpoint.""")
# tf.app.flags.DEFINE_string("config", "config.yaml", """Path to config.yaml file""")


def train():
    if FLAGS.config is None:
        print("Please provide a config.")

    with tf.Graph().as_default():
        with tf.device("/gpu:0"):

            sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
            with sess.as_default():

                # Init Data Loader
                image_type = config["image_type"]  # "mel" or "spectrogram"
                loader = getattr(loaders, image_type + "_loader")

                image_shape = [
                    config[image_type + "_image_height"],
                    int(ceil(config[image_type + "_image_width"] * config["segment_length"])),
                    config[image_type + "_image_depth"]
                ]

                images, labels = loader.get(config["train_data_dir"], image_shape, config["batch_size"], config["segment_length"])

                validation_images, validation_labels = loader.get(config["validation_data_dir"], image_shape, config["batch_size"], config["segment_length"])

                # Init Model
                model = cnn_model

                scope = model.NAME
                logits, endpoints = model.create_model(images, config, is_training=True, scope=scope)
                loss_op = model.loss(logits, labels)
                prediction_op = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)
                tf.scalar_summary("loss", loss_op)

                # Add summaries for viewing model statistics on TensorBoard.
                # Make sure they are named uniquely
                summaries = {}
                for act in endpoints.values():
                    summaries[act.op.name] = act

                slim.summarize_tensors(summaries.values())

                scope = tf.VariableScope(reuse=True, name=model.NAME)
                validation_logits, _ = model.create_model(validation_images, config, is_training=False, scope=scope)
                validation_loss_op = model.loss(validation_logits, validation_labels)
                validation_prediction_op = tf.cast(tf.argmax(tf.nn.softmax(validation_logits), 1), tf.int32)
                tf.scalar_summary("validation_loss", validation_loss_op)

                # Adam optimizer already does LR decay
                train_op = tf.train.AdamOptimizer(learning_rate=config["learning_rate"], beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                                   name="AdamOptimizer").minimize(loss_op)
                # global_step = tf.Variable(0, trainable=False)
                # learning_rate = tf.train.exponential_decay(config["learning_rate"], global_step, 1000, 0.96, staircase=False, name="LearningRate")
                # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_op, global_step=global_step)
                # tf.scalar_summary("learning rate", learning_rate)

                # Create a saver.
                saver = tf.train.Saver(tf.all_variables())

                # Build an initialization operation to run below.
                init = tf.initialize_all_variables()
                sess.run(init)


                # Add histograms for trainable variables.
                for var in tf.trainable_variables():
                    tf.histogram_summary(var.op.name, var)

                summary_op = tf.merge_all_summaries()

                # Start the queue runners.
                tf.train.start_queue_runners(sess=sess)

                summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

                # Learning Loop
                for step in range(config["max_train_steps"]):
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss_op])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), "Model diverged with loss = NaN"

                    # Print the loss & examples/sec periodically
                    if step % 1 == 0 and step > 0:
                        examples_per_sec = config["batch_size"] / float(duration)
                        format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
                        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

                    # Evaluate a training batch periodically
                    if step % 100 == 0 and step > 0:
                        predicted_labels, true_labels = sess.run([prediction_op, labels])
                        evaluation_metrics(true_labels, predicted_labels, summary_writer, step, prefix="training")

                    # Run a validation set of 100*batch_size samples periodically
                    if step % 500 == 0 and step > 0:
                        eval_results = map(lambda x: sess.run([validation_loss_op, validation_prediction_op, validation_labels]), range(0, 100))
                        validation_loss, predicted_labels, true_labels = map(list, zip(*eval_results))
                        evaluation_metrics(np.concatenate(true_labels), np.concatenate(predicted_labels), summary_writer, step, prefix="validation")
                        print("Validation loss: ", np.mean(validation_loss))

                    # Save the summary periodically
                    if step % 100 == 0 and step > 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == config["max_train_steps"]:
                        checkpoint_path = os.path.join(log_dir, "model.ckpt")
                        saver.save(sess, checkpoint_path, global_step=step)

    command = ["python", "evaluate.py", "--checkpoint_dir", log_dir, "--log_dir", "log/test"]
    subprocess.check_call(command)

if __name__ == "__main__":

    log_dir = os.path.join(FLAGS.log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    print("Logging to {}".format(log_dir))

    # copy models & config for later
    shutil.copytree("models", log_dir)  # creates the log_dir
    shutil.copy("config.yaml", log_dir)


    config["training_mode"] = True
    train()
