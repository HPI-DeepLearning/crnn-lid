import os.path
import subprocess
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model_deploy
from yaml import load

from models import crnn_model

from loaders import sound_loader
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

        sess = tf.InteractiveSession()
        with sess.as_default():

            deployment_config = model_deploy.DeploymentConfig(num_clones=FLAGS.num_gpus, clone_on_cpu=True)

            # Create the global step on the device storing the variables.
            with tf.device(deployment_config.variables_device()):
                global_step = slim.create_global_step()

            with tf.device(deployment_config.inputs_device()):
                # Init Data Loader
                loader = sound_loader  # image_loader
                image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
                images, labels = loader.get(config["train_data_dir"], image_shape, config["batch_size"])

            def model_fn():

                # Init Model
                model = crnn_model
                logits = model.inference(images, config)
                loss_op = model.loss(logits, labels, config["batch_size"])
                # prediction_op = tf.cast(tf.argmax(tf.nn.softmax(logits[-1]), 1), tf.int32) # For evaluation
                # tf.scalar_summary("loss", loss_op)


            with tf.device(deployment_config.optimizer_device()):
                # Adam optimizer already does LR decay
                optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"], beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                               name="AdamOptimizer")

            model_dp = model_deploy.deploy(deployment_config, model_fn, optimizer=optimizer)
            train_op = model_dp.train_op
            loss_op = model_dp.total_loss

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

            log_dir = os.path.join(FLAGS.log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            os.makedirs(log_dir)
            summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

            # Learning Loop
            for step in range(config["max_train_steps"]):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss_op])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), "Model diverged with loss = NaN"

                # Print the loss & examples/sec periodically
                if step % 1 == 0:
                    examples_per_sec = config["batch_size"] / float(duration)
                    format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

                # # Evaluate a test batch periodically
                # if step % 100 == 0:
                #     prediction_op = tf.cast(tf.argmax(tf.nn.softmax(logits[-1]), 1), tf.int32)  # For evaluation
                #
                #     predicted_labels, true_labels = sess.run([prediction_op, labels])
                #     evaluation_metrics(true_labels, predicted_labels, summary_writer, step)

                # Save the summary periodically
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == config["max_train_steps"]:
                    checkpoint_path = os.path.join(log_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)

    command = ["python", "evaluate.py", "--checkpoint_dir", log_dir, "--log_dir", "log/test"]
    subprocess.check_call(command)

if __name__ == "__main__":

    config["training_mode"] = True
    train()
