import os.path
import re
import subprocess
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from yaml import load

# from models import cnn_model
from models import crnn_model
# from models import lstm_model

# import tfrecord_loader
# import csv_loader
from learning.loaders import sound_loader

# Defines are in "evaluation.py"
FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string("log_dir", "log", """Directory where to write event logs and checkpoint.""")
# tf.app.flags.DEFINE_string("config", "config.yaml", """Path to config.yaml file""")


config = load(open(FLAGS.config, "rb"))


def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the tower, e.g. "tower_0"
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Init Data Loader
    loader = sound_loader  # image_loader
    image_shape = [config["image_height"], config["image_width"], config["image_depth"]]
    images, labels = loader.get(config["train_data_dir"], image_shape, config["batch_size"])

    # Init Model
    model = crnn_model
    logits = model.inference(images, config)

    prediction_op = tf.cast(tf.argmax(tf.nn.softmax(logits[-1]), 1), tf.int32)  # For evaluation # TODO

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model.loss(logits, labels, config["batch_size"])

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection("losses", scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name="total_loss")

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove "tower_[0-9]/" from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub("%s_[0-9]*/tower", "", l.op.name)
        # Name each loss as "(raw)" and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss_name + " (raw)", l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a "tower" dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the "tower" dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower"s pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train():
    if FLAGS.config is None:
        print("Please provide a config.")

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            "global_step", [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Adam optimizer already does LR decay
        optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"], beta1=0.9, beta2=0.999, epsilon=1e-08,
                                           use_locking=False, name="AdamOptimizer")

        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("%s_%d" % ("tower", i)) as scope:
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = tower_loss(scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.histogram_summary(var.op.name + "/gradients", grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        log_dir = os.path.join(FLAGS.log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(log_dir)
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

        for step in range(config["max_train_steps"]):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(
                loss_value), "Model diverged with loss = NaN"  # Print the loss & examples/sec periodically

            if step % 1 == 0:
                examples_per_sec = config["batch_size"] / float(duration)
                format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

                # # Evaluate a test batch periodically
                # if step % 100 == 0:
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
    train()
