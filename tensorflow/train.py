import numpy as np
import tensorflow as tf
import time
import os.path
from datetime import datetime
from yaml import load
import model
# import tfrecord_loader
import csv_loader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('config', 'dataset_config.yaml',
                           """Config file""")


def train():
    if FLAGS.config is None:
        print("Please provide a config.")

    dataset_config = load(open(FLAGS.config, "rb"))

    with tf.Graph().as_default():

        # Init Data Loader
        image_shape = [dataset_config["image_height"], dataset_config["image_width"], dataset_config["image_depth"]]
        images, labels = csv_loader.get(dataset_config["data_dir"], image_shape, dataset_config["batch_size"], "train")


        # Init Model
        logits = model.inference(images, dataset_config["num_classes"])
        loss_op = model.loss(logits, labels, dataset_config["batch_size"])
        tf.scalar_summary('loss', loss_op)


        # Adam optimizer already does LR decay
        train_op = tf.train.AdamOptimizer(learning_rate=dataset_config["learning_rate"], beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                           name='AdamOptimizer').minimize(loss_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        # Build the summary operation from all summaries.
        # Learning Rate is created by AdamOptimizer
        # lr = tf.get_variable("_lr_t")
        # tf.scalar_summary('learning_rate', lr)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        summary_op = tf.merge_all_summaries()

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        # Learning Loop
        for step in xrange(dataset_config["max_steps"]):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss_op])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Print the loss & examples/sec periodically
            if step % 10 == 0:
                examples_per_sec = dataset_config["batch_size"] / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            # Save the summary periodically
            if step % 1 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == dataset_config["max_steps"]:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
