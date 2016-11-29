import numpy as np
import csv
from random import shuffle

from scipy.misc import imread
from keras.utils.np_utils import to_categorical


class CSVImageLoader():
    def __init__(self, data_dir, config):

        self.config = config
        self.images_label_pairs = []
        self.input_shape = tuple(config["input_shape"])

        with open(data_dir, "rb") as csvfile:
            for row in csv.reader(csvfile):
                self.images_label_pairs.append(row)

    def get_data(self):


        start = 0

        while True:

            image_batch = np.zeros((self.config["batch_size"], ) + self.input_shape)  # (batch_size, rows, cols, channels)
            label_batch = np.zeros((self.config["batch_size"], self.config["num_classes"]))  # (batch_size,  num_classes)

            for i, (image_path, label) in enumerate(self.images_label_pairs[start:start + self.config["batch_size"]]):
                image = imread(image_path, mode=self.config["color_mode"])

                # Image shape should be (rows, cols, channels)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, -1)

                assert len(image.shape) == 3

                height, width, channels = image.shape
                image_batch[i, : height, :width, :] = image
                label_batch[i, :] = to_categorical([label], nb_classes=self.config["num_classes"]) # one-hot encoding

            start += self.config["batch_size"]

            # Reset generator
            if start + self.config["batch_size"] > self.get_num_files():
                start = 0
                self.images_label_pairs = shuffle(self.images_label_pairs)

            yield image_batch, label_batch

    def get_input_shape(self):

        return self.input_shape

    def get_num_files(self):

        return len(self.images_label_pairs)
