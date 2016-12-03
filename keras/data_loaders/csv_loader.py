import numpy as np
import csv
import abc

from keras.utils.np_utils import to_categorical


class CSVLoader(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, config):

        self.config = config
        self.images_label_pairs = []
        self.input_shape = tuple(config["input_shape"])

        with open(data_dir, "rb") as csvfile:
            for (file_path, label)in list(csv.reader(csvfile)):
                self.images_label_pairs.append((file_path, int(label)))

    def get_data(self, should_shuffle=True, is_prediction=False):

        start = 0

        while True:

            data_batch = np.zeros((self.config["batch_size"], ) + self.input_shape)  # (batch_size, cols, rows, channels)
            label_batch = np.zeros((self.config["batch_size"], self.config["num_classes"]))  # (batch_size,  num_classes)

            for i, (file_path, label) in enumerate(self.images_label_pairs[start:start + self.config["batch_size"]]):

                data = self.process_file(file_path)
                height, width, channels = data.shape
                data_batch[i, : height, :width, :] = data
                label_batch[i, :] = to_categorical([label], nb_classes=self.config["num_classes"]) # one-hot encoding

            start += self.config["batch_size"]

            # Reset generator
            if start + self.config["batch_size"] > self.get_num_files():
                start = 0
                if should_shuffle:
                    np.random.shuffle(self.images_label_pairs)

            # For predicitions only return the data
            if is_prediction:
                yield data_batch
            else:
                yield data_batch, label_batch

    def get_input_shape(self):

        return self.input_shape

    def get_num_files(self):

        # Minimum number of data points without overlapping batches
        return (len(self.images_label_pairs) // self.config["batch_size"]) * self.config["batch_size"]


    def get_labels(self):

        return [label for (file_path, label) in self.images_label_pairs]


    @abc.abstractmethod
    def process_file(self, file_path):

        raise NotImplementedError("Implement in child class.")