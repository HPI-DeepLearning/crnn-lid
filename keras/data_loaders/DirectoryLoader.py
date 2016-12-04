import os
import random
import numpy as np
from itertools import cycle
from keras.utils.np_utils import to_categorical
from keras.engine.training import generator_queue

from SpectrogramGenerator import SpectrogramGenerator

class DirectoryLoader(object):

    def __init__(self, source_directory, config, shuffle=True):

        self.config = config
        self.source_directory = source_directory
        self.shuffle = shuffle

        # Start a spectrogram generator for each class
        # Each generator will scan a directory for audio files and convert them to spectrogram images

        self.generators = [
            SpectrogramGenerator(os.path.join(self.source_directory, "english"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "german"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "french"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "spanish"), config, shuffle=shuffle),
        ]

        # generator_queue returns (data_gen_queue, _stop, generator_threads)
        max_q_size = 50

        # self.generator_queues = [generator_queue(SpectrogramGen.get_generator(), max_q_size=max_q_size, nb_worker=1)[0] for SpectrogramGen in self.generators]
        self.generator_queues = [SpectrogramGen.get_generator() for SpectrogramGen in self.generators]

    def get_data(self):

        config = self.config

        while True:

            num_classes = len(self.generators)
            if self.shuffle:
                sample_selection = [random.randint(0, num_classes - 1) for r in range(config["batch_size"])]
            else:
                label_sequence = cycle(range(num_classes))  # generate (0, 1, 2, 3, 0, 1, 2, 3, ...)
                sample_selection = [label_sequence.next() for r in range(config["batch_size"])]

            data_batch = np.zeros((self.config["batch_size"], ) + tuple(self.config["input_shape"]))  # (batch_size, cols, rows, channels)
            label_batch = np.zeros((self.config["batch_size"], self.config["num_classes"]))  # (batch_size,  num_classes)

            for i, label in enumerate(sample_selection):

                # data = self.generator_queues[label].get()
                data = self.generator_queues[label].next()
                data = np.divide(data, 255.0)

                height, width, channels = data.shape
                data_batch[i, : height, :width, :] = data
                label_batch[i, :] = to_categorical([label], nb_classes=self.config["num_classes"])  # one-hot encoding

            yield data_batch, label_batch


    def get_num_files(self):

        # Number of files per class can vary. Use the smallest set as reference.
        min_num_files = min(map(lambda x: x.get_num_files(), self.generators))

        return len(self.generators) * min_num_files * self.config["batch_size"]



if __name__ == "__main__":

    import scipy.misc
    a = DirectoryLoader("/Users/therold/Downloads/Speech Data/EU Speech", {"pixel_per_second": 50, "input_shape": [129, 100, 1], "batch_size": 32, "num_classes": 4}, shuffle=True)
    print a.get_num_files()



    for (data, labels) in a.get_data():
        # print data, labels

        i = 0
        for image in np.vsplit(data, 32):
            scipy.misc.imsave("/Users/therold/Downloads/Speech Data/EU Speech/png/%s.png" % i, np.squeeze(image) * 255.)
            i += 1