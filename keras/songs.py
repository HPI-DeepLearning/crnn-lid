
# -*- coding: utf-8 -*-

import os
import numpy as np
import fnmatch
from predict import predict
from collections import namedtuple

from sklearn.metrics import classification_report, accuracy_score

# model_name = "logs/2017-01-02-13-39-41/weights.06.model"
# model_name = "/home/mpss2015m_1/master-thesis/keras/logs/2016-12-16-16-28-42/weights.20.model"

DummyCLIArgs = namedtuple("DummyCLIArgs", ["model_dir", "input_file"])

LABELS = {
    "english": 0,
    "german": 1,
    "french": 2,
    "spanish": 3,
}

def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename


def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def eval(root_dir):
    languages = get_immediate_subdirectories(root_dir)

    # Count all files for each language
    for lang in languages:
        print(lang)
        files = list(recursive_glob(os.path.join(root_dir, lang), "*.mp3"))
        classes = []

        for file in files:
            probabilities = predict(DummyCLIArgs(model_name, file))
            average_prob = np.mean(probabilities, axis=0)
            average_class = np.argmax(average_prob)
            classes.append(average_class)

        y_true = np.full((len(classes)), LABELS[lang])

        print(lang)
        print(accuracy_score(y_true, classes))
        print(classification_report(y_true, classes))


if __name__ == '__main__':

    eval("/data/tom/songs/hiphop")
    eval("/data/tom/songs/hiphop")