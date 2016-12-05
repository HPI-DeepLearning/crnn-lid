import os
import argparse
import fnmatch
import math
import itertools
from random import shuffle

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


def create_csv(root_dir, train_test_split=0.8):
    languages = get_immediate_subdirectories(root_dir)
    counter = {}
    file_names = {}

    # Count all files for each language
    for lang in languages:
        print lang
        files = list(recursive_glob(os.path.join(root_dir, lang), "*.wav"))
        files.extend(recursive_glob(os.path.join(root_dir, lang), "*.png"))
        num_files = len(files)

        file_names[lang] = files
        counter[lang] = num_files

    # Calculate train / validation split
    print(counter)
    smallest_count = min(counter.values())

    num_train = int(smallest_count * train_test_split)
    num_validation = smallest_count - num_train

    # Split datasets and shuffle languages
    training_set = []
    validation_set = []

    for lang in languages:
        label = LABELS[lang]
        training_set += zip(file_names[lang][:num_train], itertools.repeat(label))
        validation_set += zip(file_names[lang][num_train:num_train + num_validation], itertools.repeat(label))

    shuffle(training_set)
    shuffle(validation_set)

    # Write to CSV
    train_file = open(os.path.join(root_dir, "training.csv"), "w")
    validation_file = open(os.path.join(root_dir, "validation.csv"), "w")

    for (filename, label) in training_set:
        train_file.write("{}, {}\n".format(filename, label))

    for (filename, label) in validation_set:
        validation_file.write("{}, {}\n".format(filename, label))

    train_file.close()
    validation_file.close()

    # Stats
    print("[Training]   Files per language: {} Total: {}".format(num_train, num_train * len(languages)))
    print("[Validation] Files per language: {} Total: {}".format(num_validation, num_validation * len(languages)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='root_dir', default=os.getcwd())
    parser.add_argument('--split', dest='train_test_split', default=0.8)
    args = parser.parse_args()

    create_csv(args.root_dir, args.train_test_split)
