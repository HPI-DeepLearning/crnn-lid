import os
import argparse
import fnmatch
import math
from random import shuffle

LABELS = {
  "english" : 0,
  "german" : 1,
  "french" : 2,
  "spanish" : 3,
}

def recursive_glob(path, pattern):

  for root, dirnames, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, pattern):
          yield os.path.abspath(os.path.join(path, filename))


def get_immediate_subdirectories(path):
  return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def create_csv():

  languages = get_immediate_subdirectories(args.root_dir)
  counter = {}
  file_names = {}

  # Count all files for each language
  for lang in languages:
    print lang
    files = list(recursive_glob(os.path.join(args.root_dir, lang), "*.wav"))
    num_files = len(files)

    file_names[lang] = files
    shuffle(file_names[lang])
    counter[lang] = num_files

  # Calculate train / validation split
  smallest_count = min(counter.values())
  num_train = int(smallest_count * args.train_test_split)
  num_validation = smallest_count - num_train

  # Split dataset and write to CSV
  train_file_name = os.path.join(args.root_dir, "training.csv")
  validation_file_name = os.path.join(args.root_dir, "validation.csv")

  train_file = open(train_file_name, "w")
  validation_file = open(validation_file_name, "w")

  for lang in languages:

    training_set = file_names[lang][:num_train]
    validation_set = file_names[lang][num_train:num_train + num_validation]

    for f in training_set:
      train_file.write("{}, {}\n".format(f, LABELS[lang]))

    for f in validation_set:
      validation_file.write("{}, {}\n".format(f, LABELS[lang]))

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

  create_csv()