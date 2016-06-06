import os
import argparse
import csv
import caffe
import sys
import numpy as np
import shutil

lib_path = os.path.abspath(os.path.join('../preprocessing'))
sys.path.append(lib_path)
from preprocessing_commons import wav_to_images

def evaluate(input_csv, proto, model):
  ''' Run evaluation on a list of WAV, label files '''

  if not os.path.isdir("tmp"):
    os.mkdir("tmp")

  correct_files = []
  incorrect_files = []
  skipped_files = []

  caffe.set_mode_cpu()
  net = caffe.Classifier(proto, model, raw_scale=255) # convert 0..255 values into range 0..1

  reader = csv.reader(file(input_csv, 'rU'))
  for filename, label in reader:

    # Convert WAV to images
    image_files = wav_to_images(filename, "tmp")

    # some files fail during the conversion, so skip them
    if len(image_files["melfilter"]) == 0:
      skipped_files.append(filename)
      continue

    # Call Caffe and do predicition
    input_images = np.array([caffe.io.load_image(image_file, color=False) for image_file in image_files["melfilter"]])
    prediction = net.predict(input_images, False)
    mean_prediction = np.mean(prediction, axis=0)

    # Evaluation
    best_label = mean_prediction.argmax()
    if best_label == np.int64(label):
      correct_files.append(filename)
    else:
      incorrect_files.append(filename)

  shutil.rmtree("tmp")

  # Stats
  num_correct = len(correct_files)
  num_incorrect = len(incorrect_files)
  print "Correctly Classified: {0} ({1:.2f}%)".format(num_correct, num_correct / float(num_correct + num_incorrect) * 100)
  print "Incorrectly Classified: {0} ({1:.2f}%)".format(num_incorrect, num_incorrect / float(num_correct + num_incorrect) * 100)
  print "Skipped Files: {0}".format(len(skipped_files))

  # Save correct / incorrect filenames in txt file
  correct_out = open("correct_files.txt", "wb")
  incorrect_out = open("incorrect_files.txt", "wb")
  skipped_out = open("skipped_files.txt", "wb")

  correct_out.write("\n".join(correct_files))
  incorrect_out.write("\n".join(incorrect_files))
  skipped_out.write("\n".join(skipped_files))

  correct_out.close()
  incorrect_out.close()
  skipped_out.close()


if __name__ == '__main__':

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--csv", required=True)
  argparser.add_argument("--proto", required=True)
  argparser.add_argument("--model", required=True)

  args = argparser.parse_args()
  evaluate(args.csv, args.proto, args.model)