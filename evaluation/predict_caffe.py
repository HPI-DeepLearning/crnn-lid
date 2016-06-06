import numpy as np
import caffe
import os
import sys
import argparse

lib_path = os.path.abspath(os.path.join('../preprocessing'))
sys.path.append(lib_path)
from preprocessing_commons import wav_to_images


def predict(sound_file, prototxt, model, output_path):

  image_files = wav_to_images(sound_file, output_path)

  caffe.set_mode_cpu()
  net = caffe.Classifier(prototxt, model,
                         #image_dims=(224, 224)
                         #channel_swap=(2,1,0),
                         raw_scale=255 # convert 0..255 values into range 0..1
                         #caffe.TEST
                        )

  input_images = np.array([caffe.io.load_image(image_file, color=False) for image_file in image_files["melfilter"]])
  #input_images = np.swapaxes(input_images, 1, 3)

  #prediction = net.forward_all(data=input_images)["prob"]

  prediction = net.predict(input_images, False)  # predict takes any number of images, and formats them for the Caffe net automatically

  print prediction
  print 'prediction shape:', prediction[0].shape
  print 'predicted class:', prediction[0].argmax()
  print image_files

  return prediction

if __name__ == '__main__':

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--input", required=True)
  argparser.add_argument("--proto", required=True)
  argparser.add_argument("--model", required=True)
  argparser.add_argument("--output", required=True)

  args = argparser.parse_args()
  predict(args.input, args.proto, args.model, args.output)