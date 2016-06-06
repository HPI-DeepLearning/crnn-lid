from pyspark import SparkContext
from util import *
import graphic
import output
import sys
import scipy.signal
from preprocessing_commons import sliding_audio, downsample, apply_melfilter, read_wav, generate_spectrograms

def main(args):
  window_size = 600
  files = filecollector.collect(args.input_path)

  sc = SparkContext("local", "sparkline")
  pipeline = (
    sc.parallelize(files, 4)
    .map(lambda f: read_wav(f))
    .flatMap(lambda (f, signal, samplerate): sliding_audio(f, signal, samplerate))
    .map(lambda (f, signal, samplerate): downsample(f, signal, samplerate))
    .map(lambda (f, signal, samplerate): apply_melfilter(f, signal, samplerate))
    .map(lambda (f, image): (f, graphic.colormapping.to_grayscale(image, bytes=True)))
    .map(lambda (f, image): (f, graphic.histeq.histeq(image)))
    .map(lambda (f, image): (f, graphic.histeq.clamp_and_equalize(image)))
    .map(lambda (f, image): (f, graphic.windowing.cut_or_pad_window(image, window_size)))
    .map(lambda (f, image): output.image.save(f, image, args.output_path))
  )

  pipeline.collect()

#.map(lambda (f, signal, samplerate): generate_spectrograms(f, signal, samplerate))
if __name__ == '__main__':

  args = argparser.parse()
  main(args)
