import audio
import graphic
import os
import output
import cv2
import scipy.io.wavfile as wav
from util import *
import argparse

def read_wav_dirty(f):
  samplerate, signal = wav.read(f)
  f = filename.truncate_extension(f)
  return (signal, samplerate)

def read_wav(f):
  samplerate, signal = wav.read(f)
  #if len(signal.shape) > 1:
  #  signal = signal[:,0]
  f = filename.truncate_extension(filename.clean(f))
  return (f, signal, samplerate)

def apply_melfilter(signal, samplerate, nfilt=40):
  filterbank_energies = audio.melfilterbank.logfilter(samplerate, signal, winlen=0.00833, winstep=0.00833, nfilt=nfilt, lowfreq=0, preemph=1.0)
  #print f, samplerate, filterbank_energies.shape
  return filterbank_energies

def generate_spectrograms(f, signal, samplerate):
  Sxx = audio.spectrogram.spectrogram_cutoff(samplerate, signal, winlen=0.00833, winstep=0.00833)
  return (f, Sxx)

def sliding_audio(f, signal, samplerate):
  for window_name, window in audio.windowing.sliding_with_filename(f, signal, samplerate, 5, 5, 0.6):
    yield (window_name, window, samplerate)

def downsample(f, signal, samplerate):
  target_samplerate = 16000
  downsampled_signal, downsampled_samplerate = audio.resample.downsample(signal, samplerate, target_samplerate)
  return (f, downsampled_signal, downsampled_samplerate)

def wav_to_images(sound_file, output_path):
  '''Converts a WAV file input several images and writes them to disk'''

  if not os.path.isdir(output_path):
    os.mkdir(output_path)

  # filenames of the generated images
  image_files = {
    "spectros" : [],
    "melfilter" : []
  }
  window_size = 2408 # MFCC sliding window

  f, signal, samplerate = read_wav_dirty(sound_file)
  segments = sliding_audio(f, signal, samplerate)

  for (filename, signal, samplerate) in segments:
    _, signal, samplerate = downsample(filename, signal, samplerate)

    _, mel_image = apply_melfilter(filename, signal, samplerate)
    _, spectro_image = generate_spectrograms(f, signal, samplerate)


    mel_image = graphic.colormapping.to_grayscale(mel_image, bytes=True)
    mel_image = graphic.histeq.histeq(mel_image)
    mel_image = graphic.histeq.clamp_and_equalize(mel_image)
    mel_image = graphic.windowing.cut_or_pad_window(mel_image, window_size)

    #spectro_image = graphic.colormapping.to_grayscale(spectro_image, bytes=True)
    spectro_image = graphic.colormapping.to_rgb(spectro_image, colormap="jet", bytes=True)
    spectro_image = graphic.histeq.histeq(spectro_image)
    spectro_image = graphic.histeq.clamp_and_equalize(spectro_image)
    spectro_image = graphic.windowing.cut_or_pad_window(spectro_image, window_size)

    mel_filename = "melfilter_%s" % os.path.basename(filename)
    spectro_filename = "spectrogram_%s" % os.path.basename(filename)
    output.image.save(mel_filename, mel_image, output_path)
    output.image.save(spectro_filename, cv2.resize(spectro_image, (299, 299)), output_path)

    image_files["melfilter"].append(os.path.join(output_path, mel_filename + ".png"))
    image_files["spectros"].append(os.path.join(output_path, spectro_filename + ".png"))

  return image_files

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', dest='input_path', default=os.getcwd(), help='Input Path to wav file',
                      required=True)
  parser.add_argument('--output', dest='output_path', default=os.path.join(os.getcwd(), "spectrograms"),
                      required=True,
                      help='Output Path for spectrogram images.')

  args = parser.parse_args()

  wav_to_images(args.input_path, args.output_path)