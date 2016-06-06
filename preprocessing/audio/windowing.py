import math

def generate_windows(signal, samplerate, windowsize = 5, stride = None):
  if not stride:
    stride = windowsize

  number_windows = int(math.ceil((len(signal)/float(samplerate)) / float(stride)))

  for i in range(number_windows):
    window = signal[i*stride*samplerate:i*stride*samplerate+windowsize*samplerate]
    yield window

def sliding(signal, samplerate, windowsize = 5, stride = None, cutoff = 0.0):
  for window in generate_windows(signal, samplerate, windowsize, stride):
    if len(window) >= windowsize * samplerate * cutoff:
      yield window
    else:
      continue

def sliding_with_filename(filename, signal, samplerate, windowsize = 5, stride = None, cutoff = 0.0):
  for i, window in enumerate(sliding(signal, samplerate, windowsize, stride, cutoff)):
    counter = "_%02d" % i 
    window_filename = "".join([filename, counter])
    yield (window_filename, window)