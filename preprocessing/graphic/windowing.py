import numpy as np
import math

def generate_windows(image, windowsize, stride):
  """creates an generator of sliding window along the width of an image

  Args:
      image (2 or 3 dimensional numpy array): the image the sliding windows are created for
      windowsize (int): width of the sliding window
      stride (int): stepsize of the sliding

  Returns:
      generator for the sliding windows

  """
  assert len(image.shape) > 1 and len(image.shape) < 4

  number_windows = int(math.ceil(_width(image) / float(stride)))
  
  for i in range(number_windows):
    window = image[:,i*stride:i*stride+windowsize]
    assert _height(window) == _height(image)

    yield window

def pad_window(window, windowsize):
  """if necessary, pads a window with zeros at the end to ensure windowsize

  Args:
      window (2 or 3 dimensional numpy array): the window to be padded
      windowsize (int): width of the sliding window

  Returns:
      padded window (2 or 3 dimension numpy array)

  """
  assert _width(window) <= windowsize

  has_channels = len(window.shape) > 2
  window_channels = None
  if has_channels:
    window_channels = window.shape[2]

  if _width(window) < windowsize:
    missing_window_width = windowsize - _width(window)
    padding = None
    if has_channels:
      padding = np.zeros((_height(window), missing_window_width, window_channels))
    else:
      padding = np.zeros((_height(window), missing_window_width))
    window = np.append(window, padding, axis=1)

  return window

def cut_or_pad_window(window, windowsize):
  if _width(window) > windowsize:
    # cut
    assert _width(window) / float(windowsize) < 1.1 # we don't want huge cuts, just cutting some irregularities
    return window[:,:windowsize]
  else:
    return pad_window(window, windowsize)

def sliding(image, windowsize, stride, cutoff = 0.0):
  """creates an generator of sliding window along the width of an image

  Args:
      image (2 or 3 dimensional numpy array): the image the sliding windows are created for
      windowsize (int): width of the sliding window
      stride (int): stepsize of the sliding
      cutoff (float): drop windows with width below windowsize * cutoff 

  Returns:
      generator for the sliding windows
  """
  for window in generate_windows(image, windowsize, stride):
    if _width(window) >= windowsize * cutoff:
      yield window
    else:
      continue

def sliding_with_filenames(filename, image, windowsize, stride, cutoff= 0.0):
  """Same as sliding, but adding the window number to the filename per window

  Args:
      filename (string): name of the file being processed
      image (2 or 3 dimensional numpy array): the image the sliding windows are created for
      windowsize (int): width of the sliding window
      stride (int): stepsize of the sliding
      cutoff (float): drop windows with width below windowsize * cutoff 

  Returns:
      generator for the sliding windows and their filenames
  """
  for i,window in enumerate(sliding(image, windowsize, stride, cutoff)):
    counter = "_%02d" % i 
    window_filename = "".join([filename, counter])
    yield window_filename, window


def _height(img): return img.shape[0] 
def _width(img): return img.shape[1] 