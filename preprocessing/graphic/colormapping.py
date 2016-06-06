import matplotlib.cm
import numpy as np
import cv2 as cv

def to_rgba(signal, colormap="gray", norm=None, bytes=False):
  """converts a two dimensional numpy array to an rgba array

  Args:
      signal (2 dimensional numpy array): the data the colormap should be applied to
      colormap (str): the colormap to use
      norm (matplotlib.colors.Normalize instance): the norm used to project data to interval [0,1]
      bytes: (bool): if False output range will be [0,1], if True uint8 from 0 to 255

  Returns:
      image (3 dimensional numpy array): rgba array

  """
  colormapper = matplotlib.cm.ScalarMappable(norm, colormap)
  image = colormapper.to_rgba(signal, bytes=bytes)
  return image

def to_rgb(signal, *args, **kwargs):
  """converts a two dimensional numpy array to an rgb array

  Args:
      signal (2 dimensional numpy array): the data the colormap should be applied to
      *args: piped through to to_rgba
      **kwargs: pipe through to to_rgba

  Returns:
      image (3 dimensional numpy array): rgb array
  """
  image = to_rgba(signal, *args, **kwargs)
  #drop alpha channel
  return image[:,:,:-1]

def to_grayscale(signal, *args, **kwargs):
  """converts a two dimensional numpy array to an grayscale array

  Args:
      signal (2 dimensional numpy array): the data the colormap should be applied to
      *args: piped through to to_rgb
      **kwargs: pipe through to to_rgb

  Returns:
      image (2 dimensional numpy array): grayscale array
  """
  image = to_rgb(signal, *args, **kwargs)
  result = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  return result