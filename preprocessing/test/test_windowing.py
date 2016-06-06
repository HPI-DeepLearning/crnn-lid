from graphic import windowing
import numpy as np
import unittest
import math

class WindowingTest(unittest.TestCase):
 
  def test_windowing(self):
    image_height = 40
    image_width = 300
    image_channels = 3
    image = np.random.random((image_height, image_width, image_channels))
    windowsize = 400
    stride = windowsize / 2

    windows = windowing.sliding(image, windowsize, stride)
    self.assertTrue(all([window.shape[1] <= windowsize for window in windows]))
    padded_windows = [windowing.pad_window(window, window_size) for window in windows]
    self.assertTrue(all([window.shape[1] == windowsize for window in windows]))
    self.assertTrue(all([window.shape[2] == image_channels for window in windows]))

  def test_stride_properly(self):
    image = np.random.random((40, 300, 3))
    windowsize = 200
    stride = windowsize / 2

    expected_number_of_windows = int(math.ceil(image.shape[1] / float(stride)))

    self.assertEqual(expected_number_of_windows, len(list(windowing.sliding(image, windowsize, stride))))
