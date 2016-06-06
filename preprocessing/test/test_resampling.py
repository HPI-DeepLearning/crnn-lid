from audio import resample
import numpy as np
import unittest
import math
import scipy.io.wavfile as wav

class ResamplingTest(unittest.TestCase):
 
  def test_downsampling(self):
    samplerate = 22100
    length = np.random.randint(0,5) + round(np.random.random(), 2)
    num_samples = samplerate * length
    signal = np.random.randint(0,256, num_samples)
    
    target_samplerate = 16000
    target_num_samples = math.ceil(target_samplerate * length)

    resampled_signal, resampled_samplerate = resample.downsample(signal, samplerate, target_samplerate)

    self.assertEqual(resampled_samplerate, target_samplerate)

    #print "%d == %d?" %(target_num_samples, len(resampled_signal))
    #self.assertEqual(len(resampled_signal), target_num_samples)
    
  def test_upsampling(self):
    samplerate = 8000
    length = np.random.randint(0,5) + round(np.random.random(), 3)
    num_samples = samplerate * length
    signal = np.random.randint(0,256, num_samples)
    
    target_samplerate = 16000
    target_num_samples = target_samplerate * length

    self.assertRaises(ValueError, resample.downsample, signal, samplerate, target_samplerate)
