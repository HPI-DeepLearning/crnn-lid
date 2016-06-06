import os, sys
dirname = os.path.dirname(__file__)
lib_path = os.path.join(dirname, "python_speech_features")
sys.path.append(lib_path)

import features as speechfeatures
import numpy as np

def filter(samplerate, signal, winlen=0.02, winstep=0.01,
            nfilt=40, nfft=512, lowfreq=100, highfreq=5000, preemph=0.97):
  """extracts mel filterbank energies from a given signal

  Args:
    samplerate (int): samples taken per second
    signal(1d numpy array): sample values
    winlen(float): sliding window size in seconds
    winstep(float): overlap of sliding windows in seconds
    nfilt(int): number of mel filters to apply
    nfft(int): size of the discrete fourier transform to use
    lowfreq(int): lowest frequency to collect
    highfreq(int): highest frequency to collect
    preemph(float): preemphesis factor

  Returns:
    feat(2d numpy array): filterbank energies

  """
  feat, energy = speechfeatures.fbank(np.array(signal), samplerate, winlen=winlen,winstep=winstep,
            nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

  return np.swapaxes(feat,0,1)


def logfilter(samplerate, signal, *args, **kwargs):
  """extracts log mel filterbank energies from a given signal

  Args:
    samplerate (int): samples taken per second
    signal(1d numpy array): sample values
    *args: piped through to filter
    **kwargs: piped thought to filter

  Returns:
    feat(2d numpy array): logarithm of filterbank energies

  """
  feat = filter(samplerate, signal, *args, **kwargs)
  return np.log(feat)