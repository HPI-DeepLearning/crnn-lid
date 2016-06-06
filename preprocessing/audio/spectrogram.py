import scipy.signal
import numpy as np

def _spectrogram(samplerate, signal, window="hamming", winlen=0.02, winstep=0.01, nfft=512):
  nperseg = int(winlen * samplerate)
  noverlap = int(winstep * samplerate)
  f, t, Sxx = scipy.signal.spectrogram(signal, samplerate, window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
  return f, t, 10 * np.log10(Sxx + 1)

def spectrogram_cutoff(samplerate, signal, window="hamming", winlen=0.02, winstep=0.01, nfft=512, lowfreq=100, highfreq=8000):
  f, t, Sxx = _spectrogram(samplerate, signal, window="hamming", winlen=0.02, winstep=0.01, nfft=1024)
  Sxx_cut = Sxx[(f >= lowfreq) & (f <= highfreq), :]
  return np.flipud(Sxx_cut)


