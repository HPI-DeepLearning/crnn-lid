import numpy as np
import scipy.io.wavfile as wav
from csv_loader import CSVLoader
from numpy.lib import stride_tricks

class Spectrogram2Loader(CSVLoader):


    def stft(self, sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
        # cols for windowing
        cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                          strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)


    def logscale_spec(self, spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
        spec = spec[:, 0:128]
        time_bins, freq_bins = np.shape(spec)
        scale = np.linspace(0, 1, freq_bins)  # ** factor

        # Voice Perturbation
        # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
        scale = np.array(
            map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale))
        scale *= (freq_bins - 1) / max(scale)

        newspec = np.complex128(np.zeros([time_bins, freq_bins]))
        all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
        freqs = [0.0] * freq_bins
        totw = [0.0] * freq_bins

        for i in range(0, freq_bins):
            if (i < 1 or i + 1 >= freq_bins):
                newspec[:, i] += spec[:, i]
                freqs[i] += all_freqs[i]
                totw[i] += 1.0
                continue
            else:
                # scale[15] = 17.2
                w_up = scale[i] - np.floor(scale[i])
                w_down = 1 - w_up
                j = int(np.floor(scale[i]))

                newspec[:, j] += w_down * spec[:, i]
                freqs[j] += w_down * all_freqs[i]
                totw[j] += w_down

                newspec[:, j + 1] += w_up * spec[:, i]
                freqs[j + 1] += w_up * all_freqs[i]
                totw[j + 1] += w_up

        for i in range(len(freqs)):
            if (totw[i] > 1e-6):
                freqs[i] /= totw[i]

        return newspec, freqs


    def create_spectrogram(self, file, bin_size=1024, alpha=1):

        sample_rate, samples = wav.read(file)
        s = self.stft(samples, bin_size)

        sshow, freq = self.logscale_spec(s, factor=1, sr=sample_rate, alpha=alpha)
        sshow = sshow[2:, :]
        ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

        ims = np.transpose(ims)
        ims = ims[0:128, :]  # 0-5.5khz

        return np.expand_dims(ims, -1)

    def process_file(self, file_path):

        spectrogram = self.create_spectrogram(file_path)

        # Image shape should be (cols, rows, channels)
        assert len(spectrogram.shape) == 3

        return np.divide(spectrogram, 255.0)  # Normalize images to 0-1.0