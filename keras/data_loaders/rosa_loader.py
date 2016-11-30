import numpy as np
import librosa

from csv_loader import CSVLoader


class RosaLoader(CSVLoader):
    def process_file(self, file_path):

        # mel-spectrogram parameters
        SR = 12000
        N_FFT = 512
        N_MELS = 96
        HOP_LEN = 256

        src, sr = librosa.load(file_path, sr=SR)  # whole signal

        logam = librosa.logamplitude
        melgram = librosa.feature.melspectrogram
        mel_spectrogram = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                                ref_power=1.0)

        mel_spectrogram = np.expand_dims(mel_spectrogram, -1)

        # for 10secs shape (96, 469, 1)
        return mel_spectrogram
