import numpy as np
import os
import sys
import csv
# import librosa
# from tensorflow.contrib import ffmpeg

lib_dir = os.path.join(os.path.abspath(__file__), "..", "preprocessing")
sys.path.append(lib_dir)

from preprocessing.preprocessing_commons import apply_melfilter, generate_spectrograms, read_wav_dirty, sliding_audio, downsample
from preprocessing import audio
from preprocessing import graphic

class BatchSoundLoader():


    def __init__(self, csv_path, config):

        self.batch_size = config["batch_size"]
        self.start = 0
        self.end = 0
        self.filenames_labels = csv.reader(open(csv_path)).readlines()

        self.sample_size = len(self.filenames_labels)
        self.sequence_length = 512


    def wav_to_spectrogram(self, sound_file):
        # filenames of the generated images
        window_size = 600  # MFCC sliding window

        f, signal, samplerate = read_wav_dirty(sound_file)
        # signal, samplerate = librosa.core.load(sound_file[0])
        filename = os.path.basename(sound_file)
        # segments = sliding_audio(f, signal, samplerate)

        _, mel_image = apply_melfilter(filename, signal, samplerate)
        mel_image = graphic.colormapping.to_grayscale(mel_image, bytes=True)
        mel_image = graphic.histeq.histeq(mel_image)
        mel_image = graphic.histeq.clamp_and_equalize(mel_image)


        image = graphic.windowing.cut_or_pad_window(mel_image, window_size)

        return [image]

    def to_sparse_labels(self, labels, sequence_lengths):

        indices = []
        vals = []
        for i, label in enumerate(labels):
            for seq_i in np.repeat(label, sequence_lengths[i]):
                indices.append([i, seq_i])
                vals.append(label)

        shape = [len(labels), np.asarray(indices).max(0)[1] + 1]

        return (np.array(indices), np.array(vals), np.array(shape))


    def next_batch(self):

        start = self.start
        end = self.start + self.batch_size

        sound_files, labels = self.filenames_labels[start:end]

        images = [self.wav_to_spectrogram(sound) for sound in sound_files]
        sparse_labels = self.to_sparse_labels(labels)


        return images, sparse_labels






