import os
import random
import numpy as np
from PIL import Image
import fnmatch
from math import ceil
import sox
import tempfile
import shutil
import sys

from pydub import AudioSegment

if (sys.version_info >= (3,0)):
    from queue import Queue
else:
    from Queue import Queue

NOISE_FILES_LENGTH = [118, 14, 20, 46, 126, 8, 124]

def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename

class VinylBackgroundSpectrogramGenerator(object):
    def __init__(self, source, config, shuffle=False, max_size=100, run_only_once=False):

        self.source = source
        self.config = config
        self.queue = Queue(max_size)
        self.shuffle = shuffle
        self.run_only_once = run_only_once

        if os.path.isdir(self.source):
            files = []
            files.extend(recursive_glob(self.source, "*.wav"))
            files.extend(recursive_glob(self.source, "*.mp3"))
            files.extend(recursive_glob(self.source, "*.m4a"))
        else:
            files = [self.source]

        self.files = files


    def audioToSpectrogram(self, file, pixel_per_sec, height):

        noise_file_index = random.randint(1, len(NOISE_FILES_LENGTH))
        noise_file_name = "vinyl_noise/normalized-noise{}.wav".format(noise_file_index)

        with tempfile.NamedTemporaryFile(suffix='.wav') as noisy_speech_file:

            noise = AudioSegment.from_file(noise_file_name)
            speech = AudioSegment.from_file(file)

            speech.apply_gain(noise.dBFS - speech.dBFS)

            noisy_speech = speech.overlay(noise - 10, loop=True)
            noisy_speech.export(noisy_speech_file.name, format="wav")

            # shutil.copyfile(noisy_speech_file.name, os.path.join("/extra/tom/news2/debug", "mixed_" + os.path.basename(noisy_speech_file.name)))

            with tempfile.NamedTemporaryFile(suffix='.png') as image_file:
                command = "{} -n remix 1 rate 10k spectrogram -y {} -X {} -m -r -o {}". format(noisy_speech_file.name, height, pixel_per_sec, image_file.name)
                sox.core.sox([command])

                # spectrogram can be inspected at image_file.name
                image = Image.open(image_file.name)

                return np.array(image)

    def get_generator(self):

        start = 0

        while True:

            file = self.files[start]

            try:

                target_height, target_width, target_channels = self.config["input_shape"]

                image = self.audioToSpectrogram(file, self.config["pixel_per_second"], target_height)
                image = np.expand_dims(image, -1)  # add dimension for mono channel

                height, width, channels = image.shape

                assert target_height == height, "Heigh mismatch {} vs {}".format(target_height, height)

                num_segments = width // target_width

                for i in range(0, num_segments):
                    slice_start = i * target_width
                    slice_end = slice_start + target_width

                    slice = image[:, slice_start:slice_end]

                    # Ignore black images
                    if slice.max() == 0 and slice.min() == 0:
                        continue

                    yield slice

            except Exception as e:
                print("VinylBackgroundSpectrogramGenerator Exception: ", e, file)
                pass

            finally:

                start += 1
                if start >= len(self.files):

                    if self.run_only_once:
                        break

                    start = 0

                    if self.shuffle:
                        np.random.shuffle(self.files)


    def get_num_files(self):

        return len(self.files)


if __name__ == "__main__":

    a = VinylBackgroundSpectrogramGenerator("/Users/therold/Uni/master-thesis/datasets/EUSpeech/english/input.wav", {"pixel_per_second": 50, "input_shape": [129, 100, 1], "batch_size": 32, "num_classes": 4}, shuffle=True)
    gen = a.get_generator()


    for a in gen:
        pass