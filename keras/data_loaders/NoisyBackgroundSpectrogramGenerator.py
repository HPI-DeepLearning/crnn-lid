import os
import random
import numpy as np
from PIL import Image
import fnmatch
import sys
from subprocess import Popen, PIPE, STDOUT, check_output, call

if (sys.version_info >= (3,0)):
    from queue import Queue
else:
    from Queue import Queue

def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename

class NoisyBackgroundSpectrogramGenerator(object):
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

        '''
        V0 - Verbosity level: ignore everything
        c 1 - channel 1 / mono
        n - apply filter/effect
        rate 10k - limit sampling rate to 10k --> max frequency 5kHz (Shenon Nquist Theorem)
        y - small y: defines height
        X capital X: defines pixels per second
        m - monochrom
        r - no legend
        o - output to stdout (-)
        '''
        file_name = "tmp_{}.png".format(random.randint(0, 100000))
        noise_file_name = "noise_{}.wav".format(random.randint(0, 100000))

        scale_factor1 = 0.94
        scale_factor2 = 0.05

        get_volume_command = "sox \"{}\" -n stat -v 2>&1 | bc -l".format(file)
        input_volume = check_output(get_volume_command, shell=True)

        create_white_noise_command = "sox \"{}\" {} synth whitenoise".format(file, noise_file_name)
        call(create_white_noise_command, shell=True)

        command = "sox -m -v {1} \"{0}\" -v {2} {3} -n remix 1 rate 10k spectrogram -y {4} -X {5} -m -r -o {6}". format(file, scale_factor1 * float(input_volume), scale_factor2, noise_file_name, height, pixel_per_sec, file_name)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)

        output, errors = p.communicate()
        if errors:
            print(errors)

        # image = Image.open(StringIO(output))
        image = Image.open(file_name)
        os.remove(file_name)
        os.remove(noise_file_name)

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
                print("SpectrogramGenerator Exception: ", e, file)
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

    a = SpectrogramGenerator("/Users/therold/Uni/master-thesis/datasets/EU Speech/english/input.wav", {"pixel_per_second": 50, "input_shape": [129, 100, 1], "batch_size": 32, "num_classes": 4}, shuffle=True)
    gen = a.get_generator()


    for a in gen:
        pass