import os
import numpy as np
from PIL import Image
from StringIO import StringIO
from glob import glob
from Queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread


class SpectrogramGenerator(object):
    def __init__(self, source, config, shuffle=False, max_size=100):

        self.source = source
        self.config = config
        self.queue = Queue(max_size)
        self.shuffle = shuffle

        self._thread = Thread(target=self._run)

        if os.path.isdir(self.source):
            files = glob(os.path.join(self.source, "*.wav"))
            files.extend(glob(os.path.join(self.source, "*.mp3")))
            files.extend(glob(os.path.join(self.source, "*.m4a")))
        else:
            files = [self.source]

        self.files = files

        # Let's get started
        self._thread.start()

    def audioToSpectrogram(self, file):

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

        command = "sox -V0 '{}' -c 1 -n rate 10k spectrogram -y 129 -X {} -m -r -o -".format(file, self.config[
            "pixel_per_second"])
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)

        output, errors = p.communicate()
        if errors:
            print errors

        image = Image.open(StringIO(output))
        return np.array(image)

    def _run(self):

        start = 0

        while True:

            file = self.files[start]

            try:
                image = self.audioToSpectrogram(file)
                image = np.expand_dims(image, -1)  # add dimension for mono channel

                height, width, channels = image.shape
                target_height, target_width, target_channels = self.config["input_shape"]
                num_segements = width // target_width

                for i in range(0, num_segements):
                    slice_start = i * target_width
                    slice_end = slice_start + target_width

                    slice = image[:, slice_start:slice_end]
                    self.queue.put(slice, block=True)

            except:
                pass

            finally:

                start += 1
                if start >= len(self.files):
                    start = 0

                    if self.shuffle:
                        np.random.shuffle(self.files)

    def __iter__(self):

        for value in iter(self.queue.get):
            yield value

    def next(self):

        return self.queue.get(block=True)

    def stop(self):

        self._thread.join()

    def get_num_files(self):

        return len(self.files)
