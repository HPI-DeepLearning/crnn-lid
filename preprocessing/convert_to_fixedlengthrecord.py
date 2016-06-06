import argparse
from scipy.ndimage import imread
import numpy as np
import os


def read_csv(csv_input, chunksize=10000):
    with open(csv_input, "rb") as csvfile:
        lines = csvfile.readlines()
        num_records = len(lines)
        print num_records
        for i in range(0, num_records, chunksize):
            print i
            print len(lines)
            result = [line.strip().split(",") for line in lines[i:i+chunksize]]
            yield result


def read_image(path):
    return imread(path, mode="L")


def main(csv_input, output_path, prefix):
    for i, chunk in enumerate(read_csv(csv_input)):
        file_name = "%s_%03d.bin" % (prefix, i)
        print len(chunk)
        with open(os.path.join(output_path, file_name), "wb") as output:
            for image_path, label in chunk:
                image = read_image(image_path)
                output.write(np.uint8(label).tobytes())
                output.write(np.uint8(image).tobytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='csv_input', required=True,
                        help='Path to the csv file containing paths and labels of images')
    parser.add_argument('--outputPath', dest='output_path',
                        required=True, help='Path to the output dir')
    parser.add_argument('--prefix', dest='prefix', required=True,
                        help='Prefix of the fixed length record files')
    args = parser.parse_args()
    main(args.csv_input, args.output_path, args.prefix)
