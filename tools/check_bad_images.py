import numpy as np
from scipy.ndimage import imread
import os
import argparse
import csv

def main(args):
    with open(args.csv_input, "rb") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            image_path, label = row
            check_image(image_path)

def check_image(image_path):
    image = imread(image_path, mode="L")
    if(np.count_nonzero(image-np.mean(image)) == 0):
        print image_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='csv_input', required=True,
                        help='Path to the csv file containing paths and labels of images')
    args = parser.parse_args()
    main(args)
