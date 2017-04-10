import argparse
import numpy as np
from yaml import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from data_loaders import SpectrogramGenerator

def predict(cli_args):

    config = {"pixel_per_second": 50, "input_shape": [129, 100, 1], "num_classes": 4}
    data_generator = SpectrogramGenerator(cli_args.input_file, config)

    # Model Generation
    model = load_model(cli_args.model_dir)

    probabilities = model.predict_generator(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        nb_worker=1,  # parallelization messes up data order. careful!
        pickle_safe=True
    )

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=1)

    print classes, average_prob
    return probabilities

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--input', dest='input_file', required=True)
    cli_args = parser.parse_args()

    predict(cli_args)
