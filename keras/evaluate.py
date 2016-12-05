import argparse
import numpy as np
from yaml import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import load_model

import data_loaders

def metrics_report(y_true, y_pred, label_names=None):

    available_labels = range(0, len(label_names))

    print("Accuracy %s" % accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    print(confusion_matrix(y_true, y_pred, labels=available_labels))


def evaluate(cli_args):

    config = load(open(cli_args.config, "rb"))

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(config["validation_data_dir"], config)

    # Model Generation
    model = load_model(cli_args.model_dir)

    probabilities = model.predict_generator(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        val_samples=data_generator.get_num_files(),
        nb_worker=1,  # parallelization messes up data order. careful!
        max_q_size=config["batch_size"],
        pickle_safe=True
    )

    y_pred = np.argmax(probabilities, axis=1)
    y_true = data_generator.get_labels()[:len(y_pred)]
    metrics_report(y_true, y_pred , label_names=config["label_names"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    cli_args = parser.parse_args()

    evaluate(cli_args)
