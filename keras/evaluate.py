import os
import argparse
from yaml import load

import data_loaders
from train import metrics_report
from keras.models import load_model


def evaluate(cli_args):

    config = load(open(cli_args.config, "rb"))

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(config["validation_data_dir"], config)

    # Model Generation
    model = load_model(cli_args.model_dir)
    print(model.summary())

    # Detailed statistics after the training has finished
    probabilities = model.predict_generator(
        data_generator.get_data(should_shuffle=False),
        val_samples=data_generator.get_num_files(),
        nb_worker=2,
        max_q_size=config["batch_size"],
        pickle_safe=True
    )

    y_pred = [np.argmax(prob) for prob in probabilities]
    y_true = data_generator.get_labels()[:len(y_pred)]
    metrics_report(y_true, y_pred)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    cli_args = parser.parse_args()

    evaluate(cli_args)
