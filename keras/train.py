import os
import shutil
from datetime import datetime
from yaml import load

import models
import data_loaders

# from evaluate import evaluation_metrics

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, BaseLogger, ProgbarLogger
from keras.optimizers import Adam

config = load(open("config.yaml", "rb"))


def train(log_dir):
    if config is None:
        print("Please provide a config.")

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])

    train_data_generator = DataLoader(config["train_data_dir"], config)
    validation_data_generator = DataLoader(config["validation_data_dir"], config)

    # Training Callbacks
    checkpoint_filename = os.path.join(log_dir, "weights.{epoch:02d}.model")
    model_checkpoint_callback = ModelCheckpoint(checkpoint_filename)

    tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
    csv_logger_callback = CSVLogger(os.path.join(log_dir, "log.csv"))

    # Model Generation
    model_class = getattr(models, config["model"])
    model = model_class.create_model(train_data_generator.get_input_shape(), config)  #

    optimizer = Adam(lr=config["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", "recall", "precision", "fmeasure"])

    # Training
    history = model.fit_generator(
        train_data_generator.get_data(),
        samples_per_epoch=train_data_generator.get_num_files(),
        nb_epoch=config["num_epochs"],
        callbacks=[BaseLogger(), ProgbarLogger(), model_checkpoint_callback, tensorboard_callback, csv_logger_callback],
        verbose=2,
        validation_data=validation_data_generator.get_data(),
        nb_val_samples=validation_data_generator.get_num_files(),
    )


if __name__ == "__main__":
    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Logging to {}".format(log_dir))

    # copy models & config for later
    shutil.copytree("models", log_dir)  # creates the log_dir
    shutil.copy("config.yaml", log_dir)

    train(log_dir)
