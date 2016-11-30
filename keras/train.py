import os
import shutil
from datetime import datetime
from yaml import load

import models
import data_loaders
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from keras.optimizers import Adam

config = load(open("config.yaml", "rb"))


def metrics_report(y_true, y_pred):

    available_labels = range(0, config["num_classes"])

    print(classification_report(y_true, y_pred, labels=available_labels, target_names=config["label_names"]))
    print(confusion_matrix(y_true, y_pred, labels=available_labels))


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
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode="auto")

    # Model Generation
    model_class = getattr(models, config["model"])
    model = model_class.create_model(train_data_generator.get_input_shape(), config)
    print(model.summary())

    optimizer = Adam(lr=config["learning_rate"], decay=1e-6)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", "recall", "precision", "fmeasure"])


    # Training
    history = model.fit_generator(
        train_data_generator.get_data(),
        samples_per_epoch=train_data_generator.get_num_files(),
        nb_epoch=1, # config["num_epochs"],
        callbacks=[model_checkpoint_callback, tensorboard_callback, csv_logger_callback, early_stopping_callback],
        verbose=1,
        validation_data=validation_data_generator.get_data(should_shuffle=False),
        nb_val_samples=validation_data_generator.get_num_files(),
        nb_worker=2,
        max_q_size=config["batch_size"],
        pickle_safe=True
    )

    # Detailed statistics after the training has finished
    predictions = model.predict_generator(
        validation_data_generator.get_data(should_shuffle=False, is_prediction=True),
        val_samples=validation_data_generator.get_num_files(),
        nb_worker=2,
        max_q_size=config["batch_size"],
        pickle_safe=True,
        verbose=1
    )

    y_true = [label for (data, label) in validation_data_generator.get_data(should_shuffle=False)]
    metrics_report(y_true, predictions)


if __name__ == "__main__":
    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Logging to {}".format(log_dir))

    # copy models & config for later
    shutil.copytree("models", log_dir)  # creates the log_dir
    shutil.copy("config.yaml", log_dir)

    train(log_dir)
