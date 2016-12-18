from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.regularizers import l2

NAME = "topcoder_5s_finetune"

def create_model(input_shape, config, is_training=True):

    weight_decay = 0.001

    model = Sequential()

    model.add(Convolution2D(16, 7, 7, W_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(32, 5, 5, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(weight_decay), activation="relu"))

    model.add(Dense(config["num_classes"], activation="softmax"))

    ref_model = load_model("logs/2016-12-08-15-14-06/weights.20.model")
    for ref_layer in ref_model.layers[:-2]:
        layer = model.get_layer(ref_layer.name)
        if layer:
            print ref_layer.name
            layer.set_weights(ref_layer.get_weights())
            layer.trainable = False

    return model
