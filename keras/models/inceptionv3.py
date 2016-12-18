from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D

NAME = "InceptionV3"

def create_model(input_shape, config):

    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    inception_model = InceptionV3(include_top=False, weights=None, input_tensor=input_tensor)
    print(inception_model.summary())

    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(config["num_classes"], activation='softmax')(x)

    return Model(input=inception_model.input, output=predictions)