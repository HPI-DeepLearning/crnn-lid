from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D

NAME = "xception"

def create_model(input_shape, config):

    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    xception_model = Xception(include_top=False, weights=None, input_tensor=input_tensor)
    print(xception_model.summary())

    x = xception_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(config["num_classes"], activation='softmax')(x)

    return Model(input=xception_model.input, output=predictions)