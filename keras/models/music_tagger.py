from keras.applications.music_tagger_crnn import MusicTaggerCRNN
from keras.models import Model
from keras.layers import Dense, Input, Flatten

NAME = "music tagger"

def create_model(input_shape, config):

    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    tagger_model = MusicTaggerCRNN(include_top=False, weights=None, input_tensor=input_tensor)
    print(tagger_model.summary())

    x = tagger_model.output
    x = Flatten()(x)
    predictions = Dense(config["num_classes"], activation='softmax')(x)

    return Model(input=tagger_model.input, output=predictions)