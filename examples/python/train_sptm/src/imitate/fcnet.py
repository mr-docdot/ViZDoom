import six
from keras.models import Model
from keras.layers import (
    Activation,
    Dense,
    Flatten,
    Input
)
from keras.layers.core import Lambda
from keras.layers.merge import (dot, concatenate)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def FCNet(input_shape, action_size):
    # Create embedding from goal vector using 3 layer FC network
    ga_input = Input(shape=input_shape)
    ga_fc1 = Dense(units=32, kernel_initializer='he_normal',
                   activation='relu')(ga_input)
    ga_fc2 = Dense(units=128, kernel_initializer='he_normal',
                   activation='relu')(ga_fc1)
    ga_embed = Dense(units=512, kernel_initializer='he_normal',
                     activation='relu')(ga_fc2)

    # Classifier block
    dense = Dense(units=action_size, kernel_initializer="he_normal",
                  activation="softmax")(ga_embed)

    model = Model(inputs=ga_input, outputs=dense)
    return model
