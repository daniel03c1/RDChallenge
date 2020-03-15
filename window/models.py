import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
import keras.backend as K


def suggestion(input_shape,
               n_layer_per_block=[4, 6, 10, 6],
               growth_rate=12,
               **kwargs):
    """
    :param input_shape: LIST of length 3
        [height, width, chan]
    :param n_layer_per_block: LIST of length (n_block)
        each integer indicates the number of layer per block
    :param growth_rate: INT
        the growth rate of dense block

    :return: Keras Model
    """
    model_input = Input(shape=input_shape)
    n_block = len(n_layer_per_block)

    # FRONT
    x = BatchNormalization()(model_input)
    x = K.expand_dims(x, axis=-1)

    x = Conv2D(filters=growth_rate*2,
               kernel_size=[7, 7],
               padding='same',
               **kwargs)(x)
    x = AveragePooling2D((3, 3), strides=(1, 2), padding='same')(x)

    # MIDDLE
    for i in range(n_block - 1):
        x = dense_block(n_layer=n_layer_per_block[i],
                        growth_rate=growth_rate,
                        **kwargs)(x)
        x = dense_net_conv2d(filters=growth_rate * 4,
                             kernel_size=[1, 1],
                             padding='same',
                             **kwargs)(x)
        x = AveragePooling2D(strides=(1, 2), padding='same')(x)

    # FINAL
    x = dense_block(n_layer=n_layer_per_block[-1],
                    growth_rate=growth_rate,
                    **kwargs)(x)
    print(x.shape)
    _, time, bins, chan = x.shape

    x = AveragePooling2D((1, bins), padding='same')(x)
    x = K.squeeze(x, axis=2)
    x = TimeDistributed(Dense(growth_rate*16, activation='relu'))(x)
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    x = K.squeeze(x, axis=2)

    return tf.keras.Model(inputs=model_input, outputs=x)


def dense_net_conv2d(*args, **kwargs):
    def _dense_net_conv2d(tensor):
        x = BatchNormalization()(tensor)
        x = ReLU()(x)
        x = Conv2D(*args, **kwargs)(x)
        return x
    return _dense_net_conv2d


def dense_block(n_layer, growth_rate, se=False, *args, **kwargs):
    def _dense_block(tensor):
        x = tensor

        for i in range(n_layer):
            y = Dropout(0.2)(x)
            y = dense_net_conv2d(filters=growth_rate,
                                 kernel_size=[1, 1],
                                 padding='same',
                                 *args, **kwargs)(y)

            y = dense_net_conv2d(filters=growth_rate,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 *args, **kwargs)(y)
            x = Concatenate(axis=-1)([x, y])

        if se:
            x = SE(reduction=4, axis=3, *args, **kwargs)(x)

        return x
    return _dense_block


def SE(reduction=2, axis=1, *args, **kwargs):
    def _se(tensor):
        assert len(tensor.shape) == 4
        x = K.stop_gradient(tensor)
        reduce_axis = tuple([i for i in range(1, 4) if i != axis])
        mean = Lambda(lambda x: K.mean(x, axis=reduce_axis))(x)
        std = Lambda(lambda x: K.std(x, axis=reduce_axis))(x)
        squeeze = Concatenate(axis=-1)([mean, std])

        excitation = Dense(tensor.shape[axis] // reduction,
                           activation='relu', *args, **kwargs)(squeeze)
        excitation = Dense(tensor.shape[axis],
                           activation='tanh', *args, **kwargs)(excitation)
        excitation = Reshape(
            [1 if i != axis else tensor.shape[i] for i in range(1, 4)])(excitation)

        return Multiply()([tensor, excitation])
    return _se


def dnn(input_shape, dropout_rate=0.5, **kwargs):
    model_input = Input(shape=input_shape)

    x = Flatten()(model_input)
    for i in range(2):
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=model_input, outputs=x)


def bdnn(input_shape, dropout_rate=0.5, **kwargs):
    model_input = Input(shape=input_shape)

    x = Flatten()(model_input)
    for i in range(2):
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(input_shape[0], activation='sigmoid')(x)
    return tf.keras.Model(inputs=model_input, outputs=x)



