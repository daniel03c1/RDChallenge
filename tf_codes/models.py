import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
import keras.backend as K


def dense_net_based_model(input_shape,
                          n_classes,
                          n_layer_per_block,
                          growth_rate,
                          normalization='batchnorm',
                          activation='softmax',
                          se=False,
                          *args, **kwargs):
    """
    :param input_shape: LIST of length 3
        [height, width, chan]
    :param n_classes: INT
        the number of output classes
    :param n_layer_per_block: LIST of length (n_block)
        each integer indicates the number of layer per block
    :param growth_rate: INT
        the growth rate of dense block
    :param normalization: ('layernorm', 'batchnorm', 'none')
    :param activation: STR
        activation function of the final layer

    :return: Keras Model
    """
    model_input = Input(shape=input_shape)
    n_block = len(n_layer_per_block)

    # FRONT
    if normalization == 'batchnorm':
        x = BatchNormalization()(model_input)
    elif normalization == 'layernorm':
        x = LayerNormalization(center=False, scale=False)(model_input)
    else:
        x = model_input

    x = Conv2D(filters=growth_rate*2,
               kernel_size=[7, 7],
               padding='same',
               *args, **kwargs)(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # MIDDLE
    for i in range(n_block - 1):
        x = dense_block(n_layer=n_layer_per_block[i],
                        growth_rate=growth_rate,
                        se=se,
                        *args, **kwargs)(x)
        x = dense_net_conv2d(filters=growth_rate * 4,
                             kernel_size=[1, 1],
                             padding='same',
                             *args, **kwargs)(x)
        x = AveragePooling2D()(x)

    # FINAL
    x = dense_block(n_layer=n_layer_per_block[-1],
                    growth_rate=growth_rate,
                    se=se,
                    *args, **kwargs)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=growth_rate*16, activation='relu', *args, **kwargs)(x)
    x = Dense(units=n_classes, activation=activation)(x)

    model = tf.keras.Model(inputs=model_input, outputs=x)
    return model


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


def model_two(input_shape,
              n_classes,
              n_layer_per_block,
              growth_rate,
              *args, **kwargs):
    """
    :param input_shape: LIST of length 3
        [height, width, chan]
    :param n_classes: INT
        the number of output classes
    :param n_layer_per_block: LIST of length (n_block)
        each integer indicates the number of layer per block
    :param growth_rate: INT
        the growth rate of dense block
    :param normalization: ('layernorm', 'batchnorm', 'none')
    :param activation: STR
        activation function of the final layer

    :return: Keras Model
    """
    model_input = Input(shape=input_shape)
    n_block = len(n_layer_per_block)

    # FRONT
    x = BatchNormalization()(model_input)

    x = Conv2D(filters=growth_rate*2,
               kernel_size=[7, 7],
               padding='same',
               *args, **kwargs)(x)
    x = AveragePooling2D((3, 3), strides=(2, 1), padding='same')(x)

    # MIDDLE
    for i in range(n_block - 1):
        x = dense_block(n_layer=n_layer_per_block[i],
                        growth_rate=growth_rate,
                        *args, **kwargs)(x)
        x = dense_net_conv2d(filters=growth_rate * 4,
                             kernel_size=[1, 1],
                             padding='same',
                             *args, **kwargs)(x)
        x = AveragePooling2D(strides=(2,1), padding='same')(x)

    # FINAL
    x = dense_block(n_layer=n_layer_per_block[-1],
                    growth_rate=growth_rate,
                    *args, **kwargs)(x)
    print(x.shape)
    _, bins, time, chan = x.shape

    x = AveragePooling2D((bins, 1), padding='same')(x)
    x = K.squeeze(x, axis=1)
    x = TimeDistributed(Dense(growth_rate*16, activation='relu'))(x)
    x = TimeDistributed(Dense(n_classes, activation='softmax'))(x)

    model = tf.keras.Model(inputs=model_input, outputs=x)
    return model


