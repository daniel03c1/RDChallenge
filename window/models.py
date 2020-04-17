import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
import keras.backend as K


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


def lstm(input_shape, **kwargs):
    model_input = Input(shape=input_shape)
    x = model_input

    for i in range(2):
        x = LSTM(128, dropout=0.2, return_sequences=True)(x)

    for i in range(2):
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    x = K.squeeze(x, axis=-1)

    return tf.keras.Model(inputs=model_input, outputs=x)


def multihead_attention(d_attn, n_heads, div=1, **kwargs):
    def _multihead(x):
        heads = [
            Attention(use_scale=True)([
                Dense(int(d_attn/div), **kwargs)(x),
                Dense(int(d_attn/div), **kwargs)(x),
                Dense(int(d_attn/div), **kwargs)(x)])
            for i in range(n_heads)]
        x = Concatenate()(heads)
        x = Dense(d_attn, **kwargs)(x)
        return x
    return _multihead


def attention_layer(d_attn, n_heads, dropout_rate=0.1, div=1, mul=4, **kwargs):
    def _attention(x):
        # Attention
        y = multihead_attention(d_attn, n_heads, div=div, **kwargs)(x)
        y = Dropout(dropout_rate)(y)
        x = LayerNormalization()(x + y)

        # Feed-Forward Networks
        y = Dense(int(d_attn * mul), activation='relu', **kwargs)(x)
        y = Dense(d_attn, **kwargs)(y)
        y = Dropout(dropout_rate)(y)
        x = LayerNormalization()(x + y)
        return x
    return _attention


def multi_stream(input_shape, n_blocks=2, n_streams=5, **kwargs):
    model_input = Input(shape=input_shape) # (time, freq)
    x = model_input

    d_model = 128
    nh = 15
    dq, dk, dv = 40, 40, 80

    for i in range(n_blocks):
        streams = []

        for j in range(n_streams):
            stream = Conv1D(d_model, 3, padding='same', dilation_rate=i+1)(x)
            stream = attention_layer(
                d_model, nh//n_streams, div=d_model/dq, mul=0.5)(stream)
            streams.append(stream)

        x = Concatenate()(streams)
        x = Dense(d_model, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

    x = Dense(d_model, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    x = K.squeeze(x, axis=-1)

    return tf.keras.Model(inputs=model_input, outputs=x)


def st_attention(input_shape, **kwargs):
    model_input = Input(shape=input_shape) # (time, freq)
    x = model_input

    # spectral attention
    x = K.expand_dims(x, -1) # for conv
    
    n_chan = 16
    for i in range(4):
        conv1 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Activation('sigmoid')(conv2)

        x = conv1 * conv2
        x = MaxPool2D(pool_size=(1, 2))(x) # max pool 1d (freq)
        n_chan *= 2

    x = Reshape((input_shape[0], x.shape[-1] * x.shape[-2]))(x) # (time, -1)

    # Pipe-Net
    for i in range(2):
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

    # out_pipe = Dense(1, activation='sigmoid')(x)
    # out_pipe = K.squeeze(out_pipe, -1)

    # Temporal Attention
    n_heads, head_size = 4, 32
    g = K.mean(x, axis=1) # (batch, 256)
    query = Dense(128, use_bias=False, activation='sigmoid')(g) # (batch, 128)
    Key = Dense(128, use_bias=False, activation='sigmoid')(x) # (batch, l, 128)
    Value = Dense(128, use_bias=False, activation='sigmoid')(x)
    
    heads = []
    for i in range(4): # n_heads = 4
        q = query[:, head_size*i:head_size*(i+1)] # (batch, 32)
        k = Key[:, :, head_size*i:head_size*(i+1)] # (batch, l, 32)
        k = K.permute_dimensions(k, (0, 2, 1)) # (batch, 32, l)
        head = K.batch_dot(q, k) / np.sqrt(32)
        head = Activation('softmax')(head)
        head = K.expand_dims(head, axis=-1)
        head *= Value[:, :, head_size*i:head_size*(i+1)]
        heads.append(head)

    x = Concatenate()(heads)

    # Key = K.permute_dimensions(Key, (0, 2, 1)) # (batch, 128, l)
    # out_attn = K.batch_dot(query, Key) / np.sqrt(128)
    # out_attn = Activation('softmax')(out_attn)

    # Post-Net
    for i in range(2):
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)
    x = K.squeeze(x, axis=-1)
    
    return tf.keras.Model(inputs=model_input, outputs=x)


def test_model(input_shape, **kwargs):
    model_input = Input(shape=input_shape) # (time, freq)
    x = model_input 
    x = K.expand_dims(x, -1) # for conv
   
    n_chan = 16
    for _ in range(4):
        conv1 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Activation('sigmoid')(conv2)

        x = conv1 * conv2
        x = MaxPool2D(pool_size=(1, 2))(x) # max pool 1d (freq)
        n_chan *= 2

    x = Reshape((input_shape[0], x.shape[-2] * x.shape[-1]))(x) # (time, -1)
    x = Dense(128, **kwargs)(x)

    for _ in range(2):
        x = attention_layer(128, 8, 0.2, div=4, mul=0.5, **kwargs)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid')(x)
    x = K.squeeze(x, axis=-1)
    
    return tf.keras.Model(inputs=model_input, outputs=x)


def test2(input_shape, **kwargs):
    model_input = Input(shape=input_shape) # (time, freq)
    x = model_input 
    x = BatchNormalization()(x)
    x = K.expand_dims(x, -1) # for conv

    n_chan = 24
    for _ in range(3):
        conv1 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Conv2D(filters=n_chan, kernel_size=3, padding='same')(x)
        conv2 = Activation('sigmoid')(conv2)

        x = conv1 * conv2
        x = MaxPool2D(pool_size=(1, 2))(x) # max pool 1d (freq)
        x = BatchNormalization()(x)
        n_chan = int(n_chan * np.sqrt(2))

    # (t, 5)
    size = x.shape[1:-1]
    x1 = Conv2D(filters=n_chan//3, kernel_size=1)(x)
    x2 = AveragePooling2D(pool_size=3, strides=(1, 2), padding='valid')(x)
    x2 = Conv2D(filters=n_chan//3, kernel_size=3)(x2)
    x2 = Lambda(lambda x: tf.image.resize(x, size=size))(x2)
    x3 = AveragePooling2D(pool_size=(input_shape[0], 5), padding='valid')(x)
    x3 = Conv2D(filters=n_chan//3, kernel_size=1)(x3)
    x3 = Lambda(lambda x: tf.image.resize(x, size=size))(x3)
    x = Concatenate()([x, x1, x2, x3])
    x = BatchNormalization()(x)

    x = Conv2D(filters=n_chan, kernel_size=3)(x)
    x = AveragePooling2D(pool_size=3, strides=2, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(input_shape[0], activation='sigmoid')(x)

    return tf.keras.Model(inputs=model_input, outputs=x)

