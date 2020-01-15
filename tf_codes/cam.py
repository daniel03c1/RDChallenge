import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt


def apply_cam(model, global_pool_layer_index):
    total_layers = len(model.layers)
    global_pool_layer_index %= total_layers
    x = model.layers[global_pool_layer_index-1].output

    for i in range(global_pool_layer_index+1, total_layers-1):
        x = model.layers[i](x)
    x = K.dot(x, model.layers[-1].weights[0])
    x = K.bias_add(x, model.layers[-1].weights[1])
    return tf.keras.models.Model(inputs=model.input, outputs=x)


def show_cam(original, output, title=None):
    original = np.repeat(original[:, :, :2], 2, axis=2)
    fig, axes = plt.subplots(2, 2)

    # 1. Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('original')

    axes[0, 1].imshow(original)
    axes[0, 1].imshow(output[:, :, -1], alpha=0.4)
    axes[0, 1].set_title('NO VOICE')

    axes[1, 0].imshow(original)
    axes[1, 0].imshow(output.max(axis=-1), alpha=0.4)
    axes[1, 0].set_title('TOTAL(max)')

    axes[1, 1].imshow(original)
    axes[1, 1].imshow(output[:, :, :-1].max(axis=-1), alpha=0.4)
    axes[1, 1].set_title('VOICE(max)')

    fig.suptitle(title)
    plt.show()


if __name__ == '__main__':
    model = tf.keras.models.load_model('base.h5', compile=False)
    cam = apply_cam(model, 188)

    x = np.load('test_x.npy')
    y = np.load('test_y.npy')

    x[:, :, :, (0, 1)] = np.log(x[:, :, :, (0, 1)] + 1e-8)
    # x -= x.min(axis=(1, 2), keepdims=True)
    # x /= x.max(axis=(1, 2), keepdims=True) + 1e-8

    pred_label = np.argmax(model.predict(x), axis=1) != 10
    y = y >= 0
    pred = cam.predict(x)
    pred = tf.image.resize(pred, size=x.shape[1:3]).numpy()

    print(y)
    print(pred_label)

    for i in (359,): # reversed(range(len(y))):
        # if pred_label[i]:
        #     continue
        # if y[i] == pred_label[i]:
        #     continue
        show_cam(x[i], pred[i], '({}) true: {}, pred: {}'.format(i+2, y[i], pred_label[i]))
