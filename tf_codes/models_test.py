import unittest
import tensorflow as tf
import numpy as np
from models import *


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.N_SAMPLE = 32
        self.N_CLASS = 10
        self.SEQ_LEN = 100

        self.x = np.random.randn(self.N_SAMPLE, 257, self.SEQ_LEN, 4).astype(np.float32)
        self.y = np.random.randint(self.N_CLASS,
                                   size=self.N_SAMPLE)
        
    def test_dense_net_based_model(self):
        m = dense_net_based_model(input_shape=self.x.shape[1:],
                                  n_classes=self.N_CLASS,
                                  n_layer_per_block=[8, 8, 4],
                                  growth_rate=4,
                                  activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        m.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

        before = m.evaluate(self.x, self.y, verbose=0)
        m.fit(self.x, self.y, epochs=10, verbose=0)
        after = m.evaluate(self.x, self.y, verbose=0)
        self.assertTrue(before > after)

    def test_squeeze_excitation(self):
        se = SE(reduction=4, axis=1)(self.x)
        self.assertEqual(se.shape, self.x.shape)

    def test_conv_lstm_model(self):
        x = np.expand_dims(self.x.transpose([0, 2, 1, 3]), axis=3)
        m = conv_lstm_model(input_shape=x.shape[1:],
                            n_classes=self.N_CLASS,
                            n_layer_per_block=[8, 8, 4],
                            growth_rate=4,
                            activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        m.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
        m.summary()

        before = m.evaluate(x, self.y, verbose=0)
        m.fit(x, self.y, epochs=10, verbose=0)
        after = m.evaluate(x, self.y, verbose=0)
        self.assertTrue(before > after)



if __name__ == '__main__':
    unittest.main()
