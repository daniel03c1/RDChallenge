import unittest
import tensorflow as tf
import numpy as np
from utils import *


class UtilsTest(unittest.TestCase):
    def test_from_wav_to_npy(self):
        from_wav_to_npy('../../ai_challenge/samples/')

    def test_azimuth_to_classes(self):
        azimuth = [-1, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        target = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.assertEqual(target, azimuth_to_classes(azimuth, 11))
        self.assertEqual(np.eye(11, dtype=np.float32)[target],
                         azimuth_to_classes(azimuth, 11, one_hot=True))
        self.assertEqual(target[1:], azimuth_to_classes(azimuth, 10))
        self.assertEqual(np.eye(10, dtype=np.float32)[target[1:]],
                         azimuth_to_classes(azimuth, 10, one_hot=True))

        # vad
        target = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(target, azimuth_to_classes(azimuth, 2))
        self.assertEqual(np.eye(2, dtype=np.float32)[target],
                         azimuth_to_classes(azimuth, 2, one_hot=True))

    def test_class_to_azimuth(self):
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, -1],
                         class_to_azimuth(classes))

    def test_normalize_spec(self):
        pass

    def test_freq_mask(self):
        pass

    def test_time_mask(self):
        pass

    def test_random_equalizer(self):
        pass
        

if __name__ == '__main__':
    unittest.main()
