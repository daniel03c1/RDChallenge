import os
import numpy as np
import tensorflow as tf
from metrics import *


class MetricsTest(tf.test.TestCase):
    def test_challenge_score(self):
        true = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        pred = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertAlmostEqual(np.pi**2, 
                               challenge_score(true, pred).numpy(),
                               places=5)

    def test_score(self):
        degree1 = np.array([ 0, 40, -1, -1])
        degree2 = np.array([-1, -1,100,160]) 
        self.assertAlmostEqual(np.pi**2, 
                               score(degree1, degree2).numpy(),
                               places=5)

        degree1 = np.array([ 0, 20,  0])
        degree2 = np.array([ 0, 40,180])
        self.assertAlmostEqual(np.mean(np.square((degree1-degree2)/180*np.pi)),
                               score(degree1, degree2).numpy(),
                               places=5)

    def test_to_degrees(self):
        dists = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        targets = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, -1])

        self.assertAllEqual(targets, to_degrees(dists))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()
