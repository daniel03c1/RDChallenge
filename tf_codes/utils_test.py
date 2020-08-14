import numpy as np
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_degree_to_class(self):
        degrees = [-1, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        target = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.assertEqual(target, 
                         degree_to_class(degrees, 20, 0, 180, False).tolist())
        self.assertEqual(np.eye(11)[target].tolist(), 
                         degree_to_class(degrees, 20, 0, 180, True).tolist())

    def test_class_to_degree(self):
        classes = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual([-1, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                         class_to_degree(classes, 20, 0, 180, -1).tolist())

    def test_seq_to_windows(self):
        seq = np.array([1, 2, 3, 4, 5])
        window = np.array([-3, -1, 0, 1, 3])

        target = np.array([[0, 0, 1, 2, 4],
                           [0, 1, 2, 3, 5],
                           [0, 2, 3, 4, 0],
                           [1, 3, 4, 5, 0],
                           [2, 4, 5, 0, 0]])
        self.assertEqual(target.tolist(), 
                         seq_to_windows(seq, window).tolist())
        self.assertEqual(target[::2].tolist(),
                         seq_to_windows(seq, window, 2).tolist())

    def test_windows_to_seq(self):
        windows = np.array([[0, 0, 1, 2, 4],
                            [0, 1, 2, 3, 5],
                            [0, 2, 3, 4, 0],
                            [1, 3, 4, 5, 0],
                            [2, 4, 5, 0, 0]])
        window = np.array([-3, -1, 0, 1, 3])

        target = np.array([1, 2, 3, 4, 5])
        self.assertTrue(
            np.allclose(target, windows_to_seq(windows, window)))
        self.assertTrue(
            np.allclose(target, windows_to_seq(windows[::2], window, skip=2)))

    def test_window_generator(self):
        np.random.seed(0)
        specs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        target = zip(
            [[1, 2, 3], [3, 4, 5], [4, 5, 6], [6, 7, 8]],
            [2, 4, 5, 7])
        gen = window_generator(specs, labels, 3, infinite=False)
        output = [(s, l) for s, l in gen()]

        for (t_s, t_l), (s, l) in zip(target, output):
            self.assertEqual(t_s, s.tolist())
            self.assertEqual(t_l, l.tolist())

        # infinite test
        win_size = 3
        gen = window_generator(specs, labels, win_size, infinite=True)
        for i, (s, l) in enumerate(gen()):
            if i > len(specs) * 2:
                break
            self.assertEqual(win_size, len(s))
            # self.assertEqual(1, len(l))

        

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()
