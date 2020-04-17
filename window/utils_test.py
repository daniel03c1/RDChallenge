import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_sequence_to_windows(self):
        sequence = np.array(
            [[0, 9],
             [1, 8],
             [2, 7],
             [3, 6],
             [4, 5],
             [5, 4],
             [6, 3]])

        # With Padding
        targets = [ # window: (-3, -1, 0, 1, 3)
            [[0, 0], [0, 0], [0, 9], [1, 8], [3, 6]],
            [[0, 0], [0, 9], [1, 8], [2, 7], [4, 5]],
            [[0, 0], [1, 8], [2, 7], [3, 6], [5, 4]],
            [[0, 9], [2, 7], [3, 6], [4, 5], [6, 3]],
            [[1, 8], [3, 6], [4, 5], [5, 4], [0, 0]],
            [[2, 7], [4, 5], [5, 4], [6, 3], [0, 0]],
            [[3, 6], [5, 4], [6, 3], [0, 0], [0, 0]]
        ]
        windows = sequence_to_windows(sequence, 3, 2, padding=True)
        self.assertEqual(targets, windows.tolist())

        # With Padding (skip=2)
        targets = [ # window: (-3, -1, 0, 1, 3)
            [[0, 0], [0, 0], [0, 9], [1, 8], [3, 6]],
            [[0, 0], [1, 8], [2, 7], [3, 6], [5, 4]],
            [[1, 8], [3, 6], [4, 5], [5, 4], [0, 0]],
            [[3, 6], [5, 4], [6, 3], [0, 0], [0, 0]]
        ]
        windows = sequence_to_windows(sequence, 3, 2, skip=2, padding=True)
        self.assertEqual(targets, windows.tolist())

        # Without Padding
        targets = [
            [[0, 9], [2, 7], [3, 6], [4, 5], [6, 3]]
        ]
        windows = sequence_to_windows(sequence, 3, 2, padding=False)
        self.assertEqual(targets, windows.tolist())

    def test_windows_to_sequence(self):
        windows = [ # window: (-1, 0, 1)
            [0., 1., 1.],
            [1., .5, .5],
            [0., 0., 0.]
        ]
        self.assertEqual(
            [1., .5, .25],
            windows_to_sequence(windows, 1, 1).tolist())

        windows = [ # window: (-3, -1, 0, 1, 3)
            [0, 0, 1, 2, 4],
            [0, 1, 2, 3, 5],
            [0, 2, 3, 4, 0],
            [1, 3, 4, 5, 0],
            [2, 4, 5, 0, 0]
        ]
        self.assertEqual(
            [1, 2, 3, 4, 5],
            windows_to_sequence(windows, 3, 2).tolist())

    def test_pad(self):
        arr = np.array([1, 2, 3])
        self.assertEqual([-1, -1, 1, 2, 3, -1, -1],
                         pad(arr, 2, axis=0, const_value=-1).tolist())

        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        self.assertEqual([[0, 1, 2, 3, 0],
                          [0, 4, 5, 6, 0]],
                         pad(matrix, 1, axis=1, const_value=0).tolist())


if __name__ == '__main__':
    unittest.main()
