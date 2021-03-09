import unittest
from flow_lib.flow_class import Flow
import numpy as np
import math


class TestFlow(unittest.TestCase):
    # All numerical values tested were calculated manually as an independent check
    def test_zero(self):
        dims = [200, 300]
        zero_flow = Flow.zero(dims)
        self.assertIsNone(np.testing.assert_equal(zero_flow.shape[:2], dims))
        self.assertIsNone(np.testing.assert_equal(zero_flow.vecs, 0))
        self.assertIs(zero_flow.ref, 't')
        zero_flow = Flow.zero(dims, 's')
        self.assertIs(zero_flow.ref, 's')

    def test_from_matrix(self):
        # With reference 's', this simply corresponds to using flow_from_matrix, tested in test_utils.
        # With reference 't':
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        dims = [200, 300]
        flow = Flow.from_matrix(matrix, dims, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_from_transforms(self):
        dims = [200, 300]
        transforms = [['rotation', 10, 50, -30]]
        flow = Flow.from_transforms(transforms, dims)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))
        self.assertEqual(flow.ref, 't')

        transforms = [['scaling', 20, 30, 2]]
        flow = Flow.from_transforms(transforms, dims, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))
        self.assertEqual(flow.ref, 's')

        transforms = [
            ['translation', -20, -30],
            ['scaling', 0, 0, 2],
            ['translation', 20, 30]
        ]
        flow = Flow.from_transforms(transforms, dims, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))
        self.assertEqual(flow.ref, 's')

        transforms = [
            ['translation', -10, -50],
            ['rotation', 0, 0, -30],
            ['translation', 10, 50]
        ]
        flow = Flow.from_transforms(transforms, dims, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))
        self.assertEqual(flow.ref, 't')


if __name__ == '__main__':
    unittest.main()
