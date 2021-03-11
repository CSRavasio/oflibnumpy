import unittest
import numpy as np
from flow_lib.utils import matrix_from_transform, flow_from_matrix
import math


class TestMatrixFromTransform(unittest.TestCase):
    # All numerical values in desired_matrix calculated manually
    def test_translation(self):
        # Translation of 15 horizontally, 10 vertically
        desired_matrix = np.eye(3)
        transform = 'translation'
        values = [15, 10]
        desired_matrix[0, 2] = 15
        desired_matrix[1, 2] = 10
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [0, 0, 30]
        desired_matrix[:2, :2] = [[math.sqrt(3) / 2, .5], [-.5, math.sqrt(3) / 2]]
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Rotation of 45 degrees clockwise
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [0, 0, -45]
        desired_matrix[:2, :2] = [[1 / math.sqrt(2), -1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]]
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [10, 50, -30]
        desired_matrix[:2, :2] = [[math.sqrt(3) / 2, -.5], [.5, math.sqrt(3) / 2]]
        desired_matrix[0, 2] = 26.3397459622
        desired_matrix[1, 2] = 1.69872981078
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Rotation of 45 degrees counter-clockwise around point [-20, -30] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'rotation'
        values = [-20, -30, 45]
        desired_matrix[:2, :2] = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), 1 / math.sqrt(2)]]
        desired_matrix[0, 2] = 15.3553390593
        desired_matrix[1, 2] = -22.9289321881
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_scaling(self):
        # Scaling factor 0.8
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [0, 0, 0.8]
        desired_matrix[0, 0] = 0.8
        desired_matrix[1, 1] = 0.8
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Scaling factor 2
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [0, 0, 2]
        desired_matrix[0, 0] = 2
        desired_matrix[1, 1] = 2
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

    def test_scaling_with_shift(self):
        # Scaling factor 0.8 around point [10, 50] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [10, 50, 0.8]
        desired_matrix[0, 0] = 0.8
        desired_matrix[1, 1] = 0.8
        desired_matrix[0, 2] = 2
        desired_matrix[1, 2] = 10
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))

        # Scaling factor 2 around point [20, 30] (hor, ver)
        desired_matrix = np.eye(3)
        transform = 'scaling'
        values = [20, 30, 2]
        desired_matrix[0, 0] = 2
        desired_matrix[1, 1] = 2
        desired_matrix[0, 2] = -20
        desired_matrix[1, 2] = -30
        self.assertIsNone(np.testing.assert_allclose(desired_matrix, matrix_from_transform(transform, values)))


class TestFlowFromMatrix(unittest.TestCase):
    # All numerical values in calculated manually and independently
    def test_identity(self):
        # No transformation, equals passing identy matrix, to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.eye(3)
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_equal(flow, 0))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_translation(self):
        # Translation of 10 horizontally, 20 vertically, to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_equal(flow[..., 0], 10))
        self.assertIsNone(np.testing.assert_equal(flow[..., 1], 20))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise, to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.array([[math.sqrt(3) / 2, .5, 0], [-.5, math.sqrt(3) / 2, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_equal(flow[0, 0], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[0, 299], [-40.0584042685, -149.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 0], [99.5, -26.6609446469]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver), to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[50, 299], [-38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 10], [-74.5, -19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_scaling(self):
        # Scaling factor 0.8, to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.array([[.8, 0, 0], [0, .8, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_equal(flow[0, 0], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[0, 100], [-20, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[100, 0], [0, -20]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_scaling_with_shift(self):
        # Scaling factor 2 around point [20, 30] (hor, ver), to 200 by 300 flow field
        dims = [200, 300]
        matrix = np.array([[2, 0, -20], [0, 2, -30], [0, 0, 1]])
        flow = flow_from_matrix(matrix, dims)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], dims))

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            flow_from_matrix(np.eye(3), 'test')
        with self.assertRaises(ValueError):
            flow_from_matrix(np.eye(3), [10, 10, 10])
        with self.assertRaises(ValueError):
            flow_from_matrix(np.eye(3), [-1, 10])
        with self.assertRaises(ValueError):
            flow_from_matrix(np.eye(3), [10., 10])


if __name__ == '__main__':
    unittest.main()
