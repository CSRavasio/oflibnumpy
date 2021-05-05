#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: MIT (https://opensource.org/licenses/MIT)
# Author: Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's College
# London (KCL), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of oflibnumpy

import unittest
import math
import cv2
import numpy as np
from scipy.ndimage import rotate, shift
from skimage.metrics import structural_similarity
from oflibnumpy.utils import get_valid_ref, get_valid_padding, validate_shape, \
    matrix_from_transforms, matrix_from_transform, flow_from_matrix, bilinear_interpolation, apply_flow, \
    points_inside_area, threshold_vectors
from oflibnumpy.flow_class import Flow


class TestValidityChecks(unittest.TestCase):
    def test_get_valid_ref(self):
        self.assertEqual(get_valid_ref(None), 't')
        self.assertEqual(get_valid_ref('s'), 's')
        self.assertEqual(get_valid_ref('t'), 't')
        with self.assertRaises(TypeError):
            get_valid_ref(0)
        with self.assertRaises(ValueError):
            get_valid_ref('test')

    def test_get_valid_padding(self):
        with self.assertRaises(TypeError):
            get_valid_padding(100)
        with self.assertRaises(ValueError):
            get_valid_padding([10, 20, 30, 40, 50])
        with self.assertRaises(ValueError):
            get_valid_padding([10., 20, 30, 40])
        with self.assertRaises(ValueError):
            get_valid_padding([-10, 10, 10, 10])

    def test_validate_shape(self):
        with self.assertRaises(TypeError):
            validate_shape('test')
        with self.assertRaises(ValueError):
            validate_shape([10, 10, 10])
        with self.assertRaises(ValueError):
            validate_shape([-1, 10])
        with self.assertRaises(ValueError):
            validate_shape([10., 10])


class TestMatrixFromTransforms(unittest.TestCase):
    # All numerical values in desired_matrix calculated manually
    def test_combined_transforms(self):
        transforms = [
            ['translation', -100, -100],
            ['rotation', 0, 0, 30],
            ['translation', 100, 100]
        ]
        actual_matrix = matrix_from_transforms(transforms)
        desired_matrix = matrix_from_transform('rotation', [100, 100, 30])
        self.assertIsNone(np.testing.assert_equal(actual_matrix, desired_matrix))


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
        shape = [200, 300]
        matrix = np.eye(3)
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(flow, 0))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_translation(self):
        # Translation of 10 horizontally, 20 vertically, to 200 by 300 flow field
        shape = [200, 300]
        matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(flow[..., 0], 10))
        self.assertIsNone(np.testing.assert_equal(flow[..., 1], 20))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_rotation(self):
        # Rotation of 30 degrees counter-clockwise, to 200 by 300 flow field
        shape = [200, 300]
        matrix = np.array([[math.sqrt(3) / 2, .5, 0], [-.5, math.sqrt(3) / 2, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(flow[0, 0], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[0, 299], [-40.0584042685, -149.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 0], [99.5, -26.6609446469]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_rotation_with_shift(self):
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver), to 200 by 300 flow field
        shape = [200, 300]
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[50, 299], [-38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 10], [-74.5, -19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_scaling(self):
        # Scaling factor 0.8, to 200 by 300 flow field
        shape = [200, 300]
        matrix = np.array([[.8, 0, 0], [0, .8, 0], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_equal(flow[0, 0], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[0, 100], [-20, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[100, 0], [0, -20]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_scaling_with_shift(self):
        # Scaling factor 2 around point [20, 30] (hor, ver), to 200 by 300 flow field
        shape = [200, 300]
        matrix = np.array([[2, 0, -20], [0, 2, -30], [0, 0, 1]])
        flow = flow_from_matrix(matrix, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))


class TestBilinearInterpolation(unittest.TestCase):
    def test_int_on_shift(self):
        flow = flow_from_matrix(matrix_from_transform('translation', [10, 20]), (512, 512))
        pts = np.array([[200, 300], [1, 2]])
        desired_result = np.array([[20, 10], [20, 10]])
        actual_result = bilinear_interpolation(flow[..., ::-1], pts)
        self.assertIsNone(np.testing.assert_equal(actual_result, desired_result))

    def test_float_on_rot(self):
        flow = flow_from_matrix(matrix_from_transform('rotation', [0, 0, 30]), (512, 512))
        pts = np.array([[20.5, 10.5], [8.3, 7.2], [120.4, 160.2]])
        desired_pts = [
            [12.5035207776, 19.343266740],
            [3.58801085141, 10.385382907],
            [24.1694586156, 198.93726969]
        ]
        desired_result = np.array(desired_pts) - pts
        actual_result = bilinear_interpolation(flow[..., ::-1], pts)
        self.assertIsNone(np.testing.assert_allclose(actual_result, desired_result,
                                                     atol=1e-1, rtol=1e-2))
        # High tolerance needed as exact result is compared to an interpolated one

    def test_invalid_input(self):
        flow = flow_from_matrix(matrix_from_transform('rotation', [0, 0, 30]), (512, 512))
        with self.assertRaises(IndexError):
            bilinear_interpolation(flow, np.array([[-1, 0], [10, 10]]))
        with self.assertRaises(IndexError):
            bilinear_interpolation(flow, np.array([[0, 0], [511.01, 10]]))


class TestApplyFlow(unittest.TestCase):
    def test_rotation(self):
        img = cv2.imread('smudge.png')
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 255.5, 255.5, -30]], img.shape[:2], ref).vecs
            control_img = rotate(img, -30, reshape=False)
            warped_img = apply_flow(flow, img, ref)
            # Values will not be exactly the same due to rounding etc., so use SSIM instead
            ssim = structural_similarity(control_img, warped_img, multichannel=True)
            self.assertTrue(ssim > 0.98)

    def test_translation(self):
        img = cv2.imread('smudge.png')
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['translation', 10, 20]], img.shape[:2], ref).vecs
            control_img = shift(img, [20, 10, 0])
            warped_img = apply_flow(flow, img, ref)
            self.assertIsNone(np.testing.assert_equal(warped_img, control_img))


class TestPointsInsideArea(unittest.TestCase):
    def test_points(self):
        shape = [10, 20]
        pts = np.array([
            [-1, -1],
            [-1, 0],
            [0, -1],
            [0, 0],
            [9, 20],
            [9, 19],
            [10, 19],
            [10, 20]
        ])
        desired_array = [False, False, False, True, False, True, False, False]
        self.assertIsNone(np.testing.assert_equal(points_inside_area(pts, shape), desired_array))


class TestThresholdVectors(unittest.TestCase):
    def test_threshold(self):
        vecs = np.zeros((10, 1, 2))
        vecs[0, 0, 0] = 1e-5
        vecs[1, 0, 0] = 1e-4
        vecs[2, 0, 0] = 1e-3
        vecs[3, 0, 0] = 1
        thresholded = threshold_vectors(vecs, threshold=1e-3)
        self.assertIsNone(np.testing.assert_equal(thresholded[:4, 0, 0], [0, 0, 1e-3, 1]))
        thresholded = threshold_vectors(vecs, threshold=1e-4)
        self.assertIsNone(np.testing.assert_equal(thresholded[:4, 0, 0], [0, 1e-4, 1e-3, 1]))
        thresholded = threshold_vectors(vecs, threshold=1e-5)
        self.assertIsNone(np.testing.assert_equal(thresholded[:4, 0, 0], [1e-5, 1e-4, 1e-3, 1]))


if __name__ == '__main__':
    unittest.main()
