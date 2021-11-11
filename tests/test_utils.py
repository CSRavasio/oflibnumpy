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
from oflibnumpy.utils import get_valid_ref, get_valid_padding, validate_shape, \
    matrix_from_transforms, matrix_from_transform, flow_from_matrix, bilinear_interpolation, apply_flow, \
    points_inside_area, threshold_vectors, from_matrix, from_transforms, load_kitti, load_sintel, load_sintel_mask, \
    resize_flow
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
        # No transformation, equals passing identity matrix, to 200 by 300 flow field
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
        img = cv2.imread('smudge.png', 0)
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 255.5, 255.5, -30]], img.shape[:2], ref).vecs
            control_img = rotate(img, -30, reshape=False)
            warped_img = apply_flow(flow, img, ref)
            self.assertIsNone(np.testing.assert_allclose(control_img[200:300, 200:300], warped_img[200:300, 200:300],
                                                         atol=20, rtol=0.05))
            # Note: using SSIM would be a better measure of similarity here, but wanted to avoid extra dependency

    def test_translation(self):
        img = cv2.imread('smudge.png')
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['translation', 10, 20]], img.shape[:2], ref).vecs
            control_img = shift(img, [20, 10, 0])
            warped_img = apply_flow(flow, img, ref)
            self.assertIsNone(np.testing.assert_equal(warped_img, control_img))

    def test_failed_apply(self):
        # Test failure cases
        img = cv2.imread('smudge.png')
        flow = Flow.from_transforms([['translation', 10, 20]], img.shape[:2], 't').vecs
        with self.assertRaises(TypeError):  # Target wrong type
            apply_flow(flow, 2)
        with self.assertRaises(TypeError):  # Flow wrong type
            apply_flow(2, img)
        with self.assertRaises(ValueError):  # Flow ndim only 2
            apply_flow(flow[..., 0], img)
        with self.assertRaises(ValueError):  # Flow channel length only 1
            apply_flow(flow[..., 0:1], img)
        with self.assertRaises(ValueError):  # Target ndim smaller than 2
            apply_flow(flow, img[0, :, 0])
        with self.assertRaises(ValueError):  # Target ndim larger than 2
            apply_flow(flow, img[..., np.newaxis])
        with self.assertRaises(ValueError):  # Target shape does not match flow
            apply_flow(flow, img[:10])
        with self.assertRaises(TypeError):  # Mask wrong type
            apply_flow(flow, img, mask=0)
        with self.assertRaises(TypeError):  # Mask values wrong type
            apply_flow(flow, img, mask=img[..., 0])
        with self.assertRaises(ValueError):  # Mask wrong shape
            apply_flow(flow, img, mask=img)


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


class TestFromMatrix(unittest.TestCase):
    def test_from_matrix(self):
        # With reference 's', this simply corresponds to using flow_from_matrix, tested in test_utils.
        # With reference 't':
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        shape = [200, 300]
        flow = from_matrix(matrix, shape, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_failed_from_matrix(self):
        with self.assertRaises(TypeError):  # Invalid matrix type
            from_matrix('test', [10, 10])
        with self.assertRaises(ValueError):  # Invalid matrix shape
            from_matrix(np.eye(4), [10, 10])


class TestFromTransforms(unittest.TestCase):
    def test_from_transforms_rotation(self):
        shape = [200, 300]
        transforms = [['rotation', 10, 50, -30]]
        flow = from_transforms(transforms, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_from_transforms_scaling(self):
        shape = [200, 300]
        transforms = [['scaling', 20, 30, 2]]
        flow = from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_from_transforms_multiple_s(self):
        shape = [200, 300]
        transforms = [
            ['translation', -20, -30],
            ['scaling', 0, 0, 2],
            ['translation', 20, 30]
        ]
        flow = from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_from_transforms_multiple_t(self):
        shape = [200, 300]
        transforms = [
            ['translation', -10, -50],
            ['rotation', 0, 0, -30],
            ['translation', 10, 50]
        ]
        flow = from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

    def test_failed_from_transforms(self):
        shape = [200, 300]
        transforms = 'test'
        with self.assertRaises(TypeError):  # transforms not a list
            from_transforms(transforms, shape)
        transforms = ['test']
        with self.assertRaises(TypeError):  # transform not a list
            from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation']]
        with self.assertRaises(ValueError):  # transform missing information
            from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation', 1]]
        with self.assertRaises(ValueError):  # transform with incomplete information
            from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation', 1, 'test', 10]]
        with self.assertRaises(ValueError):  # transform with invalid information
            from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['test', 1, 1, 10]]
        with self.assertRaises(ValueError):  # transform type invalid
            from_transforms(transforms, shape)


class TestFromKITTI(unittest.TestCase):
    def test_load(self):
        output = load_kitti('kitti.png')
        self.assertIsInstance(output, np.ndarray)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(output[..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(output[..., 1], 0))
        self.assertIsNone(np.testing.assert_equal(output[:, 0, 2], 1))
        self.assertIsNone(np.testing.assert_equal(output[:, 10, 2], 0))

    def test_failed_load(self):
        with self.assertRaises(ValueError):  # Wrong path
            load_kitti('test')
        with self.assertRaises(ValueError):  # Wrong flow shape
            load_kitti('kitti_wrong.png')


class TestFromSintel(unittest.TestCase):
    def test_load_flow(self):
        f = load_sintel('sintel.flo')
        self.assertIsInstance(f, np.ndarray)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(f[..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(f[..., 1], 0))

    def test_failed_load_flow(self):
        with self.assertRaises(TypeError):  # Path not a string
            load_sintel(0)
        with self.assertRaises(ValueError):  # Wrong tag
            load_sintel('sintel_wrong.flo')

    def test_load_mask(self):
        m = load_sintel_mask('sintel_invalid.png')
        self.assertIsInstance(m, np.ndarray)
        self.assertIsNone(np.testing.assert_equal(m[:, 0], True))
        self.assertIsNone(np.testing.assert_equal(m[:, 10], False))

    def test_failed_load_mask(self):
        with self.assertRaises(TypeError):  # Path not a string
            load_sintel_mask(0)
        with self.assertRaises(ValueError):  # File does not exist
            load_sintel_mask('test.png')


class TestResizeFlow(unittest.TestCase):
    def test_resize(self):
        shape = [20, 10]
        ref = 's'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref).vecs
        # Different scales
        scales = [.2, .5, 1, 1.5, 2, 10]
        for scale in scales:
            resized_flow = resize_flow(flow, scale)
            resized_shape = scale * np.array(shape)
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape[:2], resized_shape))
            self.assertIsNone(np.testing.assert_allclose(resized_flow[0, 0], flow[0, 0] * scale, rtol=.1))

        # Scale list
        scale = [.5, 2]
        resized_flow = resize_flow(flow, scale)
        resized_shape = np.array(scale) * np.array(shape)
        self.assertIsNone(np.testing.assert_equal(resized_flow.shape[:2], resized_shape))
        self.assertIsNone(np.testing.assert_allclose(resized_flow[0, 0], flow[0, 0] * np.array(scale)[::-1], rtol=.1))

        # Scale tuple
        scale = (2, .5)
        resized_flow = resize_flow(flow, scale)
        resized_shape = np.array(scale) * np.array(shape)
        self.assertIsNone(np.testing.assert_equal(resized_flow.shape[:2], resized_shape))
        self.assertIsNone(np.testing.assert_allclose(resized_flow[0, 0], flow[0, 0] * np.array(scale)[::-1], rtol=.1))

    def test_resize_on_fields(self):
        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref).vecs
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref).vecs
        flow_resized = resize_flow(flow_large, 1 / 3)
        self.assertIsNone(np.testing.assert_allclose(flow_resized, flow_small, atol=1, rtol=.1))

    def test_failed_resize(self):
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], [20, 10], 's').vecs
        with self.assertRaises(TypeError):  # Wrong flow array type
            resize_flow('test', 2)
        with self.assertRaises(ValueError):  # Wrong flow array shape
            resize_flow(flow[..., 0], 2)
        with self.assertRaises(TypeError):  # Wrong shape type
            resize_flow(flow, 'test')
        with self.assertRaises(ValueError):  # Wrong shape values
            resize_flow(flow, ['test', 0])
        with self.assertRaises(ValueError):  # Wrong shape shape
            resize_flow(flow, [1, 2, 3])
        with self.assertRaises(ValueError):  # Shape is 0
            resize_flow(flow, 0)
        with self.assertRaises(ValueError):  # Shape below 0
            resize_flow(flow, -0.1)


if __name__ == '__main__':
    unittest.main()
