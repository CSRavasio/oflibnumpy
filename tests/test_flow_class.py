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
import numpy as np
import cv2
import math
from oflibnumpy.utils import matrix_from_transforms, apply_flow
from oflibnumpy.flow_class import Flow


class TestFlow(unittest.TestCase):
    # All numerical values tested were calculated manually as an independent check
    def test_flow(self):
        # Flow from vectors
        vectors = np.random.rand(200, 200, 2)
        flow = Flow(vectors)
        self.assertIsNone(np.testing.assert_allclose(flow.vecs, vectors))
        self.assertEqual(flow.ref, 't')
        self.assertEqual(flow.shape, vectors.shape[:2])
        self.assertEqual(flow.mask.shape, vectors.shape[:2])
        self.assertEqual(np.sum(flow.mask), vectors.size // 2)

        # Incorrect flow type
        vectors = np.random.rand(200, 200, 2).tolist()
        with self.assertRaises(TypeError):
            Flow(vectors)

        # Incorrect flow shape
        vectors = np.random.rand(200, 200)
        with self.assertRaises(ValueError):
            Flow(vectors)

        # Incorrect flow values
        vectors = np.random.rand(200, 200, 2)
        vectors[10, 10] = np.NaN
        with self.assertRaises(ValueError):
            Flow(vectors)

        # Incorrect flow shape
        vectors = np.random.rand(200, 200, 3)
        with self.assertRaises(ValueError):
            Flow(vectors)

        # Flow from vectors and reference
        vectors = np.random.rand(200, 200, 2)
        ref = 's'
        flow = Flow(vectors, ref)
        self.assertEqual(flow.ref, 's')

        # Incorrect reference value
        ref = 'test'
        with self.assertRaises(ValueError):
            Flow(vectors, ref)
        ref = 10
        with self.assertRaises(TypeError):
            Flow(vectors, ref)

        # Flow from vectors, reference, mask dtype 'bool'
        mask = np.ones((200, 200), 'bool')
        ref = 's'
        flow = Flow(vectors, ref, mask)
        self.assertIsNone(np.testing.assert_equal(flow.mask, mask))
        self.assertEqual(flow.ref, 's')

        # Flow from vectors, reference, mask dtype 'f'
        mask = np.ones((200, 200), 'f')
        ref = 't'
        flow = Flow(vectors, ref, mask)
        self.assertIsNone(np.testing.assert_equal(flow.mask, mask))
        self.assertEqual(flow.ref, 't')

        # Incorrect mask type
        mask = np.ones((200, 200), 'bool').tolist()
        with self.assertRaises(TypeError):
            Flow(vectors, mask=mask)

        # Incorrect mask shape
        mask = np.ones((210, 200), 'bool')
        with self.assertRaises(ValueError):
            Flow(vectors, mask=mask)

        # Incorrect mask shape
        mask = np.ones((210, 200, 2), 'bool')
        with self.assertRaises(ValueError):
            Flow(vectors, mask=mask)

        # Incorrect mask shape
        mask = np.ones((200, 200), 'f')
        mask[[10, 10], [20, 30]] = 2
        with self.assertRaises(ValueError):
            Flow(vectors, mask=mask)

    def test_zero(self):
        shape = [200, 300]
        zero_flow = Flow.zero(shape)
        self.assertIsNone(np.testing.assert_equal(zero_flow.shape[:2], shape))
        self.assertIsNone(np.testing.assert_equal(zero_flow.vecs, 0))
        self.assertIs(zero_flow.ref, 't')
        zero_flow = Flow.zero(shape, 's')
        self.assertIs(zero_flow.ref, 's')

    def test_from_matrix(self):
        # With reference 's', this simply corresponds to using flow_from_matrix, tested in test_utils.
        # With reference 't':
        # Rotation of 30 degrees clockwise around point [10, 50] (hor, ver)
        matrix = np.array([[math.sqrt(3) / 2, -.5, 26.3397459622],
                           [.5, math.sqrt(3) / 2, 1.69872981078],
                           [0, 0, 1]])
        shape = [200, 300]
        flow = Flow.from_matrix(matrix, shape, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))

        # Invalid input
        with self.assertRaises(TypeError):
            Flow.from_matrix('test', [10, 10])
        with self.assertRaises(ValueError):
            Flow.from_matrix(np.eye(4), [10, 10])

    def test_from_transforms(self):
        shape = [200, 300]
        # Invalid transform values
        transforms = 'test'
        with self.assertRaises(TypeError):
            Flow.from_transforms(transforms, shape)
        transforms = ['test']
        with self.assertRaises(TypeError):
            Flow.from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation']]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation', 1]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['rotation', 1, 'test', 10]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, shape)
        transforms = [['translation', 20, 10], ['test', 1, 1, 10]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, shape)

        transforms = [['rotation', 10, 50, -30]]
        flow = Flow.from_transforms(transforms, shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))
        self.assertEqual(flow.ref, 't')

        transforms = [['scaling', 20, 30, 2]]
        flow = Flow.from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))
        self.assertEqual(flow.ref, 's')

        transforms = [
            ['translation', -20, -30],
            ['scaling', 0, 0, 2],
            ['translation', 20, 30]
        ]
        flow = Flow.from_transforms(transforms, shape, 's')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[30, 20], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[30, 70], [50, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[80, 20], [0, 50]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))
        self.assertEqual(flow.ref, 's')

        transforms = [
            ['translation', -10, -50],
            ['rotation', 0, 0, -30],
            ['translation', 10, 50]
        ]
        flow = Flow.from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_array_almost_equal(flow.vecs[50, 10], [0, 0]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[50, 299], [38.7186583063, 144.5]))
        self.assertIsNone(np.testing.assert_allclose(flow.vecs[199, 10], [-74.5, 19.9622148361]))
        self.assertIsNone(np.testing.assert_equal(flow.shape[:2], shape))
        self.assertEqual(flow.ref, 't')
    
    def test_from_kitti(self):
        path = 'kitti.png'
        f = Flow.from_kitti(path, load_valid=True)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(f.vecs[..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(f.vecs[..., 1], 0))
        self.assertIsNone(np.testing.assert_equal(f.mask[:, 0], True))
        self.assertIsNone(np.testing.assert_equal(f.mask[:, 10], False))
        f = Flow.from_kitti(path, load_valid=False)
        self.assertIsNone(np.testing.assert_equal(f.mask, True))

        with self.assertRaises(TypeError):  # Wrong load_valid type
            Flow.from_kitti(path, load_valid='test')
        with self.assertRaises(ValueError):  # Wrong path
            Flow.from_kitti('test')
        with self.assertRaises(ValueError):  # Wrong flow shape
            Flow.from_kitti('kitti_wrong.png')

    def test_from_sintel(self):
        path = 'sintel.flo'
        f = Flow.from_sintel(path)
        desired_flow = np.arange(0, 10)[:, np.newaxis] * np.arange(0, 20)[np.newaxis, :]
        self.assertIsNone(np.testing.assert_equal(f.vecs[..., 0], desired_flow))
        self.assertIsNone(np.testing.assert_equal(f.mask, True))
        f = Flow.from_sintel(path, 'sintel_invalid.png')
        self.assertIsNone(np.testing.assert_equal(f.mask[:, 0], True))
        self.assertIsNone(np.testing.assert_equal(f.mask[:, 10], False))

        with self.assertRaises(ValueError):  # Wrong tag
            Flow.from_sintel('sintel_wrong.flo')
        with self.assertRaises(ValueError):  # Wrong mask path
            Flow.from_sintel(path, 'test.png')
        with self.assertRaises(ValueError):  # Wrong mask shape
            Flow.from_sintel(path, 'sintel_invalid_wrong.png')

    def test_getitem(self):
        vectors = np.random.rand(200, 200, 2)
        flow = Flow(vectors)
        indices = np.random.randint(0, 150, size=(20, 2))
        for i in indices:
            # Cutting a number of elements
            self.assertIsNone(np.testing.assert_allclose(flow[i].vecs, vectors[i]))
            # Cutting a specific item
            self.assertIsNone(np.testing.assert_allclose(flow[i[0]:i[0] + 1, i[1]:i[1] + 1].vecs,
                                                         vectors[i[0]:i[0] + 1, i[1]:i[1] + 1]))
            # Cutting an area
            self.assertIsNone(np.testing.assert_allclose(flow[i[0]:i[0] + 40, i[1]:i[1] + 40].vecs,
                                                         vectors[i[0]:i[0] + 40, i[1]:i[1] + 40]))

    def test_copy(self):
        vectors = np.random.rand(200, 200, 2)
        mask = np.random.rand(200, 200) > 0.5
        for ref in ['t', 's']:
            flow = Flow(vectors, ref, mask)
            flow_copy = flow.copy()
            self.assertIsNone(np.testing.assert_equal(flow.vecs, flow_copy.vecs))
            self.assertIsNone(np.testing.assert_equal(flow.mask, flow_copy.mask))
            self.assertEqual(flow.ref, flow_copy.ref)
            self.assertNotEqual(id(flow), id(flow_copy))

    def test_add(self):
        mask1 = np.ones((100, 200), 'bool')
        mask1[:40] = 0
        mask2 = np.ones((100, 200), 'bool')
        mask2[60:] = 0
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        vecs3 = np.random.rand(200, 200, 2)
        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)

        # Addition
        self.assertIsNone(np.testing.assert_allclose((flow1 + vecs2).vecs, vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_allclose((flow1 + flow2).vecs, vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum((flow1 + flow2).mask), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 + 'test'
        with self.assertRaises(ValueError):
            flow1 + flow3
        with self.assertRaises(ValueError):
            flow1 + vecs3

    def test_sub(self):
        mask1 = np.ones((100, 200), 'bool')
        mask1[:40] = 0
        mask2 = np.ones((100, 200), 'bool')
        mask2[60:] = 0
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        vecs3 = np.random.rand(200, 200, 2)
        flow1 = Flow(vecs1, mask=mask1)
        flow2 = Flow(vecs2, mask=mask2)
        flow3 = Flow(vecs3)

        # Subtraction
        self.assertIsNone(np.testing.assert_allclose((flow1 - flow2).vecs, vecs1 - vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_allclose((flow1 - vecs2).vecs, vecs1 - vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum((flow1 - flow2).mask), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 - 'test'
        with self.assertRaises(ValueError):
            flow1 - flow3
        with self.assertRaises(ValueError):
            flow1 - vecs3

    def test_mul(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Multiplication
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 * i).vecs, vecs1 * i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 * f).vecs, vecs1 * f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 * list(li)).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] *= li[0]
            v[..., 1] *= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 * li).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow
        self.assertIsNone(np.testing.assert_allclose((flow1 * vecs2[..., 0]).vecs, vecs1 * vecs2[..., :1],
                                                     rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow vectors
        self.assertIsNone(np.testing.assert_allclose((flow1 * vecs2).vecs, vecs1 * vecs2, rtol=1e-6, atol=1e-6))
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 * [0, 1, 2]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 * np.array([0, 1, 2])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200, 2)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 * np.random.rand(200, 200, 2, 1)

    def test_div(self):
        vecs1 = np.random.rand(100, 200, 2) + .5
        vecs2 = -np.random.rand(100, 200, 2) - .5
        flow1 = Flow(vecs1)

        # Division
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            if i < -1e-5 or i > 1e-5:
                self.assertIsNone(np.testing.assert_allclose((flow1 / i).vecs, vecs1 / i, rtol=1e-6, atol=1e-6))
            if f < -1e-5 or f > 1e-5:
                self.assertIsNone(np.testing.assert_allclose((flow1 / f).vecs, vecs1 / f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                self.assertIsNone(np.testing.assert_allclose((flow1 / list(li)).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            if li[0] != 0 and li[1] != 0:
                v = vecs1.astype('f')
                v[..., 0] /= li[0]
                v[..., 1] /= li[1]
                self.assertIsNone(np.testing.assert_allclose((flow1 / li).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow
        self.assertIsNone(np.testing.assert_allclose((flow1 / vecs2[..., 0]).vecs, vecs1 / vecs2[..., :1],
                                                     rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow vectors
        self.assertIsNone(np.testing.assert_allclose((flow1 / vecs2).vecs, vecs1 / vecs2, rtol=1e-6, atol=1e-6))
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 / [1, 2, 3]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 / np.array([1, 2, 3])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200))
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200, 2))
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.ones((200, 200, 2, 1))

    def test_pow(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Exponentiation
        ints = np.random.randint(-2, 2, 100)
        floats = (np.random.rand(100) - .5) * 4
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 ** i).vecs, vecs1 ** i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 ** f).vecs, vecs1 ** f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 ** list(li)).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-5, 5, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 ** li).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow
        self.assertIsNone(np.testing.assert_allclose((flow1 ** vecs2[..., 0]).vecs, vecs1 ** vecs2[..., :1],
                                                     rtol=1e-6, atol=1e-6))
        # ... using a numpy array of the same shape as the flow vectors
        self.assertIsNone(np.testing.assert_allclose((flow1 ** vecs2).vecs, vecs1 ** vecs2, rtol=1e-6, atol=1e-6))
        # ... using a list of the wrong length
        with self.assertRaises(ValueError):
            flow1 ** [0, 1, 2]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 ** np.array([0, 1, 2])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200, 2)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 ** np.random.rand(200, 200, 2, 1)

    def test_neg(self):
        vecs1 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)
        self.assertIsNone(np.testing.assert_allclose((-flow1).vecs, -vecs1))

    def test_resize(self):
        shape = [20, 10]
        ref = 's'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)

        # Different scales
        scales = [.2, .5, 1, 1.5, 2, 10]
        for scale in scales:
            resized_flow = flow.resize(scale)
            resized_shape = scale * np.array(shape)
            self.assertIsNone(np.testing.assert_equal(resized_flow.shape, resized_shape))
            self.assertIsNone(np.testing.assert_allclose(resized_flow.vecs[0, 0], flow.vecs[0, 0] * scale, rtol=.1))

        # Scale list
        scale = [.5, 2]
        resized_flow = flow.resize(scale)
        resized_shape = np.array(scale) * np.array(shape)
        self.assertIsNone(np.testing.assert_equal(resized_flow.shape, resized_shape))
        self.assertIsNone(np.testing.assert_allclose(resized_flow.vecs[0, 0],
                                                     flow.vecs[0, 0] * np.array(scale)[::-1],
                                                     rtol=.1))

        # Scale tuple
        scale = (2, .5)
        resized_flow = flow.resize(scale)
        resized_shape = np.array(scale) * np.array(shape)
        self.assertIsNone(np.testing.assert_equal(resized_flow.shape, resized_shape))
        self.assertIsNone(np.testing.assert_allclose(resized_flow.vecs[0, 0],
                                                     flow.vecs[0, 0] * np.array(scale)[::-1],
                                                     rtol=.1))

        # Scale mask
        shape_small = (20, 40)
        shape_large = (30, 80)
        mask_small = np.ones(shape_small, 'bool')
        mask_small[:6, :20] = 0
        mask_large = np.ones(shape_large, 'bool')
        mask_large[:9, :40] = 0
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], shape_small, 't', mask_small)
        flow_large = flow_small.resize((1.5, 2))
        self.assertIsNone(np.testing.assert_equal(flow_large.mask, mask_large))

        # Check scaling is performed correctly based on the actual flow field
        ref = 't'
        flow_small = Flow.from_transforms([['rotation', 0, 0, 30]], (50, 80), ref)
        flow_large = Flow.from_transforms([['rotation', 0, 0, 30]], (150, 240), ref)
        flow_resized = flow_large.resize(1/3)
        self.assertIsNone(np.testing.assert_allclose(flow_resized.vecs, flow_small.vecs, atol=1, rtol=.1))

        # Invalid input
        with self.assertRaises(TypeError):
            flow.resize('test')
        with self.assertRaises(ValueError):
            flow.resize(['test', 0])
        with self.assertRaises(ValueError):
            flow.resize([1, 2, 3])
        with self.assertRaises(ValueError):
            flow.resize(0)
        with self.assertRaises(ValueError):
            flow.resize(-0.1)

    def test_pad(self):
        shape = [100, 80]
        for ref in ['t', 's']:
            flow = Flow.zero(shape, ref, np.ones(shape, 'bool'))
            flow = flow.pad([10, 20, 30, 40])
            self.assertIsNone(np.testing.assert_equal(flow.shape[:2], [shape[0] + 10 + 20, shape[1] + 30 + 40]))
            self.assertIsNone(np.testing.assert_equal(flow.vecs, 0))
            self.assertIsNone(np.testing.assert_equal(flow[10:-20, 30:-40].mask, 1))
            flow.mask[10:-20, 30:-40] = 0
            self.assertIsNone(np.testing.assert_equal(flow.mask, 0))
            self.assertIs(flow.ref, ref)

        # 'Edge' padding
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], shape, ref)
        padded_flow = flow.pad([10, 10, 20, 20], mode='edge')
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs[0, 20:-20], flow.vecs[0]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs[10:-10, 0], flow.vecs[:, 0]))

        # 'Symmetric' padding
        padded_flow = flow.pad([10, 10, 20, 20], mode='symmetric')
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs[0, 20:-20], flow.vecs[9]))
        self.assertIsNone(np.testing.assert_equal(padded_flow.vecs[10:-10, 0], flow.vecs[:, 19]))

        # Invalid padding mode
        with self.assertRaises(ValueError):
            flow.pad([10, 10, 20, 20], mode='test')

    def test_apply(self):
        img = cv2.imread('smudge.png')
        # Check flow.apply results in the same as using apply_flow directly
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], ref)
            mask = np.ones(img.shape[:2], 'bool')
            # Target is a 3D numpy array, without / with mask
            warped_img_desired = apply_flow(flow.vecs, img, ref)
            warped_img_actual = flow.apply(img)
            self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired))
            warped_img_actual, _ = flow.apply(img, mask, return_valid_area=True)
            self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired))
            # Target is a 2D numpy array
            warped_img_desired = apply_flow(flow.vecs, img[..., 0], ref)
            warped_img_actual = flow.apply(img[..., 0])
            self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired))
            warped_img_actual, _ = flow.apply(img[..., 0], mask, return_valid_area=True)
            self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired))
            # Target is a flow object
            warped_flow_desired = apply_flow(flow.vecs, flow.vecs, ref)
            warped_flow_actual = flow.apply(flow)
            self.assertIsNone(np.testing.assert_equal(warped_flow_actual.vecs, warped_flow_desired))
        # Check using a smaller flow field on a larger target works the same as a full flow field on the same target
        ref = 't'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], ref)
        warped_img_desired = apply_flow(flow.vecs, img, ref)
        shape = [img.shape[0] - 90, img.shape[1] - 110]
        padding = [50, 40, 30, 80]
        cut_flow = Flow.from_transforms([['rotation', 0, 0, 30]], shape, ref)
        # ... not cutting (target numpy array)
        warped_img_actual = cut_flow.apply(img, padding=padding, cut=False)
        self.assertIsNone(np.testing.assert_equal(warped_img_actual[padding[0]:-padding[1], padding[2]:-padding[3]],
                                                  warped_img_desired[padding[0]:-padding[1],
                                                                     padding[2]:-padding[3]]))
        # ... cutting (target numpy array)
        warped_img_actual = cut_flow.apply(img, padding=padding, cut=True)
        self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired[padding[0]:-padding[1],
                                                                                        padding[2]:-padding[3]]))
        # ... not cutting (target flow object)
        target_flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], ref)
        warped_flow_desired = apply_flow(flow.vecs, target_flow.vecs, ref)
        warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=False)
        self.assertIsNone(np.testing.assert_equal(warped_flow_actual.vecs[padding[0]:-padding[1],
                                                                          padding[2]:-padding[3]],
                                                  warped_flow_desired[padding[0]:-padding[1],
                                                                      padding[2]:-padding[3]]))
        # ... cutting (target flow object)
        warped_flow_actual = cut_flow.apply(target_flow, padding=padding, cut=True)
        self.assertIsNone(np.testing.assert_equal(warped_flow_actual.vecs, warped_flow_desired[padding[0]:-padding[1],
                                                                                               padding[2]:-padding[3]]))

        # Non-valid input values
        for ref in ['t', 's']:
            shape = (10, 10)
            flow = Flow.from_transforms([['rotation', 0, 0, 30]], shape, ref)
            img = np.ones(shape + (3,), 'uint8')
            with self.assertRaises(ValueError):  # 1D input
                flow.apply(img[0, 0])
            with self.assertRaises(ValueError):  # 4D input
                flow.apply(img[..., np.newaxis])
            with self.assertRaises(TypeError):
                flow.apply(flow, padding=100, cut=True)
            with self.assertRaises(ValueError):
                flow.apply(flow, padding=[10, 20, 30, 40, 50], cut=True)
            with self.assertRaises(ValueError):
                flow.apply(flow, padding=[10., 20, 30, 40], cut=True)
            with self.assertRaises(ValueError):
                flow.apply(flow, padding=[-10, 10, 10, 10], cut=True)
            with self.assertRaises(TypeError):
                flow.apply(flow, padding=[10, 20, 30, 40, 50], cut=2)
            with self.assertRaises(TypeError):
                flow.apply(flow, padding=[10, 20, 30, 40, 50], cut='true')
            with self.assertRaises(TypeError):
                flow.apply(flow, return_valid_area='test')
            with self.assertRaises(TypeError):
                flow.apply(flow, consider_mask='test')
            with self.assertRaises(TypeError):
                flow.apply(img, target_mask='test')
            with self.assertRaises(TypeError):
                flow.apply(img, target_mask=np.ones(shape, 'i'))
            with self.assertRaises(ValueError):
                flow.apply(img, target_mask=np.ones((5, 5), 'bool'))

    def test_switch_ref(self):
        img = cv2.imread('smudge.png')
        # Mode 'invalid'
        for refs in [['t', 's'], ['s', 't']]:
            flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], refs[0])
            flow = flow.switch_ref(mode='invalid')
            self.assertEqual(flow.ref, refs[1])

        # Mode 'valid'
        transforms = [['rotation', 256, 256, 30]]
        flow_s = Flow.from_transforms(transforms, img.shape[:2], 's')
        flow_t = Flow.from_transforms(transforms, img.shape[:2], 't')
        switched_s = flow_t.switch_ref()
        self.assertIsNone(np.testing.assert_allclose(switched_s.vecs[switched_s.mask],
                                                     flow_s.vecs[switched_s.mask],
                                                     rtol=1e-3, atol=1e-3))
        switched_t = flow_s.switch_ref()
        self.assertIsNone(np.testing.assert_allclose(switched_t.vecs[switched_t.mask],
                                                     flow_t.vecs[switched_t.mask],
                                                     rtol=1e-3, atol=1e-3))

        # Invalid mode passed
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], 't')
        with self.assertRaises(ValueError):
            flow.switch_ref('test')
        with self.assertRaises(ValueError):
            flow.switch_ref(1)

    def test_invert(self):
        f_s = Flow.from_transforms([['rotation', 256, 256, 30]], (512, 512), 's')   # Forwards
        f_t = Flow.from_transforms([['rotation', 256, 256, 30]], (512, 512), 't')   # Forwards
        b_s = Flow.from_transforms([['rotation', 256, 256, -30]], (512, 512), 's')  # Backwards
        b_t = Flow.from_transforms([['rotation', 256, 256, -30]], (512, 512), 't')  # Backwards

        # Inverting s to s
        b_s_inv = f_s.invert()
        self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs[b_s_inv.mask],
                                                     b_s.vecs[b_s_inv.mask],
                                                     rtol=1e-3, atol=1e-3))
        f_s_inv = b_s.invert()
        self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs[f_s_inv.mask],
                                                     f_s.vecs[f_s_inv.mask],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting s to t
        b_t_inv = f_s.invert('t')
        self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs[b_t_inv.mask],
                                                     b_t.vecs[b_t_inv.mask],
                                                     rtol=1e-3, atol=1e-3))
        f_t_inv = b_s.invert('t')
        self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs[f_t_inv.mask],
                                                     f_t.vecs[f_t_inv.mask],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting t to t
        b_t_inv = f_t.invert()
        self.assertIsNone(np.testing.assert_allclose(b_t_inv.vecs[b_t_inv.mask],
                                                     b_t.vecs[b_t_inv.mask],
                                                     rtol=1e-3, atol=1e-3))
        f_t_inv = b_t.invert()
        self.assertIsNone(np.testing.assert_allclose(f_t_inv.vecs[f_t_inv.mask],
                                                     f_t.vecs[f_t_inv.mask],
                                                     rtol=1e-3, atol=1e-3))

        # Inverting t to s
        b_s_inv = f_t.invert('s')
        self.assertIsNone(np.testing.assert_allclose(b_s_inv.vecs[b_s_inv.mask],
                                                     b_s.vecs[b_s_inv.mask],
                                                     rtol=1e-3, atol=1e-3))
        f_s_inv = b_t.invert('s')
        self.assertIsNone(np.testing.assert_allclose(f_s_inv.vecs[f_s_inv.mask],
                                                     f_s.vecs[f_s_inv.mask],
                                                     rtol=1e-3, atol=1e-3))

    def test_track(self):
        f_s = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 's')
        f_t = Flow.from_transforms([['rotation', 0, 0, 30]], (512, 512), 't')
        pts = np.array([[20.5, 10.5], [8.3, 7.2], [120.4, 160.2]])
        desired_pts = [
            [12.5035207776, 19.343266740],
            [3.58801085141, 10.385382907],
            [24.1694586156, 198.93726969]
        ]
        pts_tracked_s = f_s.track(pts)
        self.assertIsNone(np.testing.assert_allclose(pts_tracked_s, desired_pts,
                                                     atol=1e-1, rtol=1e-2))
        # High tolerance needed as exact result is compared to an interpolated one
        pts_tracked_s = f_s.track(pts, s_exact_mode=True)
        self.assertIsNone(np.testing.assert_allclose(pts_tracked_s, desired_pts))
        pts_tracked_t = f_t.track(pts)
        self.assertIsNone(np.testing.assert_allclose(pts_tracked_t, desired_pts,
                                                     atol=1e-6, rtol=1e-6))
        pts_tracked_t = f_t.track(pts, int_out=True)
        self.assertIsNone(np.testing.assert_equal(pts_tracked_t, np.round(desired_pts)))
        self.assertEqual(pts_tracked_t.dtype, int)
        pts_tracked_t, tracked = f_t.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(tracked, True))
        self.assertEqual(pts_tracked_t.dtype, float)

        # Test tracking for 's' flow and int pts (checked via debugger)
        f = Flow.from_transforms([['translation', 10, 20]], (512, 512), 's')
        pts = np.array([[20, 10], [8, 7]])
        desired_pts = [[40, 20], [28, 17]]
        pts_tracked_s = f.track(pts)
        self.assertIsNone(np.testing.assert_equal(pts_tracked_s, desired_pts))

        # Test valid status for 't' flow
        f_t.mask[:, 200:] = False
        pts = np.array([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ])
        desired_valid_status = [False, False, True, True, False]
        _, tracked = f_t.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(tracked, desired_valid_status))

        # Test valid status for 's' flow
        f_s.mask[:, 200:] = False
        pts = np.array([
            [0, 50],            # Moved out of bounds by a valid flow vector
            [0, 500],           # Moved out of bounds by an invalid flow vector
            [8.3, 7.2],         # Moved normally by valid flow vector
            [120.4, 160.2],     # Moved normally by valid flow vector
            [300, 200]          # Moved normally by invalid flow vector
        ])
        desired_valid_status = [False, False, True, True, False]
        _, tracked = f_s.track(pts, get_valid_status=True)
        self.assertIsNone(np.testing.assert_equal(tracked, desired_valid_status))

        # Invalid inputs
        with self.assertRaises(TypeError):
            f_s.track(pts='test')
        with self.assertRaises(ValueError):
            f_s.track(pts=np.eye(3))
        with self.assertRaises(ValueError):
            f_s.track(pts=pts.transpose())
        with self.assertRaises(TypeError):
            f_s.track(pts, int_out='test')
        with self.assertRaises(TypeError):
            f_s.track(pts, True, get_valid_status='test')
        with self.assertRaises(TypeError):
            f_s.track(pts, True, True, s_exact_mode='test')

    def test_matrix(self):
        # Partial affine transform, test reconstruction with all methods
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (1000, 2000), 's')
        flow_t = Flow.from_matrix(matrix, (1000, 2000), 't')
        actual_matrix_s = flow_s.matrix(dof=4, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=4, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix))
        actual_matrix_s = flow_s.matrix(dof=4, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=4, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix))
        actual_matrix_s = flow_s.matrix(dof=6, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=6, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix))
        actual_matrix_s = flow_s.matrix(dof=6, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=6, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix))
        actual_matrix_s = flow_s.matrix(dof=8, method='lms')
        actual_matrix_t = flow_t.matrix(dof=8, method='lms')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix, rtol=1e-8, atol=1e-8))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix, rtol=1e-8, atol=1e-8))
        actual_matrix_s = flow_s.matrix(dof=8, method='ransac')
        actual_matrix_t = flow_t.matrix(dof=8, method='ransac')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix, rtol=1e-8, atol=1e-8))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix, rtol=1e-8, atol=1e-8))
        actual_matrix_s = flow_s.matrix(dof=8, method='lmeds')
        actual_matrix_t = flow_t.matrix(dof=8, method='lmeds')
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_s, matrix, rtol=1e-8, atol=1e-8))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_t, matrix, rtol=1e-8, atol=1e-8))

        # Random matrix, check to see how often an approximate 'reconstruction' fails, target is <5% of cases
        failed = 0
        for i in range(1000):
            matrix = (np.random.rand(3, 3) - .5) * 20
            if -1e-4 < matrix[2, 2] < 1e-4:
                matrix[2, 2] = 0
            else:
                matrix /= matrix[2, 2]
            flow_s = Flow.from_matrix(matrix, (50, 100), 's')
            try:
                np.testing.assert_allclose(flow_s.matrix(8, 'lms'), matrix, atol=1e-2, rtol=1e-2)
                np.testing.assert_allclose(flow_s.matrix(8, 'ransac'), matrix, atol=1e-2, rtol=1e-2)
                np.testing.assert_allclose(flow_s.matrix(8, 'lmeds'), matrix, atol=1e-2, rtol=1e-2)
            except AssertionError:
                failed += 1
        self.assertTrue(failed <= 50)

        # Partial affine transform reconstruction in the presence of noise
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (1000, 2000), 's')
        flow_noise = (np.random.rand(1000, 2000, 2) - .5) * 5
        actual_matrix_4_ransac = (flow_s + flow_noise).matrix(4, 'ransac')
        actual_matrix_4_lmeds = (flow_s + flow_noise).matrix(4, 'lmeds')
        actual_matrix_6_ransac = (flow_s + flow_noise).matrix(6, 'ransac')
        actual_matrix_6_lmeds = (flow_s + flow_noise).matrix(6, 'lmeds')
        actual_matrix_8_lms = (flow_s + flow_noise).matrix(8, 'lms')
        actual_matrix_8_ransac = (flow_s + flow_noise).matrix(8, 'ransac')
        actual_matrix_8_lmeds = (flow_s + flow_noise).matrix(8, 'lmeds')
        rtol = .05
        atol = .05
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_4_ransac, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_4_lmeds, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_6_ransac, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_6_lmeds, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_8_lms, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_8_ransac, matrix, rtol=rtol, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(actual_matrix_8_lmeds, matrix, rtol=rtol, atol=atol))

        # Masked vs non-masked matrix fitting
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        mask = np.zeros((1000, 2000), 'bool')
        mask[:500, :500] = 1  # upper left corner will contain the real values
        flow = Flow.from_matrix(matrix, (1000, 2000), 's', mask)
        random_vecs = (np.random.rand(1000, 2000, 2) - 0.5) * 200
        random_vecs[:500, :500] = flow.vecs[:500, :500]
        flow.vecs = random_vecs
        # Make sure this fails with the 'lmeds' method:
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(flow.matrix(4, 'lmeds', False), matrix)
        # Test that it does NOT fail when the invalid flow elements are masked out
        self.assertIsNone(np.testing.assert_allclose(flow.matrix(4, 'lmeds', True), matrix))

        # Fallback of 'lms' to 'ransac' when dof == 4 or dof == 6
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (1000, 2000), 's')
        actual_matrix_s_lms = flow_s.matrix(dof=4, method='lms')
        actual_matrix_s_ransac = flow_s.matrix(dof=4, method='ransac')
        self.assertIsNone(np.testing.assert_equal(actual_matrix_s_lms, actual_matrix_s_ransac))

        # Invalid inputs
        transforms = [
            ['translation', 20, 10],
            ['rotation', 200, 200, 30],
            ['scaling', 100, 100, 1.1]
        ]
        matrix = matrix_from_transforms(transforms)
        flow_s = Flow.from_matrix(matrix, (1000, 2000), 's')
        with self.assertRaises(ValueError):
            flow_s.matrix(dof='test')
        with self.assertRaises(ValueError):
            flow_s.matrix(dof=5)
        with self.assertRaises(ValueError):
            flow_s.matrix(dof=4, method='test')
        with self.assertRaises(TypeError):
            flow_s.matrix(dof=4, method='lms', masked='test')

    def test_visualise(self):
        # Correct values for the different modes
        # Horizontal flow towards the right is red
        flow = Flow.from_transforms([['translation', 1, 0]], [200, 300])
        desired_img = np.tile(np.array([0, 0, 255]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb'), desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is green
        flow = Flow.from_transforms([['translation', -1, math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([0, 255, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 0], 60))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 2], 255))

        # Flow outwards at the angle of 240 degrees (counter-clockwise) is blue
        flow = Flow.from_transforms([['translation', -1, -math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([255, 0, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb'), desired_img[..., ::-1]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 0], 120))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 2], 255))

        # Show the flow mask
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True)[10, 10], [0, 0, 180]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True)[10, 10], [180, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True)[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True)[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True)[10, 10, 2], 180))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True)[100, 100, 2], 255))

        # Show the flow mask border
        mask = np.zeros((200, 300))
        mask[30:-30, 40:-40] = 1
        flow = Flow.from_transforms([['translation', 1, 0]], (200, 300), 't', mask)
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr', True, True)[30, 40], [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb', True, True)[30, 40], [0, 0, 0]))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True)[..., 0], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True)[30, 40, 1], 0))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv', True, True)[30, 40, 2], 0))

        # Invalid arguments
        flow = Flow.zero([10, 10])
        with self.assertRaises(ValueError):
            flow.visualise(mode=3)
        with self.assertRaises(ValueError):
            flow.visualise(mode='test')
        with self.assertRaises(TypeError):
            flow.visualise('rgb', show_mask=2)
        with self.assertRaises(TypeError):
            flow.visualise('rgb', show_mask_borders=2)
        with self.assertRaises(TypeError):
            flow.visualise('rgb', range_max='2')
        with self.assertRaises(ValueError):
            flow.visualise('rgb', range_max=-1)

    def test_visualise_arrows(self):
        img = cv2.imread('smudge.png')
        mask = np.zeros(img.shape[:2])
        mask[50:-50, 20:-20] = 1
        flow = Flow.from_transforms([['rotation', 256, 256, 30]], img.shape[:2], 's', mask)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(grid_dist='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(grid_dist=-1)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=mask)
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=mask[10:])
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img=img[..., :2])
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, scaling='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img, scaling=-1)
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, show_mask='test')
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, show_mask_borders='test')
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, True, colour='test')
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img, None, True, True, colour=(0, 0))
        with self.assertRaises(TypeError):
            flow.visualise_arrows(10, img, None, True, True, None, thickness=1.5)
        with self.assertRaises(ValueError):
            flow.visualise_arrows(10, img, None, True, True, None, thickness=0)

    def test_show(self):
        flow = Flow.zero([200, 300])
        with self.assertRaises(TypeError):
            flow.show('test')
        with self.assertRaises(ValueError):
            flow.show(-1)

    def test_show_arrows(self):
        flow = Flow.zero([200, 300])
        with self.assertRaises(TypeError):
            flow.show_arrows('test')
        with self.assertRaises(ValueError):
            flow.show_arrows(-1)

    def test_valid_target(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[4:, :3] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[:3, 4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        desired_area_s = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked_consider_mask = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        self.assertIsNone(np.testing.assert_equal(f_s.valid_target(), desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_target(), desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target(), desired_area_s_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_target(False), desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_target(), desired_area_t_masked))

        with self.assertRaises(TypeError):
            f_s.valid_target(consider_mask='test')

    def test_valid_source(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[4:, :3] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[:3, 4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        desired_area_s = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_s_masked = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked_consider_mask = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]
        ]).astype('bool')
        desired_area_t_masked = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0]
        ]).astype('bool')
        self.assertIsNone(np.testing.assert_equal(f_s.valid_source(), desired_area_s))
        self.assertIsNone(np.testing.assert_equal(f_t.valid_source(), desired_area_t))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.valid_source(), desired_area_s_masked))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(), desired_area_t_masked_consider_mask))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.valid_source(False), desired_area_t_masked))

        with self.assertRaises(TypeError):
            f_s.valid_target(consider_mask='test')

    def test_get_padding(self):
        transforms = [['rotation', 0, 0, 45]]
        shape = (7, 7)
        mask = np.ones(shape, 'bool')
        mask[:, 4:] = False
        f_s_masked = Flow.from_transforms(transforms, shape, 's', mask)
        mask = np.ones(shape, 'bool')
        mask[4:] = False
        f_t_masked = Flow.from_transforms(transforms, shape, 't', mask)
        f_s = Flow.from_transforms(transforms, shape, 's')
        f_t = Flow.from_transforms(transforms, shape, 't')
        f_s_desired = [5, 0, 0, 3]
        f_t_desired = [0, 3, 5, 0]
        f_s_masked_desired = [3, 0, 0, 1]
        f_t_masked_desired = [0, 1, 3, 0]
        self.assertIsNone(np.testing.assert_equal(f_s.get_padding(), f_s_desired))
        self.assertIsNone(np.testing.assert_equal(f_t.get_padding(), f_t_desired))
        self.assertIsNone(np.testing.assert_equal(f_s_masked.get_padding(), f_s_masked_desired))
        self.assertIsNone(np.testing.assert_equal(f_t_masked.get_padding(), f_t_masked_desired))

        f = Flow.zero(shape)
        f.vecs[..., 0] = np.random.rand(*shape) * 1e-4
        self.assertIsNone(np.testing.assert_equal(f.get_padding(), [0, 0, 0, 0]))

    def test_is_zero(self):
        shape = (10, 10)
        flow = Flow.zero(shape)
        self.assertEqual(flow.is_zero(thresholded=True), True)
        self.assertEqual(flow.is_zero(thresholded=False), True)

        flow.vecs[:3, :, 0] = 1e-4
        self.assertEqual(flow.is_zero(thresholded=True), True)
        self.assertEqual(flow.is_zero(thresholded=False), False)

        flow.vecs[:3, :, 1] = -1e-3
        self.assertEqual(flow.is_zero(thresholded=True), False)
        self.assertEqual(flow.is_zero(thresholded=False), False)

        transforms = [['rotation', 0, 0, 45]]
        flow = Flow.from_transforms(transforms, shape)
        self.assertEqual(flow.is_zero(thresholded=True), False)
        self.assertEqual(flow.is_zero(thresholded=False), False)

        with self.assertRaises(TypeError):
            flow.is_zero('test')


if __name__ == '__main__':
    unittest.main()
