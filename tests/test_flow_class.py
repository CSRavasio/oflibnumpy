import unittest
import numpy as np
import cv2
import math
from flow_lib.flow_class import Flow
from flow_lib.flow_operations import apply_flow


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
        # Invalid transform values
        transforms = 'test'
        with self.assertRaises(TypeError):
            Flow.from_transforms(transforms, dims)
        transforms = ['test']
        with self.assertRaises(TypeError):
            Flow.from_transforms(transforms, dims)
        transforms = [['translation', 20, 10], ['rotation']]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, dims)
        transforms = [['translation', 20, 10], ['rotation', 1]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, dims)
        transforms = [['translation', 20, 10], ['rotation', 1, 'test', 10]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, dims)
        transforms = [['translation', 20, 10], ['test', 1, 1, 10]]
        with self.assertRaises(ValueError):
            Flow.from_transforms(transforms, dims)

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

    def test_getitem(self):
        vectors = np.random.rand(200, 200, 2)
        flow = Flow(vectors)
        indices = np.random.randint(0, 150, size=(20, 2))
        for i in indices:
            # Cutting a number of elements
            self.assertIsNone(np.testing.assert_allclose(flow.vecs[i], vectors[i]))
            # Cutting a specific item
            self.assertIsNone(np.testing.assert_allclose(flow.vecs[i[0], i[1]], vectors[i[0], i[1]]))
            # Cutting an area
            self.assertIsNone(np.testing.assert_allclose(flow.vecs[i[0]:i[0] + 40, i[1]:i[1] + 40],
                                                         vectors[i[0]:i[0] + 40, i[1]:i[1] + 40]))

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
        self.assertIsNone(np.testing.assert_allclose((flow1 + flow2).vecs, vecs1 + vecs2, rtol=1e-6, atol=1e-6))
        self.assertIsNone(np.testing.assert_equal(np.sum((flow1 + flow2).mask), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 + vecs1
        with self.assertRaises(ValueError):
            flow1 + flow3

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
        self.assertIsNone(np.testing.assert_equal(np.sum((flow1 - flow2).mask), (60 - 40) * 200))
        with self.assertRaises(TypeError):
            flow1 - vecs1
        with self.assertRaises(ValueError):
            flow1 - flow3

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
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Divison
        ints = np.random.randint(-10, 10, 100)
        floats = (np.random.rand(100) - .5) * 20
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 / i).vecs, vecs1 / i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 / f).vecs, vecs1 / f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] /= li[0]
            v[..., 1] /= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 / list(li)).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
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
            flow1 / [0, 1, 2]
        # ... using a numpy array of the wrong size
        with self.assertRaises(ValueError):
            flow1 / np.array([0, 1, 2])
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.random.rand(200, 200)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.random.rand(200, 200, 2)
        # ... using a numpy array of the wrong shape
        with self.assertRaises(ValueError):
            flow1 / np.random.rand(200, 200, 2, 1)

    def test_pow(self):
        vecs1 = np.random.rand(100, 200, 2)
        vecs2 = np.random.rand(100, 200, 2)
        flow1 = Flow(vecs1)

        # Divison
        ints = np.random.randint(-2, 2, 100)
        floats = (np.random.rand(100) - .5) * 4
        # ... using ints and floats
        for i, f in zip(ints, floats):
            self.assertIsNone(np.testing.assert_allclose((flow1 ** i).vecs, vecs1 ** i, rtol=1e-6, atol=1e-6))
            self.assertIsNone(np.testing.assert_allclose((flow1 ** f).vecs, vecs1 ** f, rtol=1e-6, atol=1e-6))
        # ... using a list of length 2
        int_list = np.random.randint(-10, 10, (100, 2))
        for li in int_list:
            v = vecs1.astype('f')
            v[..., 0] **= li[0]
            v[..., 1] **= li[1]
            self.assertIsNone(np.testing.assert_allclose((flow1 ** list(li)).vecs, v, rtol=1e-6, atol=1e-6))
        # ... using a numpy array of size 2
        int_list = np.random.randint(-10, 10, (100, 2))
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

    def test_pad(self):
        dims = [100, 80]
        for ref in ['t', 's']:
            flow = Flow.zero(dims, ref, np.ones(dims, 'bool'))
            flow = flow.pad([10, 20, 30, 40])
            self.assertIsNone(np.testing.assert_equal(flow.shape[:2], [dims[0] + 10 + 20, dims[1] + 30 + 40]))
            self.assertIsNone(np.testing.assert_equal(flow.vecs, 0))
            self.assertIsNone(np.testing.assert_equal(flow[10:-20, 30:-40].mask, 1))
            flow.mask[10:-20, 30:-40] = 0
            self.assertIsNone(np.testing.assert_equal(flow.mask, 0))
            self.assertIs(flow.ref, ref)
        # Non-valid padding values
        flow = Flow.zero(dims, ref, np.ones(dims, 'bool'))
        with self.assertRaises(TypeError):
            flow.pad(100)
        with self.assertRaises(ValueError):
            flow.pad([10, 20, 30, 40, 50])
        with self.assertRaises(ValueError):
            flow.pad([10., 20, 30, 40])
        with self.assertRaises(ValueError):
            flow.pad([-10, 10, 10, 10])

    def test_apply(self):
        img = cv2.imread('lena.png')
        # Check flow.apply results in the same as using apply_flow directly
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], ref)
            # Target is a numpy array
            warped_img_desired = apply_flow(flow.vecs, img, ref)
            warped_img_actual = flow.apply(img)
            self.assertIsNone(np.testing.assert_equal(warped_img_actual, warped_img_desired))
            # Target is a flow object
            warped_flow_desired = apply_flow(flow.vecs, flow.vecs, ref)
            warped_flow_actual = flow.apply(flow)
            self.assertIsNone(np.testing.assert_equal(warped_flow_actual.vecs, warped_flow_desired))
        # Check using a smaller flow field on a larger target works the same as a full flow field on the same target
        ref = 't'
        flow = Flow.from_transforms([['rotation', 30, 50, 30]], img.shape[:2], ref)
        warped_img_desired = apply_flow(flow.vecs, img, ref)
        dims = [img.shape[0] - 90, img.shape[1] - 110]
        padding = [50, 40, 30, 80]
        cut_flow = Flow.from_transforms([['rotation', 0, 0, 30]], dims, ref)
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
        target_flow = Flow(img[..., :2])
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

        # Non-valid padding values
        with self.assertRaises(TypeError):
            cut_flow.apply(target_flow, padding=100, cut=True)
        with self.assertRaises(ValueError):
            cut_flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut=True)
        with self.assertRaises(ValueError):
            cut_flow.apply(target_flow, padding=[10., 20, 30, 40], cut=True)
        with self.assertRaises(ValueError):
            cut_flow.apply(target_flow, padding=[-10, 10, 10, 10], cut=True)
        with self.assertRaises(TypeError):
            cut_flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut=2)
        with self.assertRaises(TypeError):
            cut_flow.apply(target_flow, padding=[10, 20, 30, 40, 50], cut='true')

    def test_switch_ref(self):
        img = cv2.imread('lena.png')
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

        # Flow outwards at the andle of 240 degrees (counter-clockwise) is green
        flow = Flow.from_transforms([['translation', -1, math.sqrt(3)]], [200, 300])
        desired_img = np.tile(np.array([0, 255, 0]).reshape((1, 1, 3)), (200, 300, 1))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('bgr'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('rgb'), desired_img))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 0], 60))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 1], 255))
        self.assertIsNone(np.testing.assert_equal(flow.visualise('hsv')[..., 2], 255))

        # Flow outwards at the andle of 240 degrees (counter-clockwise) is green
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
        img = cv2.imread('lena.png')
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


if __name__ == '__main__':
    unittest.main()
