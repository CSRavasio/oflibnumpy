import unittest
from flow_lib.flow_class import Flow
import numpy as np
import math


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



if __name__ == '__main__':
    unittest.main()
