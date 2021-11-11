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
import cv2
import numpy as np
from oflibnumpy.flow_class import Flow
from oflibnumpy.flow_operations import combine_flows, switch_flow_ref, invert_flow


class TestFlowOperations(unittest.TestCase):
    def test_combine_flows(self):
        img = cv2.imread('smudge.png')
        shape = img.shape[:2]
        transforms = [
            ['rotation', 255.5, 255.5, -30],
            ['scaling', 100, 100, 0.8],
        ]
        for ref in ['s', 't']:
            f1 = Flow.from_transforms(transforms[0:1], shape, ref)
            f2 = Flow.from_transforms(transforms[1:2], shape, ref)
            f3 = Flow.from_transforms(transforms, shape, ref)

            # Mode 1
            f1_actual = combine_flows(f2, f3, 1)
            # f1.show(500, show_mask=True, show_mask_borders=True)
            # f1_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f1_actual, Flow)
            self.assertEqual(f1_actual.ref, ref)
            comb_mask = f1_actual.mask & f1.mask
            self.assertIsNone(np.testing.assert_allclose(f1_actual.vecs[comb_mask], f1.vecs[comb_mask], atol=5e-2))

            # Mode 2
            f2_actual = combine_flows(f1, f3, 2)
            # f2.show(500, show_mask=True, show_mask_borders=True)
            # f2_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f2_actual, Flow)
            self.assertEqual(f2_actual.ref, ref)
            comb_mask = f2_actual.mask & f2.mask
            self.assertIsNone(np.testing.assert_allclose(f2_actual.vecs[comb_mask], f2.vecs[comb_mask], atol=5e-2))

            # Mode 3
            f3_actual = combine_flows(f1, f2, 3)
            # f3.show(500, show_mask=True, show_mask_borders=True)
            # f3_actual.show(show_mask=True, show_mask_borders=True)
            self.assertIsInstance(f3_actual, Flow)
            self.assertEqual(f3_actual.ref, ref)
            comb_mask = f3_actual.mask & f3.mask
            self.assertIsNone(np.testing.assert_allclose(f3_actual.vecs[comb_mask], f3.vecs[comb_mask], atol=5e-2))

    def test_switch_flow_ref(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_equal(flow_s.switch_ref().vecs, switch_flow_ref(flow_s.vecs, 's')))
        self.assertIsNone(np.testing.assert_equal(flow_t.switch_ref().vecs, switch_flow_ref(flow_t.vecs, 't')))

    def test_invert_flow(self):
        shape = [10, 20]
        transforms = [['rotation', 5, 10, 30]]
        flow_s = Flow.from_transforms(transforms, shape, 's')
        flow_t = Flow.from_transforms(transforms, shape, 't')
        self.assertIsNone(np.testing.assert_equal(flow_s.invert().vecs, invert_flow(flow_s.vecs, 's')))
        self.assertIsNone(np.testing.assert_equal(flow_s.invert('t').vecs, invert_flow(flow_s.vecs, 's', 't')))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert().vecs, invert_flow(flow_t.vecs, 't')))
        self.assertIsNone(np.testing.assert_equal(flow_t.invert('s').vecs, invert_flow(flow_t.vecs, 't', 's')))


if __name__ == '__main__':
    unittest.main()
