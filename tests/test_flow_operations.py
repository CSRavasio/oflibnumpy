#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: 2021, Claudio S. Ravasio
# License: LGPL 2.1
# Author: Claudio S. Ravasio, PhD student at University College London (UCl), supervised by:
#   Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
#       Imaging Sciences (BMEIS) at King's College London (KCL)
#   Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK
#
# This file is part of flowlib

import unittest
import numpy as np
import cv2
from scipy.ndimage import rotate, shift
from skimage.metrics import structural_similarity
from flow_lib.flow_class import Flow
from flow_lib.flow_operations import apply_flow


class TestFlowOperations(unittest.TestCase):
    def test_apply(self):
        img = cv2.imread('lena.png')
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['rotation', 255.5, 255.5, -30]], img.shape[:2], ref).vecs
            control_img = rotate(img, -30, reshape=False)
            warped_img = apply_flow(flow, img, ref)
            # Values will not be exactly the same due to rounding etc., so use SSIM instead
            ssim = structural_similarity(control_img, warped_img, multichannel=True)
            print(ssim)
            self.assertTrue(ssim > 0.98)
        for ref in ['t', 's']:
            flow = Flow.from_transforms([['translation', 10, 20]], img.shape[:2], ref).vecs
            control_img = shift(img, [20, 10, 0])
            warped_img = apply_flow(flow, img, ref)
            self.assertIsNone(np.testing.assert_equal(warped_img, control_img))


if __name__ == '__main__':
    unittest.main()
