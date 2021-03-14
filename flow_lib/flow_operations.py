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

import cv2
import numpy as np
from scipy.interpolate import griddata
from .utils import get_valid_ref


def apply_flow(flow: np.ndarray, target: np.ndarray, ref: str = None, mask: np.ndarray = None) -> np.ndarray:
    """Warps target according to flow of given reference

    :param flow: Numpy array H-W-2 containing the flow vectors in cv2 convention (1st channel hor, 2nd channel ver)
    :param target: Numpy array H-W or H-W-C containing the content to be warped
    :param ref: Reference of the flow, 't' or 's'. Defaults to 't'
    :param mask: Numpy array H-W of type 'bool'
    :return: Numpy array of the same shape as the target, with the content warped by the flow
    """

    # TODO: implement / use masks

    ref = get_valid_ref(ref)
    field = flow.astype('float32')
    if np.all(flow == 0):  # If the flow field is actually 0
        return target
    if ref == 't':
        field *= -1  # Due to the direction in which cv2.remap defines the flow vectors
        field[:, :, 0] += np.arange(field.shape[1])
        field[:, :, 1] += np.arange(field.shape[0])[:, np.newaxis]
        result = cv2.remap(target, field, None, cv2.INTER_LINEAR)
        return result
    elif ref == 's':
        x, y = np.mgrid[:field.shape[0], :field.shape[1]]
        positions = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
        flow_flat = np.reshape(field[..., ::-1], (-1, 2))
        if target.ndim == 3:
            target_flat = np.reshape(target, (-1, target.shape[-1]))
        else:
            target_flat = target.ravel()
        pos = positions + flow_flat
        result = griddata(pos, target_flat, (x, y), method='linear')
        result = np.nan_to_num(result)
        # Make sure the output is returned with the same dtype as the input, if necessary rounded
        if np.issubdtype(target.dtype, np.integer):
            result = np.round(result)
        result = result.astype(target.dtype)
        return result
