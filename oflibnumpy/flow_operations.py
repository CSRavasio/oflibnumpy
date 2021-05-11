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

import cv2
import numpy as np
from typing import Union
from .flow_class import Flow
from .utils import validate_shape
from scipy.interpolate import griddata


def visualise_definition(mode: str, shape: Union[list, tuple] = None, insert_text: bool = None) -> np.ndarray:
    """Returns an image that shows the definition of the flow visualisation.

    :param mode: Output mode, options: 'rgb', 'bgr', 'hsv'
    :param shape: List or tuple of the resulting image shape
    :param insert_text: whether explanatory text should be put on the image (using cv2.putText), defaults to True
    :return: Image that shows the colour definition of the flow visualisation
    """

    # Default arguments and input validation
    shape = [601, 601] if shape is None else shape
    validate_shape(shape)
    insert_text = True if insert_text is None else insert_text
    if not isinstance(insert_text, bool):
        raise TypeError("Error visualising the flow definition: Insert_text needs to be a boolean")

    # Creating the flow and getting the flow visualisation
    h, w = shape
    flow = Flow.from_transforms([['scaling', w//2, h//2, 1.1]], shape)
    flow.vecs = (np.abs(flow.vecs) ** 1.2) * np.sign(flow.vecs)
    img = flow.visualise(mode).astype('f')  # dtype 'f' necessary for cv2.arrowedLine

    # Draw on the flow image
    line_colour = (0, 0, 0)
    font_colour = (0, 0, 0)
    cv2.arrowedLine(img, (2, h // 2 + 1), (w - 6, h // 2 + 1), line_colour, 2, tipLength=0.02)
    cv2.arrowedLine(img, (w // 2 + 1, 2), (w // 2 + 1, h - 6), line_colour, 2, tipLength=0.02)

    # Insert explanatory text if required
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    if insert_text:
        cv2.putText(img, 'flow[..., 0]', (450, 285), font, font_scale, font_colour, 1)
        cv2.putText(img, 'flow[..., 1]', (310, 570), font, font_scale, font_colour, 1)
        cv2.putText(img, '[-, -]', (90, 155), font, 1, font_colour)
        cv2.putText(img, '[-, +]', (90, 470), font, 1, font_colour)
        cv2.putText(img, '[+, -]', (400, 155), font, 1, font_colour)
        cv2.putText(img, '[+, +]', (400, 470), font, 1, font_colour)
    return np.round(img).astype('uint8')


# noinspection PyProtectedMember
def combine_flows(input_1: Flow, input_2: Flow, mode: int, thresholded: bool = None) -> Flow:
    """Function that returns the result of flow combinations

    All formulas used in this function have been derived from first principles.

    Base formula: flow_1 ⊕ flow_2 = flow_3, where '⊕' is a non-commutative flow composition operation

    Visualisation with start / end points of the flows:

    .. code-block::

        S = Start point    S1 = S3 ─────── f3 ────────────┐
        E = End point       │                             │
        f = flow           f1                             v
                            └───> E1 = S2 ── f2 ──> E2 = E3

    :param input_1: First input flow object
    :param input_2: Second input flow object
    :param mode: Integer determining how the input flows are combined, where the number corresponds to the position in
        the formula flow_1 ⊕ flow_2 = flow_3, where '⊕' is a non-commutative flow composition operation:
        Mode 1: input_1 corresponds to flow_2, input_2 corresponds to flow_3, the result will be flow_1
        Mode 2: input_1 corresponds to flow_1, input_2 corresponds to flow_3, the result will be flow_2
        Mode 3: input_1 corresponds to flow_1, input_2 corresponds to flow_2, the result will be flow_3
    :param thresholded: Boolean determining whether flows are thresholded when is_zero() is checked, defaults to False
    :return: Resulting flow object
    """

    # Check input validity
    if not isinstance(input_1, Flow) or not isinstance(input_2, Flow):
        raise ValueError("Error combining flows: Inputs need to be of type 'Flow'")
    if not input_1.shape == input_2.shape:
        raise ValueError("Error combining flows: Flow field inputs need to have the same shape")
    if not input_1.ref == input_2.ref:
        raise ValueError("Error combining flows: Flow fields need to have the same reference")
    if mode not in [1, 2, 3]:
        raise ValueError("Error combining flows: Mode needs to be 1, 2 or 3")
    thresholded = False if thresholded is None else thresholded
    if not isinstance(thresholded, bool):
        raise TypeError("Error combining flows: Thresholded needs to be a boolean")

    # Check if one input is zero, return early if so
    if input_1.is_zero(thresholded=thresholded):
        # if mode == 1:  # Flows are in order (desired_result, input_1=0, input_2)
        #     return input_2
        # elif mode == 2:  # Flows are in order (input_1=0, desired_result, input_2)
        #     return input_2
        # elif mode == 3:  # Flows are in order (input_1=0, input_2, desired_result)
        #     return input_2
        # Above code simplifies to:
        return input_2
    elif input_2.is_zero(thresholded=thresholded):
        if mode == 1:  # Flows are in order (desired_result, input_1, input_2=0)
            return input_1.invert()
        elif mode == 2:  # Flows are in order (input_1, desired_result, input_2=0)
            return input_1.invert()
        elif mode == 3:  # Flows are in order (input_1, input_2=0, desired_result)
            return input_1

    result = None
    if mode == 1:  # Flows are in order (desired_result, input_1, input_2)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f1 is (f3 minus f2), when S2 is moved to S3, achieved by applying f2 to move S2 to E3,
            # then inverted(f3) to move from E3 to S3.
            # F1_s = F2_s - combine(F2_s, F3_s^-1_s, 3){F2_s}
            result = input_2 - combine_flows(input_1, input_2.invert(), mode=3).apply(input_1)
        elif input_1._ref == input_2._ref == 't':
            # Explanation: currently no native implementation to ref 't', so just "translated" from ref 's'
            # F1_t = (F2_t-as-s - combine(F2_t, F3_t^-1_t, 3){F2_t-as-s})_as-t
            result = input_2.switch_ref() - combine_flows(input_1, input_2.invert(), mode=3).apply(input_1.switch_ref())
            result = result.switch_ref()
    elif mode == 2:  # Flows are in order (input_1, desired_result, input_2)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f2 is (f3 minus f1), when S1 = S3 is moved to S2, achieved by applying f1
            # F2_s = F1_s{F3_s - F1_s}
            result = input_1.apply(input_2 - input_1)
        elif input_1._ref == input_2._ref == 't':
            # Strictly "translated" version from the ref 's' case:
            # F2_t = F1_t{F3_t-as-s - F1_t-as-s}_as-t)
            # result = (input_1.apply(input_2.switch_ref() - input_1.switch_ref())).switch_ref()

            # Improved version cutting down on operational complexity
            # F3 - F1, where F1 has been resampled to the source positions of F3.
            coord_1 = np.copy(-input_1.vecs)
            coord_1[:, :, 0] += np.arange(coord_1.shape[1])
            coord_1[:, :, 1] += np.arange(coord_1.shape[0])[:, np.newaxis]
            coord_1_flat = np.reshape(coord_1, (-1, 2))
            vecs_with_mask = np.concatenate((input_1.vecs, input_1.mask[..., np.newaxis]), axis=-1)
            vals_flat = np.reshape(vecs_with_mask, (-1, 3))
            coord_3 = np.copy(-input_2.vecs)
            coord_3[:, :, 0] += np.arange(coord_3.shape[1])
            coord_3[:, :, 1] += np.arange(coord_3.shape[0])[:, np.newaxis]
            vals_resampled = griddata(coord_1_flat, vals_flat,
                                      (coord_3[..., 0], coord_3[..., 1]),
                                      method='linear', fill_value=0)
            result = input_2 - Flow(vals_resampled[..., :-1], 't', vals_resampled[..., -1] > .99)
    elif mode == 3:  # Flows are in order (input_1, input_2, desired_result)
        if input_1._ref == input_2._ref == 's':
            # Explanation: f3 is (f1 plus f2), when S2 is moved to S1, achieved by applying inverted(f1)
            # F3_s = F1_s + (F1_s)^-1_t{F2_s}
            result = input_1 + input_1.invert(ref='t').apply(input_2)
        elif input_1._ref == input_2._ref == 't':
            # Explanation: f3 is (f2 plus f1), with f1 pulled towards the f2 grid by applying f2 to f1.
            # F3_t = F2_t + F2_t{F1_t}
            result = input_2 + input_2.apply(input_1)

    return result
