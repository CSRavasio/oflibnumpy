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
# This file is part of oflibnumpy

import cv2
import numpy as np
from typing import Union
from .flow_class import Flow
from .utils import validate_shape


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

