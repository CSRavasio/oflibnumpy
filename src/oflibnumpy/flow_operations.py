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
# This file is part of oflibnumpy. It contains functions that use the custom flow class defined in flow_class.

import cv2
import numpy as np
from typing import Union
from .flow_class import Flow
from .utils import validate_shape


FlowAlias = 'Flow'
nd = np.ndarray


def visualise_definition(mode: str, shape: Union[list, tuple] = None, insert_text: bool = None) -> np.ndarray:
    """Return an image that shows the definition of the flow visualisation.

    :param mode: Desired output colour space: ``rgb``, ``bgr``, or ``hsv``
    :param shape: List or tuple of shape :math:`(2)` containing the desired image shape as values ``(H, W)``. Defaults
        to (601, 601) - do not change if you leave `insert_text` as ``True`` as otherwise the text will appear in the
        wrong location
    :param insert_text: Boolean determining whether explanatory text is put on the image (using :func:`cv2.putText`),
        defaults to ``True``
    :return: Numpy array of shape :math:`(H, W, 3)` and type ``uint8`` showing the colour definition of the flow
        visualisation
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


def combine_flows(
    input_1: Union[FlowAlias, nd],
    input_2: Union[FlowAlias, nd],
    mode: int,
    ref: str = None,
    thresholded: bool = None
) -> Union[FlowAlias, nd]:
    """Returns the result of the combination of two flow fields of the same shape and reference

    .. tip::
       All of the flow field combinations in this function rely on some combination of the
       :meth:`~oflibnumpy.Flow.apply`, :meth:`~oflibnumpy.Flow.invert`, and :func:`~oflibnumpy.Flow.combine_with`
       methods, and can be very slow (several seconds) due to calling :func:`scipy.interpolate.griddata` multiple
       times. The table below aids decision-making with regards to which reference a flow field should be provided
       in to obtain the fastest result.

        .. list-table:: Calls to :func:`scipy.interpolate.griddata`
           :header-rows: 1
           :stub-columns: 1
           :widths: 10, 15, 15
           :align: center

           * - `mode`
             - ``ref = 's'``
             - ``ref = 't'``
           * - 1
             - 1
             - 3
           * - 2
             - 1
             - 1
           * - 3
             - 0
             - 0

    All formulas used in this function have been derived from first principles. The base formula is
    :math:`flow_1 ⊕ flow_2 = flow_3`, where :math:`⊕` is a non-commutative flow composition operation. This can be
    visualised with the start / end points of the flows as follows:

    .. code-block::

        S = Start point    S1 = S3 ─────── f3 ────────────┐
        E = End point       │                             │
        f = flow           f1                             v
                            └───> E1 = S2 ── f2 ──> E2 = E3

    The main difficulty in combining flow fields is that it would be incorrect to simply add up or subtract flow vectors
    at one location in the flow field area :math:`H \\times W`. This appears to work given e.g. a translation to the
    right, and a translation downwards: the result will be the linear combination of the two vectors, or a translation
    towards the bottom right. However, looking more closely, it becomes evident that this approach isn't actually
    correct: A pixel that has been moved from `S1` to `E1` by the first flow field `f1` is then moved from that location
    by the flow vector of the flow field `f2` that corresponds to the new pixel location `E1`, *not* the original
    location `S1`. If the flow vectors are the same everywhere in the field, the difference will not be noticeable.
    However, if the flow vectors of `f2` vary throughout the field, such as with a rotation around some point, it will!

    In this case (corresponding to calling :func:`combine_flows(f1, f2, mode=3)<combine_flows>`), and if the flow
    reference is ``s`` ("source"), the solution is to first apply the inverse of `f1` to `f2`, essentially
    linking up each location `E1` back to `S1`, and *then* to add up the flow vectors. Analogous observations apply for
    the other permutations of flow combinations and references.

    .. note::
       This is consistent with the observation that two translations are commutative in their application - the order
       does not matter, and the vectors can simply be added up at every pixel location -, while a translation followed
       by a rotation is not the same as a rotation followed by a translation: adding up vectors at each pixel cannot be
       the correct solution as there wouldn't be a difference based on the order of vector addition.

    :param input_1: First input flow as a numpy array. Can also take a flow object, but this will be deprecated soon
    :param input_2: Second input flow, same type as ``input_1``
    :param mode: Integer determining how the input flows are combined, where the number corresponds to the position in
        the formula :math:`flow_1 ⊕ flow_2 = flow_3`:

        - Mode ``1``: `input_1` corresponds to :math:`flow_2`, `input_2` corresponds to :math:`flow_3`, the result will
          be :math:`flow_1`
        - Mode ``2``: `input_1` corresponds to :math:`flow_1`, `input_2` corresponds to :math:`flow_3`, the result will
          be :math:`flow_2`
        - Mode ``3``: `input_1` corresponds to :math:`flow_1`, `input_2` corresponds to :math:`flow_2`, the result will
          be :math:`flow_3`
    :param ref: The reference of the input flow fields, either ``s`` or ``t``
    :param thresholded: Boolean determining whether flows are thresholded during an internal call to
        :meth:`~oflibnumpy.Flow.is_zero`, defaults to ``False``
    :return: Flow object if inputs are flow objects (deprecated in future, avoid), Numpy array as standard
    """

    if isinstance(input_1, Flow) and isinstance(input_2, Flow):
        print("AVOID - future deprecation warning: using combine_flows(flow_obj1, flow_obj2) is deprecated and may "
              "not work anymore in future versions - use flow_obj1.combine_with(flow_obj2) instead. combine_flows() "
              "will be reserved for use with NumPy arrays only.")
        result = input_1.combine_with(input_2, mode=mode, thresholded=thresholded)
        return result

    result = Flow(input_1, ref).combine_with(Flow(input_2, ref), mode=mode, thresholded=thresholded)
    return result.vecs


def switch_flow_ref(flow: nd, input_ref: str) -> nd:
    """Recalculate flow vectors to correspond to a switched flow reference (see Flow reference
    :attr:`~oflibnumpy.Flow.ref`)

    :param flow: The flow field as a numpy array of shape :math:`(H, W, 2)`
    :param input_ref: The reference of the input flow field, either ``s`` or ``t``
    :return: Flow field as a numpy array of shape :math:`(H, W, 2)`
    """

    f = Flow(flow, input_ref).switch_ref()
    return f.vecs


def invert_flow(flow: nd, input_ref: str, output_ref: str = None) -> nd:
    """Inverting a flow: `img`\\ :sub:`1` -- `f` --> `img`\\ :sub:`2` becomes `img`\\ :sub:`1` <-- `f` --
    `img`\\ :sub:`2`. The smaller the input flow, the closer the inverse is to simply multiplying the flow by -1.

    :param flow: The flow field as a numpy array of shape :math:`(H, W, 2)`
    :param input_ref: Reference of the input flow field,  either ``s`` or ``t``
    :param output_ref: Desired reference of the output field, either ``s`` or ``t``. Defaults to ``input_ref``
    :return: Flow field as a numpy array of shape :math:`(H, W, 2)`
    """

    output_ref = input_ref if output_ref is None else output_ref
    f = Flow(flow, input_ref).invert(output_ref)
    return f.vecs


def valid_target(flow: nd, ref: str) -> nd:
    """Find the valid area in the target domain

    Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
    with the flow. The valid area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
    the value in the target img stems from warping a value from the source, and ``False`` where no valid information
    is known.

    Pixels that are ``False`` will often be black (or 'empty') in the warped target image - but not necessarily, due
    to warping artefacts etc. The valid area also allows a distinction between pixels that are black due to no
    actual information being available at this position (validity ``False``), and pixels that are black due to black
    pixel values having been warped to that (valid) location by the flow.

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :return: Boolean numpy array of the same shape :math:`(H, W)` as the flow
    """

    return Flow(flow, ref).valid_target()


def valid_source(flow: nd, ref: str) -> nd:
    """Finds the area in the source domain that will end up being valid in the target domain (see
    :meth:`~oflibnumpy.valid_target`) after warping

    Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
    with the flow. The source area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
    the value in the source will end up somewhere inside the valid target area, and ``False`` where the value in the
    source will either be warped outside of the target image, or not be warped at all due to a lack of valid flow
    vectors connecting to this position.

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :return: Boolean numpy array of the same shape :math:`(H, W)` as the flow
    """

    return Flow(flow, ref).valid_source()


def get_flow_padding(flow: nd, ref: str) -> list:
    """Determine necessary padding from the flow field:

    - When the flow reference is ``t`` ("target"), this corresponds to the padding needed in a source image which
      ensures that every flow vector will find a value in the source domain to warp towards the target domain.
      I.e. any invalid locations in the area :math:`H \\times W` of the target domain (see
      :func:`~oflibnumpy.valid_target`) are purely due to no valid flow vector being available to pull a
      source value to this target location, rather than no source value being available in the first place.
    - When the flow reference is ``s`` ("source"), this corresponds to the padding needed for
      the flow itself, so that applying it to a source image will result in no input image information being lost in
      the warped output, i.e each input image pixel will come to lie inside the padded area.

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :return: A list of shape :math:`(4)` with the values ``[top, bottom, left, right]``
    """

    return Flow(flow, ref).get_padding()


def get_flow_matrix(flow: nd, ref: str, dof: int = None, method: str = None) -> nd:
    """Fit a transformation matrix to the flow field using OpenCV functions

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :param dof: Integer describing the degrees of freedom in the transformation matrix to be fitted, defaults to
        ``8``. Options are:

        - ``4``: Partial affine transform with rotation, translation, scaling
        - ``6``: Affine transform with rotation, translation, scaling, shearing
        - ``8``: Projective transform, i.e estimation of a homography
    :param method: String describing the method used to fit the transformations matrix by OpenCV, defaults to
        ``ransac``. Options are:

        - ``lms``: Least mean squares
        - ``ransac``: RANSAC-based robust method
        - ``lmeds``: Least-Median robust method
    :return: Numpy array of shape :math:`(3, 3)` containing the transformation matrix
    """

    return Flow(flow, ref).matrix(dof=dof, method=method)


def visualise_flow(flow: nd, mode: str, range_max: float = None) -> nd:
    """Visualises the flow as an rgb / bgr / hsv image

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param mode: Output mode, options: ``rgb``, ``bgr``, ``hsv``
    :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
        the flow magnitudes. Defaults to the 99th percentile of the flow field magnitudes
    :return: Numpy array of shape :math:`(H, W, 3)` containing the flow visualisation
    """

    return Flow(flow).visualise(mode=mode, range_max=range_max)


def visualise_flow_arrows(
    flow: nd,
    ref: str,
    grid_dist: int = None,
    img: np.ndarray = None,
    scaling: Union[float, int] = None,
    colour: tuple = None,
    thickness: int = None
) -> nd:
    """Visualises the flow as arrowed lines

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
        ``20``
    :param img: Numpy array with the background image to use (in BGR mode), defaults to white
    :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
        lengths to be equal to twice the grid distance (empirical value)
    :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
        :func:`~oflibnumpy.visualise`
    :param thickness: Integer of the flow arrow thickness, larger than zero. Defaults to ``1``
    :return: Numpy array of shape :math:`(H, W, 3)` containing the flow visualisation, in ``bgr`` colour space
    """

    return Flow(flow, ref).visualise_arrows(grid_dist=grid_dist, img=img, scaling=scaling,
                                            colour=colour, thickness=thickness)


def show_flow(flow: nd, wait: int = None):
    """Shows the flow in an OpenCV window using :func:`~oflibnumpy.visualise`

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
        it will be shown until the window is closed, or the process is terminated
    """

    Flow(flow).show(wait=wait)


def show_flow_arrows(
    flow: nd,
    ref: str,
    wait: int = None,
    grid_dist: int = None,
    img: np.ndarray = None,
    scaling: Union[float, int] = None,
    colour: tuple = None
):
    """Shows the flow in an OpenCV window using :func:`~oflibnumpy.visualise_arrows`

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Reference of the flow field, ``s`` or ``t``
    :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
        it will be shown until the window is closed, or the process is terminated
    :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
        ``20``
    :param img: Numpy array with the background image to use (in BGR colour space), defaults to black
    :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
        lengths to be equal to twice the grid distance (empirical value)
    :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
        :func:`~oflibnumpy.visualise`
    """

    return Flow(flow, ref).show_arrows(wait=wait, grid_dist=grid_dist, img=img, scaling=scaling, colour=colour)
