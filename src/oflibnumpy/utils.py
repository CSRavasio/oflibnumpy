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
# This file is part of oflibnumpy. It contains functions needed by the methods of the custom flow class in flow_class.

import math
import cv2
import numpy as np
from typing import Union, Any, List
from scipy.interpolate import griddata


nd = np.ndarray
DEFAULT_THRESHOLD = 1e-3


def get_valid_ref(ref: Any) -> str:
    """Checks flow reference input for validity

    :param ref: Flow reference to be checked
    :return: Valid flow reference, either 't' or 's'
    """

    if ref is None:
        ref = 't'
    else:
        if not isinstance(ref, str):
            raise TypeError("Error setting flow reference: Input is not a string")
        if ref not in ['s', 't']:
            raise ValueError("Error setting flow reference: Input is not 's' or 't', but {}".format(ref))
    return ref


def get_valid_padding(padding: Any, error_string: str = None) -> list:
    """Checks padding input for validity

    :param padding: Padding to be checked, should be a list or tuple of length 4 of positive integers
    :param error_string: Optional string to be added before the error message
    :return: valid padding list, if indeed valid
    """

    error_string = '' if error_string is None else error_string
    if not isinstance(padding, (list, tuple)):
        raise TypeError(error_string + "Padding needs to be a tuple or a list list of values [top, bot, left, right]")
    if len(padding) != 4:
        raise ValueError(error_string + "Padding list needs to be a list or tuple of length 4 [top, bot, left, right]")
    if not all(isinstance(item, int) for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be integers")
    if not all(item >= 0 for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be 0 or larger")
    return padding


def validate_shape(shape: Any):
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Error creating flow from matrix: Dims need to be a list or a tuple")
    if len(shape) != 2:
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of length 2")
    if any((item <= 0 or not isinstance(item, int)) for item in shape):
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of integers above zero")


def validate_flow_array(flow, error_string: str = None) -> nd:
    """Checks flow array for validity, ensures the flow is returned with dtype ``float32``

    :param flow: Flow array to be checked, should be a numpy array of shape :math:`(H, W, 2)` containing floats
    :param error_string: Optional string to be added before the error message
    :return: Flow numpy array with dtype ``float32``
    """

    error_string = '' if error_string is None else error_string
    if not isinstance(flow, np.ndarray):
        raise TypeError(error_string + "Flow is not a numpy array")
    if flow.ndim != 3:
        raise ValueError(error_string + "Flow array is not 3-dimensional")
    if flow.shape[2] != 2:
        raise ValueError(error_string + "Flow array does not have 2 channels")
    if not np.isfinite(flow).all():
        raise ValueError(error_string + "Flow array contains NaN or Inf values")
    return flow.astype('float32')


def flow_from_matrix(matrix: np.ndarray, shape: Union[list, tuple]) -> np.ndarray:
    """Flow calculated from a transformation matrix

    NOTE: This corresponds to a flow with reference 's': based on meshgrid in image 1, warped to image 2, flow vectors
      at each meshgrid point in image 1 corresponding to (warped end points in image 2 - start points in image 1)

    :param matrix: Transformation matrix, numpy array 3-3
    :param shape: List or tuple [H, W] containing required size of the flow field
    :return: Flow field according to cv2 standards, ndarray H-W-2
    """

    # Make default vector field and populate it with homogeneous coordinates
    h, w = shape
    default_vec_hom = np.zeros((h, w, 3), 'f')
    default_vec_hom[..., 0] += np.arange(w)
    default_vec_hom[..., 1] += np.arange(h)[:, np.newaxis]
    default_vec_hom[..., 2] = 1
    # Calculate the flow from the difference of the transformed default vectors, and the original default vector field
    transformed_vec_hom = np.squeeze(np.matmul(matrix, default_vec_hom[..., np.newaxis]))
    transformed_vec = transformed_vec_hom[..., 0:2] / transformed_vec_hom[..., 2, np.newaxis]
    return np.array(transformed_vec - default_vec_hom[..., 0:2], 'float32')


def matrix_from_transforms(transform_list: list) -> np.ndarray:
    """Calculates a transformation matrix from a given list of transforms

    :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
        a list of [transform name, transform value 1, ... , transform value n]. Supported options:
            ['translation', horizontal shift in px, vertical shift in px]
            ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
            ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
    :return: Transformation matrix as numpy array of shape 3-3
    """

    matrix = np.identity(3)
    for transform in reversed(transform_list):
        matrix = matrix @ matrix_from_transform(transform[0], transform[1:])
    return matrix


def matrix_from_transform(transform: str, values: list) -> np.ndarray:
    """Calculates a transformation matrix from given transform types and values

    :param transform: Transform type. Options: 'translation', 'rotation', 'scaling'
    :param values: Transform values as list. Options:
        For 'translation':  [<horizontal shift in px>, <vertical shift in px>]
        For 'rotation':     [<horizontal centre in px>, <vertical centre in px>, <angle in degrees, counter-clockwise>]
        For 'scaling':      [<horizontal centre in px>, <vertical centre in px>, <scaling fraction>]
    :return: Transformation matrix as numpy array of shape 3-3
    """

    matrix = np.identity(3)
    if transform == 'translation':  # translate: value is a list of [horizontal movement, vertical movement]
        matrix[0:2, 2] = values[0], values[1]
    if transform == 'scaling':  # zoom: value is a list of [horizontal coord, vertical coord, scaling]
        translation_matrix_1 = matrix_from_transform('translation', [-values[0], -values[1]])
        translation_matrix_2 = matrix_from_transform('translation', values[:2])
        matrix[0, 0] = values[2]
        matrix[1, 1] = values[2]
        matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    if transform == 'rotation':  # rotate: value is a list of [horizontal coord, vertical coord, rotation in degrees]
        rot = math.radians(values[2])
        translation_matrix_1 = matrix_from_transform('translation', [-values[0], -values[1]])
        translation_matrix_2 = matrix_from_transform('translation', values[:2])
        matrix[0:2, 0:2] = [[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]]
        # NOTE: diff from usual signs in rot matrix [[+, -], [+, +]] results from 'y' axis pointing down instead of up
        matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix


def bilinear_interpolation(data, pts):
    """Fast bilinear interpolation for 2D data

    Copied and adjusted from: https://stackoverflow.com/a/12729229

    :param data: Numpy array of data points to interpolate between, shape H-W-C
    :param pts: Numpy array of points to interpolate at, shape N-2, where '2' is in format [ver, hor]
    :return: Numpy array of interpolation results, shape N-C
    """

    ver, hor = pts[:, 0], pts[:, 1]
    h, w = data.shape[:2]
    if any(~((0 <= ver) & (ver <= h - 1)) | ~((0 <= hor) & (hor <= w - 1))):
        raise IndexError("Some points are outside of the data area.")
    ver0, hor0 = np.floor(ver).astype(int), np.floor(hor).astype(int)
    ver1, hor1 = ver0 + 1, hor0 + 1

    ver0_clipped, hor0_clipped = np.clip(ver0, 0, h - 1), np.clip(hor0, 0, w - 1)
    ver1_clipped, hor1_clipped = np.clip(ver1, 0, h - 1), np.clip(hor1, 0, w - 1)

    w_a = (ver1_clipped - ver) * (hor1_clipped - hor)
    w_b = (ver1_clipped - ver) * (hor - hor0_clipped)
    w_c = (ver - ver0_clipped) * (hor1_clipped - hor)
    w_d = (ver - ver0_clipped) * (hor - hor0_clipped)

    data_a = data[ver0_clipped, hor0_clipped]
    data_b = data[ver1_clipped, hor0_clipped]
    data_c = data[ver0_clipped, hor1_clipped]
    data_d = data[ver1_clipped, hor1_clipped]

    result = w_a[..., np.newaxis] * data_a + \
        w_b[..., np.newaxis] * data_b + \
        w_c[..., np.newaxis] * data_c + \
        w_d[..., np.newaxis] * data_d

    return result


def apply_flow(flow: np.ndarray, target: np.ndarray, ref: str, mask: np.ndarray = None) -> np.ndarray:
    """Uses a given flow to warp a target. The flow reference, if not given, is assumed to be ``t``. Optionally, a mask
    can be passed which (only for flows in ``s`` reference) masks undesired (e.g. undefined or invalid) flow vectors.

    :param flow: Numpy array containing the flow vectors in cv2 convention (1st channel hor, 2nd channel ver), with
        shape :math:`(H, W, 2)`
    :param target: Numpy array containing the content to be warped, with shape :math:`(H, W)` or :math:`(H, W, C)`
    :param ref: Reference of the flow, ``t`` or ``s``
    :param mask: Boolean numpy array containing the flow mask, with shape :math:`(H, W)`. Only relevant for ``s``
        flows. Defaults to ``True`` everywhere
    :return: Numpy array of the same shape :math:`(H, W)` as the target, with the content warped by the flow
    """

    # Input validity check
    ref = get_valid_ref(ref)
    flow = validate_flow_array(flow, "Error applying flow to a target: ")
    if not isinstance(target, np.ndarray):
        raise TypeError("Error applying flow to a target: Target needs to be a numpy array")
    if target.ndim < 2 or target.ndim > 3:
        raise ValueError("Error applying flow to a target: Target array needs to have shape H-W or H-W-C")
    if target.shape[:2] != flow.shape[:2]:
        raise ValueError("Error applying flow to a target: Target height and width needs to match flow field array")
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Error applying flow to a target: Mask needs to be a numpy array")
        if mask.shape != flow.shape[:2]:
            raise ValueError("Error applying flow to a target: Mask height and width needs to match flow field array")
        if mask.dtype != bool:
            raise TypeError("Error applying flow to a target: Mask needs to be boolean")

    field = flow.astype('float32')
    if np.all(np.linalg.norm(flow, axis=-1) <= DEFAULT_THRESHOLD):  # If the flow field is actually 0 or very close
        return target
    if ref == 't':
        field *= -1  # Due to the direction in which cv2.remap defines the flow vectors
        field[:, :, 0] += np.arange(field.shape[1])
        field[:, :, 1] += np.arange(field.shape[0])[:, np.newaxis]
        result = cv2.remap(target, field, None, cv2.INTER_LINEAR)
    else:  # if ref == 's'
        # Get the positions of the unstructured points with known values
        x, y = np.mgrid[:field.shape[0], :field.shape[1]]
        positions = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
        flow_flat = np.reshape(field[..., ::-1], (-1, 2))
        pos = positions + flow_flat
        # Get the known values themselves
        if target.ndim == 3:
            target_flat = np.reshape(target, (-1, target.shape[-1]))
        else:
            target_flat = target.ravel()
        # Mask points, if required
        if mask is not None:
            pos = pos[mask.ravel()]
            target_flat = target_flat[mask.ravel()]
        # Perform interpolation of regular grid from unstructured data
        result = griddata(pos, target_flat, (x, y), method='linear')
        result = np.nan_to_num(result)
        # Make sure the output is returned with the same dtype as the input, if necessary rounded
        if np.issubdtype(target.dtype, np.integer):
            result = np.round(result)
        result = result.astype(target.dtype)
    if result.shape != target.shape:  # target was H-W-1, but e.g. remap returns H-W
        result = result[:, :, np.newaxis]
    return result


def show_masked_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mimics flow.show(), for an input image and a mask

    :param img: Numpy array, BGR input image
    :param mask: Numpy array, boolean mask showing the valid area
    :return: Masked image, in BGR colour space
    """

    hsv = cv2.cvtColor(np.round(img).astype('uint8'), cv2.COLOR_BGR2HSV)
    hsv[np.invert(mask), 2] = 180
    contours, hierarchy = cv2.findContours((255 * mask).astype('uint8'),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hsv, contours, -1, (0, 0, 0), 1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Image masked by valid area", bgr)
    cv2.waitKey()
    return bgr


def points_inside_area(pts: np.ndarray, shape: Union[tuple, list]) -> np.ndarray:
    """Returns an array which is True for all points within image area defined by shape

    :param pts: Numpy array of points of shape N-2
    :param shape: Tuple or list of the image size
    :return: Boolean array True for all points within image area defined by shape
    """

    if np.issubdtype(pts.dtype, float):
        pts = np.round(pts).astype('i')
    status_array = (pts[..., 0] >= 0) & (pts[..., 0] <= shape[0] - 1) & \
                   (pts[..., 1] >= 0) & (pts[..., 1] <= shape[1] - 1)
    return status_array


def threshold_vectors(vecs: np.ndarray, threshold: Union[float, int] = None) -> np.ndarray:
    """Sets all flow vectors with a magnitude below threshold to zero

    :param vecs: Input flow numpy array, shape H-W-2
    :param threshold: Threshold value as float or int, defaults to DEFAULT_THRESHOLD (top of file)
    :return: Flow array with vector magnitudes below the threshold set to 0
    """

    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    mags = np.linalg.norm(vecs, axis=-1)
    f = vecs.copy()
    f[mags < threshold] = 0
    return f


def from_matrix(matrix: np.ndarray, shape: Union[list, tuple], ref: str) -> nd:
    """Flow field array calculated from a transformation matrix

    :param matrix: Transformation matrix to be turned into a flow field, as numpy array of shape :math:`(3, 3)`
    :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
    :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source")
    :return: Flow field, as numpy array
    """

    # Check input validity
    validate_shape(shape)
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array")
    if matrix.shape != (3, 3):
        raise ValueError("Error creating flow from matrix: Matrix needs to be a numpy array of shape (3, 3)")
    ref = get_valid_ref(ref)
    if ref == 's':
        # Coordinates correspond to the meshgrid of the original ('s'ource) image. They are transformed according
        # to the transformation matrix. The start points are subtracted from the end points to yield flow vectors.
        flow_vectors = flow_from_matrix(matrix, shape)
    else:  # ref == 't'
        # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
        # according to the transformation matrix. The end points, which correspond to the flow origin for the
        # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
        flow_vectors = -flow_from_matrix(np.linalg.pinv(matrix), shape)
    return flow_vectors


def from_transforms(transform_list: list, shape: Union[list, tuple], ref: str) -> nd:
    """Flow field array calculated from a list of transforms

    :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
        a list of [``transform name``, ``transform value 1``, ... , ``transform value n``]. Supported options:

        - Transform ``translation``, with values ``horizontal shift in px``, ``vertical shift in px``
        - Transform ``rotation``, with values ``horizontal centre in px``, ``vertical centre in px``,
          ``angle in degrees, counter-clockwise``
        - Transform ``scaling``, with values ``horizontal centre in px``, ``vertical centre in px``,
          ``scaling fraction``
    :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
    :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source")
    :return: Flow field as a numpy array
    """

    # Check input validity
    validate_shape(shape)
    if not isinstance(transform_list, list):
        raise TypeError("Error creating flow from transforms: Transform_list needs to be a list")
    if not all(isinstance(item, list) for item in transform_list):
        raise TypeError("Error creating flow from transforms: Transform_list needs to be a list of lists")
    if not all(len(item) > 1 for item in transform_list):
        raise ValueError("Error creating flow from transforms: Invalid transforms passed")
    for t in transform_list:
        if t[0] == 'translation':
            if not len(t) == 3:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'translation' - expected 2, got {}".format(len(t) - 1))
        elif t[0] == 'rotation':
            if not len(t) == 4:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'rotation' - expected 3, got {}".format(len(t) - 1))
        elif t[0] == 'scaling':
            if not len(t) == 4:
                raise ValueError("Error creating flow from transforms: Not enough transform values passed for "
                                 "'scaling' - expected 3, got {}".format(len(t) - 1))
        else:
            raise ValueError("Error creating flow from transforms: Transform '{}' not recognised".format(t[0]))
        if not all(isinstance(item, (float, int)) for item in t[1:]):
            raise ValueError("Error creating flow from transforms: "
                             "Transform values for '{}' need to be integers or floats".format(t[0]))
    ref = get_valid_ref(ref)

    # Process for flow reference 's' is straightforward: get the transformation matrix for each given transform in
    #   the transform_list, and get the final transformation matrix by multiplying the transformation matrices for
    #   each individual transform sequentially. Finally, call flow_from_matrix to get the corresponding flow field,
    #   which works by applying that final transformation matrix to a meshgrid of vector locations, and subtracting
    #   the start points from the end points.
    #   flow_s = transformed_coords - coords
    #          = final_transform * coords - coords
    #          = t_1 * ... * t_n * coords - coords
    #
    # Process for flow reference 't' can be done in two ways:
    #   1) get the transformation matrix for each given transform in the transform_list, and get the final
    #     transformation matrix by multiplying the transformation matrices for each individual transform in inverse
    #     order. Then, call flow_from_matrix on the *inverse* of this final transformation matrix to get the
    #     negative of the corresponding flow field, which means applying the inverse of that final transformation
    #     matrix to a meshgrid of vector locations, and subtracting the end points from the start points.
    #     flow_t = coords - transformed_coords
    #            = coords - inv(final_transform) * coords
    #            = coords - inv(t_1 * ... * t_n) * coords
    #   2) get the transformation matrix for the reverse of each given transform in the "inverse inverse order",
    #     i.e. in the given order of the transform_list, and get the final transformation matrix by multiplying the
    #     results sequentially. Then, call flow_from_matrix on this final transformation matrix (already
    #     corresponding to the inverse as in method 1)) to get the negative of the corresponding flow field as
    #     before. This method is more complicated, but avoids any numerical issues potentially arising from
    #     calculating the inverse of a matrix.
    #     flow_t = coords - transformed_coords
    #            = coords - final_transform * coords
    #            = coords - inv(t_n) * ... * inv(t_1) * coords
    #     ... because: inv(t_n) * ... * inv(t_1) = inv(t_1 * ... * t_n)

    # Here implemented: method 1, via calling from_matrix where the inverse of the matrix is used if reference 't'
    matrix = matrix_from_transforms(transform_list)
    flow_vectors = from_matrix(matrix, shape, ref)
    return flow_vectors


def load_kitti(path: str) -> Union[List[nd], nd]:
    """Loads the flow field contained in KITTI ``uint16`` png images files, including the valid pixels.
    Follows the official instructions on how to read the provided .png files

    :param path: String containing the path to the KITTI flow data (``uint16``, .png file)
    :return: A numpy array with the KITTI flow data (including valid pixels)
    """

    inp = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED necessary to read uint16 correctly
    if inp is None:
        raise ValueError("Error loading flow from KITTI data: Flow data could not be loaded")
    if inp.ndim != 3 or inp.shape[-1] != 3:
        raise ValueError("Error loading flow from KITTI data: Loaded flow data has the wrong shape")
    inp = inp[..., ::-1].astype('float64')  # Invert channels as cv2 loads as BGR instead of RGB
    inp[..., :2] = (inp[..., :2] - 2 ** 15) / 64
    return inp


def load_sintel(path: str) -> nd:
    """Loads the flow field contained in Sintel .flo byte files. Follows the official instructions provided with
    the Sintel .flo data.

    :param path: String containing the path to the Sintel flow data (.flo byte file, little Endian)
    :return: A numpy array containing the Sintel flow data
    """

    if not isinstance(path, str):
        raise TypeError("Error loading flow from Sintel data: Path needs to be a string")
    with open(path, 'rb') as file:
        if file.read(4).decode('ascii') != 'PIEH':
            raise ValueError("Error loading flow from Sintel data: Path not a valid .flo file")
        w, h = int.from_bytes(file.read(4), 'little'), int.from_bytes(file.read(4), 'little')
        if 99999 < w < 1:
            raise ValueError("Error loading flow from Sintel data: Invalid width read from file ('{}')".format(w))
        if 99999 < h < 1:
            raise ValueError("Error loading flow from Sintel data: Invalid height read from file ('{}')".format(h))
        dt = np.dtype('float32')
        dt = dt.newbyteorder('<')
        flow = np.fromfile(file, dtype=dt).reshape(h, w, 2)
    return flow


def load_sintel_mask(path: str) -> nd:
    """Loads the invalid pixels contained in Sintel .png mask files. Follows the official instructions provided
    with the .flo data.

    :param path: String containing the path to the Sintel invalid pixel data (.png, black and white)
    :return: A numpy array containing the Sintel invalid pixels (mask) data
    """

    if not isinstance(path, str):
        raise TypeError("Error loading flow from Sintel data: Path needs to be a string")
    mask = cv2.imread(path, 0)
    if mask is None:
        raise ValueError("Error loading flow from Sintel data: Invalid mask could not be loaded from path")
    mask = ~(mask.astype('bool'))
    return mask


def resize_flow(flow: nd, scale: Union[float, int, list, tuple]) -> nd:
    """Resize a flow field array, scaling the flow vectors values accordingly

    :param flow: Numpy array containing the flow vectors to be resized, shape :math:`(H, W, 2)`
    :param scale: Scale used for resizing, options:

        - Integer or float of value ``scaling`` applied both vertically and horizontally
        - List or tuple of shape :math:`(2)` with values ``[vertical scaling, horizontal scaling]``
    :return: Scaled flow field as a numpy array
    """

    # Check validity
    flow = validate_flow_array(flow, "Error resizing flow: ")
    if isinstance(scale, (float, int)):
        scale = [scale, scale]
    elif isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError("Error resizing flow: Scale {} must have a length of 2".format(type(scale)))
        if not all(isinstance(item, (float, int)) for item in scale):
            raise ValueError("Error resizing flow: Scale {} items must be integers or floats".format(type(scale)))
    else:
        raise TypeError("Error resizing flow: "
                        "Scale must be an integer, float, or list or tuple of integers or floats")
    if any(s <= 0 for s in scale):
        raise ValueError("Error resizing flow: Scale values must be larger than 0")

    # Resize and adjust values
    resized = cv2.resize(flow, None, fx=scale[1], fy=scale[0])
    resized[..., 0] *= scale[1]
    resized[..., 1] *= scale[0]

    return resized


def is_zero_flow(flow: nd, thresholded: bool = None) -> bool:
    """Check whether all flow vectors are zero. Optionally, a threshold flow magnitude value of ``1e-3`` is used.
    This can be useful to filter out motions that are equal to very small fractions of a pixel, which might just be
    a computational artefact to begin with.

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param thresholded: Boolean determining whether the flow is thresholded, defaults to ``True``
    :return: ``True`` if the flow field is zero everywhere, otherwise ``False``
    """

    # Check input validity
    flow = validate_flow_array(flow, "Error checking whether flow is zero: ")
    thresholded = True if thresholded is None else thresholded
    if not isinstance(thresholded, bool):
        raise TypeError("Error checking whether flow is zero: Thresholded needs to be a boolean")

    f = threshold_vectors(flow) if thresholded else flow
    return np.all(f == 0)


def track_pts(
    flow: nd,
    ref: str,
    pts: nd,
    int_out: bool = None,
    s_exact_mode: bool = None
) -> nd:
    """Warp input points with the flow field, returning the warped point coordinates as integers if required

    .. tip::
        Calling :func:`~oflibnumpy.track_pts` on a flow field with reference ``s`` ("source") is
        significantly faster (as long as `s_exact_mode` is not set to ``True``), as this does not require a call to
        :func:`scipy.interpolate.griddata`.

    :param flow: Flow field as a numpy array of shape :math:`(H, W, 2)`
    :param ref: Flow field reference, either ``s`` or ``t``
    :param pts: Numpy array of shape :math:`(N, 2)` containing the point coordinates. ``pts[:, 0]`` corresponds to
        the vertical coordinate, ``pts[:, 1]`` to the horizontal coordinate
    :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to
        ``False``
    :param s_exact_mode: Boolean determining whether the necessary flow interpolation will be done using
        :func:`scipy.interpolate.griddata`, if the flow has the reference :attr:`~oflibnumpy.Flow.ref` value of
        ``s`` ("source"). Defaults to ``False``, which means a less exact, but around 2 orders of magnitude faster
        bilinear interpolation method will be used. This is recommended for normal point tracking applications.
    :return: Numpy array of warped ('tracked') points, and optionally a numpy array of the point tracking status
    """

    # Validate inputs
    flow = validate_flow_array(flow, "Error tracking points: ")
    if not isinstance(pts, np.ndarray):
        raise TypeError("Error tracking points: Pts needs to be a numpy array")
    if pts.ndim != 2:
        raise ValueError("Error tracking points: Pts needs to have shape N-2")
    if pts.shape[1] != 2:
        raise ValueError("Error tracking points: Pts needs to have shape N-2")
    int_out = False if int_out is None else int_out
    s_exact_mode = False if s_exact_mode is None else s_exact_mode
    if not isinstance(int_out, bool):
        raise TypeError("Error tracking points: Int_out needs to be a boolean")
    if not isinstance(s_exact_mode, bool):
        raise TypeError("Error tracking points: S_exact_mode needs to be a boolean")

    if is_zero_flow(flow, thresholded=True):
        warped_pts = pts
    else:
        if ref == 's':
            if np.issubdtype(pts.dtype, np.integer):
                flow_vecs = flow[pts[:, 0], pts[:, 1], ::-1]
            elif np.issubdtype(pts.dtype, float):
                # Using bilinear_interpolation here is not as accurate as using griddata(), but up to two orders of
                # magnitude faster. Usually, points being tracked will have to be rounded at some point anyway,
                # which means errors e.g. in the order of e-2 (much below 1 pixel) will not have large consequences
                if s_exact_mode:
                    x, y = np.mgrid[:flow.shape[0], :flow.shape[1]]
                    grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
                    flow_flat = np.reshape(flow[..., ::-1], (-1, 2))
                    flow_vecs = griddata(grid, flow_flat, (pts[:, 0], pts[:, 1]), method='linear')
                else:
                    flow_vecs = bilinear_interpolation(flow[..., ::-1], pts)
            else:
                raise TypeError("Error tracking points: Pts numpy array needs to have a float or int dtype")
            warped_pts = pts + flow_vecs
        else:  # self._ref == 't'
            x, y = np.mgrid[:flow.shape[0], :flow.shape[1]]
            grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
            flow_flat = np.reshape(flow[..., ::-1], (-1, 2))
            origin_points = grid - flow_flat
            flow_vecs = griddata(origin_points, flow_flat, (pts[:, 0], pts[:, 1]), method='linear')
            warped_pts = pts + flow_vecs
        nan_vals = np.isnan(warped_pts)
        nan_vals = nan_vals[:, 0] | nan_vals[:, 1]
        warped_pts[nan_vals] = 0
    if int_out:
        warped_pts = np.round(warped_pts).astype('i')

    return warped_pts
