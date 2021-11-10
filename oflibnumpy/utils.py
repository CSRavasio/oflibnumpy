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


def validate_shape(shape: Any) -> Union[tuple, list]:
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Error creating flow from matrix: Dims need to be a list or a tuple")
    if len(shape) != 2:
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of length 2")
    if any((item <= 0 or not isinstance(item, int)) for item in shape):
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of integers above zero")


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


def apply_flow(flow: np.ndarray, target: np.ndarray, ref: str = None, mask: np.ndarray = None) -> np.ndarray:
    """Warps target according to flow of given reference

    :param flow: Numpy array H-W-2 containing the flow vectors in cv2 convention (1st channel hor, 2nd channel ver)
    :param target: Numpy array H-W or H-W-C containing the content to be warped
    :param ref: Reference of the flow, 't' or 's'. Defaults to 't'
    :param mask: Numpy array H-W containing the flow mask, only relevant for 's' flows. Defaults to True everywhere
    :return: Numpy array of the same shape as the target, with the content warped by the flow
    """

    ref = get_valid_ref(ref)
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
    file = open(path, 'rb')
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
