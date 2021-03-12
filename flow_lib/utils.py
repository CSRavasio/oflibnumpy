import numpy as np
from typing import Union, Any


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

    :param padding: Padding to be checked, should be a list of length 4 of positive integers
    :param error_string: Optional string to be added before the error message
    :return: valid padding list, if indeed valid
    """

    error_string = '' if error_string is None else error_string
    if not isinstance(padding, list):
        raise TypeError(error_string + "Padding needs to be a list [top, bot, left, right]")
    if len(padding) != 4:
        raise ValueError(error_string + "Padding list needs to be a list of length 4 [top, bot, left, right]")
    if not all(isinstance(item, int) for item in padding):
        raise ValueError(error_string + "Padding list [top, bot, left, right] items need to be integers")
    if not all(item > 0 for item in padding):
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


def matrix_from_transform(transform: str, values: list) -> np.ndarray:
    """Calculates a transformation matrix from given transform types and values

    :param transform: Transform type. Options: 'translation', 'rotation', 'scaling'
    :param values: Transform values as list. Options:
        For 'translation':  [<horizontal shift in px>, <vertical shift in px>]
        For 'rotation':     [<horizontal centre in px>, <vertical centre in px>, <angle in degrees, counter-clockwise>]
        For 'scaling':      [<horizontal centre in px>, <vertical centre in px>, <scaling fraction>]
    :return: Transformation matrix, ndarray [3 * 3]
    """
    value = np.array(values)
    matrix = np.identity(3)
    if transform == 'translation':  # translate: value is a list of [horizontal movement, vertical movement]
        matrix[0:2, 2] = value[0], value[1]
    if transform == 'scaling':  # zoom: value is a list of [horizontal coord, vertical coord, scaling]
        translation_matrix_1 = matrix_from_transform('translation', -value[:2])
        translation_matrix_2 = matrix_from_transform('translation', value[:2])
        matrix[0, 0] = value[2]
        matrix[1, 1] = value[2]
        matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    if transform == 'rotation':  # rotate: value is a list of [horizontal coord, vertical coord, rotation in degrees]
        rot = np.radians(value[2])
        translation_matrix_1 = matrix_from_transform('translation', -value[:2])
        translation_matrix_2 = matrix_from_transform('translation', value[:2])
        matrix[0:2, 0:2] = [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]
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
