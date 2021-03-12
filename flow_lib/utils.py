import numpy as np
from typing import Union


def get_valid_ref(ref: str = None) -> str:
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


def get_valid_padding(padding: list, error_string: str = None) -> list:
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


def flow_from_matrix(matrix: np.ndarray, shape: Union[list, tuple]) -> np.ndarray:
    """Flow calculated from a transformation matrix

    NOTE: This corresponds to a flow with reference 's': based on meshgrid in image 1, warped to image 2, flow vectors
      at each meshgrid point in image 1 corresponding to (warped end points in image 2 - start points in image 1)

    :param matrix: Transformation matrix, numpy array 3-3
    :param shape: List or tuple [H, W] containing required size of the flow field
    :return: Flow field according to cv2 standards, ndarray H-W-2
    """

    # Check if inputs valid
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Error creating flow from matrix: Dims need to be a list or a tuple")
    if len(shape) != 2:
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of length 2")
    if any((item <= 0 or not isinstance(item, int)) for item in shape):
        raise ValueError("Error creating flow from matrix: Dims need to be a list or a tuple of integers above zero")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array")
    if matrix.shape != (3, 3):
        raise ValueError("Error creating flow from matrix: Matrix needs to be a numpy array of shape (3, 3)")
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
