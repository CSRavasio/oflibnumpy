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

from __future__ import annotations
from typing import Union
import warnings
import cv2
import numpy as np
from scipy.interpolate import griddata
from .utils import get_valid_ref, get_valid_padding, validate_shape, \
    flow_from_matrix, matrix_from_transforms, bilinear_interpolation, apply_flow


class Flow(object):
    def __init__(self, flow_vectors: np.ndarray, ref: str = None, mask: np.ndarray = None):
        """Flow object constructor

        :param flow_vectors: Numpy array H-W-2 containing the flow vector in OpenCV convention: [..., 0] are horizontal,
            [..., 1] are vertical vector components (rather than the numpy vertical first, horizontal second convention)
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        """
        self.vecs = flow_vectors
        self.ref = ref
        self.mask = mask
        self._threshold = 1e-3  # Used for some visualisations

    @property
    def vecs(self) -> np.ndarray:
        """Gets flow vectors

        :return: Flow vectors as numpy array of shape H-W-2 and type 'float32'
        """

        return self._vecs

    @vecs.setter
    def vecs(self, input_vecs: np.ndarray):
        """Sets flow vectors, after checking validity

        :param input_vecs: Numpy array of shape H-W-2
        """

        if not isinstance(input_vecs, np.ndarray):
            raise TypeError("Error setting flow vectors: Input is not a numpy array")
        if not input_vecs.ndim == 3:
            raise ValueError("Error setting flow vectors: Input not 3-dimensional")
        if not input_vecs.shape[2] == 2:
            raise ValueError("Error setting flow vectors: Input does not have 2 channels")
        if not np.isfinite(input_vecs).all():
            raise ValueError("Error setting flow vectors: Input contains NaN, Inf or -Inf values")
        f = input_vecs.astype('float32')
        self._vecs = f

    @property
    def ref(self) -> str:
        """Gets flow reference

        :return: Flow reference 't' or 's'
        """

        return self._ref

    @ref.setter
    def ref(self, input_ref: str = None):
        """Sets flow reference, after checking validity

        :param input_ref: Flow reference 't' or 's'. Defaults to 't'
        """

        self._ref = get_valid_ref(input_ref)

    @property
    def mask(self) -> np.ndarray:
        """Gets flow mask

        :return: Flow mask as numpy array of shape H-W and type 'bool'
        """

        return self._mask

    @mask.setter
    def mask(self, input_mask: np.ndarray = None):
        """Sets flow mask, after checking validity

        :param input_mask: bool numpy array of shape H-W (self.shape), matching flow vectors with shape H-W-2
        """

        if input_mask is None:
            self._mask = np.ones(self.shape[0:2], 'bool')
        else:
            if not isinstance(input_mask, np.ndarray):
                raise TypeError("Error setting flow mask: Input is not a numpy array")
            if not input_mask.ndim == 2:
                raise ValueError("Error setting flow mask: Input not 2-dimensional")
            if not input_mask.shape == self.shape:
                raise ValueError("Error setting flow mask: Input has a different shape than the flow vectors")
            if ((input_mask != 0) & (input_mask != 1)).any():
                raise ValueError("Error setting flow mask: Values must be 0 or 1")
            self._mask = input_mask.astype('bool')

    @property
    def shape(self) -> tuple:
        """Gets shape (resolution) of the flow

        :return: Shape (resolution) of the flow field as a tuple
        """

        return self.vecs.shape[:2]

    @classmethod
    def zero(cls, shape: Union[list, tuple], ref: str = None, mask: np.ndarray = None) -> Flow:
        """Flow object constructor, zero everywhere

        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        return cls(np.zeros((shape[0], shape[1], 2)), ref, mask)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        shape: Union[list, tuple],
        ref: str = None,
        mask: np.ndarray = None
    ) -> Flow:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as Numpy array 3-3
        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Check matrix validity
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Error creating flow from matrix: Matrix needs to be a numpy array")
        if matrix.shape != (3, 3):
            raise ValueError("Error creating flow from matrix: Matrix needs to be a numpy array of shape (3, 3)")

        ref = get_valid_ref(ref)
        if ref == 's':
            # Coordinates correspond to the meshgrid of the original ('s'ource) image. They are transformed according
            # to the transformation matrix. The start points are subtracted from the end points to yield flow vectors.
            flow_vectors = flow_from_matrix(matrix, shape)
            return cls(flow_vectors, ref, mask)
        elif ref == 't':
            # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
            # according to the transformation matrix. The end points, which correspond to the flow origin for the
            # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
            flow_vectors = -flow_from_matrix(np.linalg.pinv(matrix), shape)
            return cls(flow_vectors, ref, mask)

    @classmethod
    def from_transforms(
        cls,
        transform_list: list,
        shape: Union[list, tuple],
        ref: str = None,
        mask: np.ndarray = None
    ) -> Flow:
        """Flow object constructor, zero everywhere.

        :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
            a list of [transform name, transform value 1, ... , transform value n]. Supported options:
                ['translation', horizontal shift in px, vertical shift in px]
                ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
                ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
        :param shape: List or tuple [H, W] of flow field shape
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

        # Check shape validity
        validate_shape(shape)
        # Check transform_list validity
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
        ref = get_valid_ref(ref)
        matrix = matrix_from_transforms(transform_list)
        return cls.from_matrix(matrix, shape, ref, mask)

    def __str__(self):
        info_string = "Flow object, reference {}, shape {}*{}; ".format(self.ref, *self.shape)
        info_string += self.__repr__()
        return info_string

    def __getitem__(self, item: Union[int, list, slice]) -> Flow:
        """Mimics __getitem__ of a numpy array, returning a flow object cut accordingly

        Will throw an error if mask.__getitem__(item) or vecs.__getitem__(item) throw an error

        :param item: Slice used to select a part of the flow
        :return: New flow cut as a corresponding numpy array would be cut
        """

        return Flow(self.vecs.__getitem__(item), self.ref, self.mask.__getitem__(item))

    def __copy__(self) -> Flow:
        """Returns a copy of the flow object

        :return: Copy of the flow object
        """

        return Flow(self.vecs, self.ref, self.mask)

    def __add__(self, other: Union[np.ndarray, Flow]) -> Flow:
        """Adds a flow object or a numpy array to a flow object

        Note: this is NOT equal to applying the two flows sequentially. For that, use combine_flows(flow1, flow2, None).
        The function also does not check whether the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object or numpy array corresponding to the addend. Adding a flow object will adjust the mask
            of the resulting flow object to correspond to the logical union of the augend / addend masks
        :return: Flow object corresponding to the sum
        """

        if not isinstance(other, (np.ndarray, Flow)):
            raise TypeError("Error adding to flow: Addend is not a flow object or a numpy array")
        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error adding to flow: Augend and addend flow objects are not the same shape")
            else:
                vecs = self.vecs + other.vecs
                mask = np.logical_and(self.mask, other.mask)
                return Flow(vecs, self.ref, mask)
        if isinstance(other, np.ndarray):
            if self.shape != other.shape[:2] or other.ndim != 3 or other.shape[2] != 2:
                raise ValueError("Error adding to flow: Addend numpy array needs to have the same shape as the flow "
                                 "object, 3 dimensions overall, and a channel length of 2")
            else:
                vecs = self.vecs + other
                return Flow(vecs, self.ref, self.mask)

    def __sub__(self, other: Union[np.ndarray, Flow]) -> Flow:
        """Subtracts a flow objects or a numpy array from a flow object

        Note: this is NOT equal to subtracting the effects of applying flow fields to an image. For that, used
        combine_flows(flow1, None, flow2) or combine_flows(None, flow1, flow2). The function also does not check whether
        the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object or numpy array corresponding to the subtrahend. Subtracting a flow object will adjust
            the mask of the resulting flow object to correspond to the logical union of the minuend / subtrahend masks
        :return: Flow object corresponding to the difference
        """

        if not isinstance(other, (np.ndarray, Flow)):
            raise TypeError("Error subtracting from flow: Subtrahend is not a flow object or a numpy array")
        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error subtracting from flow: "
                                 "Minuend and subtrahend flow objects are not the same shape")
            else:
                vecs = self.vecs - other.vecs
                mask = np.logical_and(self.mask, other.mask)
                return Flow(vecs, self.ref, mask)
        if isinstance(other, np.ndarray):
            if self.shape != other.shape[:2] or other.ndim != 3 or other.shape[2] != 2:
                raise ValueError("Error subtracting from flow: Subtrahend numpy array needs to have the same shape as "
                                 "the flow object, 3 dimensions overall, and a channel length of 2")
            else:
                vecs = self.vecs - other
                return Flow(vecs, self.ref, self.mask)

    def __mul__(self, other: Union[float, int, bool, list, np.ndarray]) -> Flow:
        """Multiplies a flow object

        :param other: Multiplier: can be converted to float or is a list length 2, an array of the same shape as the
            flow object, or an array of the same shape as the flow vectors
        :return: Flow object corresponding to the product
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self.vecs * float(other), self.ref, self.mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error multiplying flow: Multiplier list not length 2")
                return Flow(self.vecs * np.array(other)[np.newaxis, np.newaxis, :], self.ref, self.mask)
            elif isinstance(other, np.ndarray):
                if other.ndim == 1 and other.size == 2:
                    other = other[np.newaxis, np.newaxis, :]
                elif other.ndim == 2 and other.shape == self.shape[:2]:
                    other = other[:, :, np.newaxis]
                elif other.shape == self.shape + (2,):
                    pass
                else:
                    raise ValueError("Error multiplying flow: Multiplier array is not one of the following: size 2, "
                                     "shape of the flow object, shape of the flow vectors")
                return Flow(self.vecs * other, self.ref, self.mask)
            else:
                raise TypeError("Error multiplying flow: Multiplier cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __truediv__(self, other: Union[float, int, bool, list, np.ndarray]) -> Flow:
        """Divides a flow object

        :param other: Divisor: can be converted to float or is a list length 2, an array of the same shape as the flow
            object, or an array of the same shape as the flow vectors
        :return: Flow object corresponding to the quotient
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self.vecs / float(other), self.ref, self.mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error dividing flow: Divisor list not length 2")
                return Flow(self.vecs / np.array(other)[np.newaxis, np.newaxis, :], self.ref, self.mask)
            elif isinstance(other, np.ndarray):
                if other.ndim == 1 and other.size == 2:
                    other = other[np.newaxis, np.newaxis, :]
                elif other.ndim == 2 and other.shape == self.shape[:2]:
                    other = other[:, :, np.newaxis]
                elif other.shape == self.shape + (2,):
                    pass
                else:
                    raise ValueError("Error dividing flow: Divisor array is not one of the following: size 2, "
                                     "shape of the flow object, shape of the flow vectors")
                return Flow(self.vecs / other, self.ref, self.mask)
            else:
                raise TypeError("Error dividing flow: Divisor cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __pow__(self, other: Union[float, int, bool, list, np.ndarray]) -> Flow:
        """Exponentiates a flow object

        :param other: Exponent: can be converted to float or is a list length 2, an array of the same shape as the flow
            object, or an array of the same shape as the flow vectors
        :return: Flow object corresponding to the power
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self.vecs ** float(other), self.ref, self.mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error exponentiating flow: Exponent list not length 2")
                return Flow(self.vecs ** np.array(other)[np.newaxis, np.newaxis, :], self.ref, self.mask)
            elif isinstance(other, np.ndarray):
                if other.ndim == 1 and other.size == 2:
                    other = other[np.newaxis, np.newaxis, :]
                elif other.ndim == 2 and other.shape == self.shape[:2]:
                    other = other[:, :, np.newaxis]
                elif other.shape == self.shape + (2,):
                    pass
                else:
                    raise ValueError("Error exponentiating flow: Exponent array is not one of the following: size 2, "
                                     "shape of the flow object, shape of the flow vectors")
                return Flow(self.vecs ** other, self.ref, self.mask)
            else:
                raise TypeError("Error exponentiating flow: Exponent cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __neg__(self) -> Flow:
        """Returns the negative of a flow object

        CAREFUL: this is NOT equal to correctly inverting a flow! For that, use invert().

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :return: Negative flow
        """

        return self * -1

    def pad(self, padding: list = None, mode: str = None) -> Flow:
        """Pads the flow with the given padding, inserting 0 values

        :param padding: [top, bot, left, right] list of padding values
        :param mode: Numpy padding mode for the flow vectors, defaults to 'constant'. Options:
            'constant', 'edge', 'symmetric' (see numpy.pad documentation). 'Constant' value is 0.
        :return: Padded flow
        """

        mode = 'constant' if mode is None else mode
        if mode not in ['constant', 'edge', 'symmetric']:
            raise ValueError("Error padding flow: Mode should be one of "
                             "'constant', 'edge', 'symmetric', 'empty', but instead got '{}'".format(mode))
        padding = get_valid_padding(padding, "Error padding flow: ")
        padded_vecs = np.pad(self.vecs, (tuple(padding[:2]), tuple(padding[2:]), (0, 0)), mode=mode)
        padded_mask = np.pad(self.mask, (tuple(padding[:2]), tuple(padding[2:])))
        return Flow(padded_vecs, self.ref, padded_mask)

    def apply(
        self,
        target: Union[np.ndarray, Flow],
        return_valid_area: bool = None,
        padding: list = None,
        cut: bool = None
    ) -> Union[np.ndarray, Flow]:
        """Applies the flow to the target, which can be a numpy array or a Flow object.

        :param target: Numpy array of shape H-W-C or flow object the flow should be applied to
        :param return_valid_area: Boolean determining whether a boolean numpy array of shape H-W containing the valid
            image area is returned (only relevant if target is a numpy array). This array is true where the image
            values in the function output:
                1) have been affected by flow vectors: always true if the flow has reference 't' as the target image by
                    default has a corresponding flow vector in each position, but only true for some parts of the image
                    if the flow has reference 's': some target image positions would only be reachable by flow vectors
                    originating outside of the source image area, which is obviously impossible
                2) have been affected by flow vectors that were themselves valid, as determined by the flow mask
        :param padding: If flow applied only covers part of the target; [top, bot, left, right]; default None
        :param cut: If padding is given, whether the input is returned as cut to shape of flow; default True
        :return: An object of the same type as the input (numpy array, or flow)
        """

        cut = False if cut is None else cut
        if not isinstance(cut, bool):
            raise TypeError("Error applying flow: Cut needs to be a boolean")
        if padding is not None:
            padding = get_valid_padding(padding, "Error applying flow: ")
            if self.shape[0] + np.sum(padding[:2]) != target.shape[0] or \
                    self.shape[1] + np.sum(padding[2:]) != target.shape[1]:
                raise ValueError("Error applying flow: Padding values do not match flow and target shape difference")

        # Type check, prepare arrays
        if isinstance(target, Flow):
            return_flow = True
            t = target.vecs
            mask = target.mask[..., np.newaxis]
        else:
            return_flow = False
            if not isinstance(target, np.ndarray):
                raise ValueError("Error applying flow: Target needs to be either a flow object, or a numpy ndarray")
            t = target
            mask = np.ones(t.shape[:2] + (1,), 'b')

        # Concatenate the flow vectors with the mask if required, so they are warped in one step
        if return_flow or return_valid_area:
            # if self.ref == 't': Just warp the mask, which self.vecs are valid taken into account after warping
            if self.ref == 's':
                # Warp the target mask after ANDing with flow mask to take into account which self.vecs are valid
                if mask.shape[:2] != self.shape:
                    # If padding in use, mask can be smaller than self.mask
                    tmp = mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1], 0].copy()
                    mask[...] = False
                    mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1], 0] = \
                        tmp & self.mask
                else:
                    mask = (mask & self.mask)
            t = np.concatenate((t, mask), axis=-1)

        # Determine flow to use for warping, and warp
        if padding is None:
            if not target.shape[:2] == self.shape[:2]:
                raise ValueError("Error applying flow: Flow and target have to have the same shape")
            warped_t = apply_flow(self.vecs, t, self.ref)
        else:
            mode = 'constant' if self.ref == 't' else 'edge'
            # Note: this mode is very important: irrelevant for flow with reference 't' as this by definition covers
            # the area of the target image, so 'constant' (defaulting to filling everything with 0) is fine. However,
            # for flows with reference 's', if locations in the source image with some flow vector border padded
            # locations with flow zero, very strange interpolation artefacts will result, both in terms of the image
            # being warped, and the mask being warped. By padding with the 'edge' mode, large gradients in flow vector
            # values at the edge of the original flow area are avoided, as are interpolation artefacts.
            flow = self.pad(padding, mode=mode)
            warped_t = apply_flow(flow.vecs, t, flow.ref)

        # Cut if necessary
        if padding is not None and cut:
            warped_t = warped_t[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]]

        # Extract and finalise mask if required
        if return_flow or return_valid_area:
            mask = np.round(warped_t[..., -1]).astype('bool')
            # if self.ref == 's': Valid self.vecs already taken into account by ANDing with self.mask before warping
            if self.ref == 't':
                # Still need to take into account which self.vecs are actually valid by ANDing with self.mask
                if mask.shape != self.mask.shape:
                    # If padding is in use, but warped_t has not been cut: AND with self.mask inside the flow area, and
                    # set everything else to False as not warped by the flow
                    t = mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]].copy()
                    mask[...] = False
                    mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]] = t & self.mask
                else:
                    mask = mask & self.mask

        # Return as correct type
        if return_flow:
            return Flow(warped_t[..., :2], target.ref, mask)
        else:
            if return_valid_area:
                return warped_t[..., :-1], mask
            else:
                return warped_t

    def switch_ref(self, mode: str = None) -> Flow:
        """Switches the reference coordinates from 's'ource to 't'arget, or vice versa

        :param mode: 'valid' or 'invalid':
            'invalid' means just the flow reference attribute is switched without any flow values being changed. This
                is functionally equivalent to simply using flow.ref = 't' for a flow of ref 's', and the flow vectors
                aren't changed.
            'valid' means actually switching the flow field to the other coordinate reference, with flow vectors being
                recalculated to correspond to this other reference.
        :return: Flow with switched coordinate reference.
        """

        mode = 'valid' if mode is None else mode
        if mode == 'valid':
            if np.all(self.vecs == 0):  # In case the flow is 0, no further calculations are necessary
                return self.switch_ref(mode='invalid')
            else:
                if self.ref == 's':
                    switched_ref_flow = self.apply(self)  # apply_to is done s-based; see window pic 08/04/19
                    switched_ref_flow.ref = 't'
                    return switched_ref_flow
                elif self.ref == 't':
                    flow_copy_s = self.switch_ref(mode='invalid')  # so apply_to is ref-s; see window pic 08/04/19
                    return (-flow_copy_s).apply(flow_copy_s)
        elif mode == 'invalid':
            if self.ref == 's':
                return Flow(self.vecs, 't', self.mask)
            elif self.ref == 't':
                return Flow(self.vecs, 's', self.mask)
        else:
            raise ValueError("Error switching flow reference: Mode not recognised, should be 'valid' or 'invalid'")

    def invert(self, ref: str = None) -> Flow:
        """Inverting a flow: img1 -- f --> img2 becomes img1 <-- f -- img2

        The smaller the input flow, the closer the inverse is to simply multiplying the flow by -1.

        :param ref: Desired reference of the output field, defaults to reference of original flow field
        :return: Inverse flow field
        """

        ref = self.ref if ref is None else get_valid_ref(ref)
        if self.ref == 's':
            if ref == 's':
                return self.apply(-self)
            elif ref == 't':
                return Flow(-self.vecs, 't', self.mask)
        elif self.ref == 't':
            if ref == 's':
                return Flow(-self.vecs, 's', self.mask)
            elif ref == 't':
                return self.invert('s').switch_ref()

    def track(
            self,
            pts: np.ndarray,
            int_out: bool = None,
            get_tracked: bool = None,
            s_exact_mode: bool = None
    ) -> np.ndarray:
        """Warps input points according to the flow field, can be returned as integers if required

        :param pts: Numpy array of points shape N-2, 1st coordinate vertical (height), 2nd coordinate horizontal (width)
        :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to False
        :param get_tracked: Boolean determining whether a numpy array containing tracked points is returned. Array will
            be True for points inside the flow area, False if outside. Points ending up outside the flow area means
            the points have been 'lost' and can no longer be tracked.
        :param s_exact_mode: Boolean determining whether interpolation will be done bilinearly if the flow has reference
            's', using bilinear_interpolation. Unless a very large number of points is tracked, this is around 2 orders
            of magnitude faster than using s_exact_mode = True, which will use scipy.interpolate.griddata.
            Defaults to False
        :return: Numpy array of warped ('tracked') points
        """

        # Validate inputs
        if not isinstance(pts, np.ndarray):
            raise TypeError("Error tracking points: Pts needs to be a numpy array")
        if pts.ndim != 2:
            raise ValueError("Error tracking points: Pts needs to have shape N-2")
        if pts.shape[1] != 2:
            raise ValueError("Error tracking points: Pts needs to have shape N-2")
        int_out = False if int_out is None else int_out
        get_tracked = False if get_tracked is None else get_tracked
        s_exact_mode = False if s_exact_mode is None else s_exact_mode
        if not isinstance(int_out, bool):
            raise TypeError("Error tracking points: Int_out needs to be a boolean")
        if not isinstance(get_tracked, bool):
            raise TypeError("Error tracking points: Get_tracked needs to be a boolean")
        if not isinstance(s_exact_mode, bool):
            raise TypeError("Error tracking points: S_exact_mode needs to be a boolean")

        pts = pts.astype('f')
        warped_pts, nan_vals = None, None
        if np.all(self.vecs == 0):
            warped_pts = pts
        else:
            if self.ref == 's':
                if np.issubdtype(pts.dtype, np.integer):
                    flow_vecs = self.vecs[pts[:, 0], pts[:, 1], ::-1]
                elif np.issubdtype(pts.dtype, float):
                    # Using bilinear_interpolation here is not as accurate as using griddata(), but up to two orders of
                    # magnitude faster. Usually, points being tracked will have to be rounded at some point anyway,
                    # which means errors e.g. in the order of e-2 (much below 1 pixel) will not have large consequences
                    if s_exact_mode:
                        x, y = np.mgrid[:self.shape[0], :self.shape[1]]
                        grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
                        flow_flat = np.reshape(self.vecs[..., ::-1], (-1, 2))
                        flow_vecs = griddata(grid, flow_flat, (pts[:, 0], pts[:, 1]), method='linear')
                    else:
                        flow_vecs = bilinear_interpolation(self.vecs[..., ::-1], pts)
                else:
                    raise TypeError("Error tracking points: Pts numpy array needs to have a float or int dtype")
                warped_pts = pts + flow_vecs
            if self.ref == 't':
                x, y = np.mgrid[:self.shape[0], :self.shape[1]]
                grid = np.swapaxes(np.vstack([x.ravel(), y.ravel()]), 0, 1)
                flow_flat = np.reshape(self.vecs[..., ::-1], (-1, 2))
                origin_points = grid - flow_flat
                flow_vecs = griddata(origin_points, flow_flat, (pts[:, 0], pts[:, 1]), method='linear')
                warped_pts = pts + flow_vecs
            nan_vals = np.isnan(warped_pts)
            nan_vals = nan_vals[:, 0] | nan_vals[:, 1]
            warped_pts[nan_vals] = 0
        if int_out:
            warped_pts = np.round(warped_pts).astype('i')
        if get_tracked:
            tracked_pts = (warped_pts[..., 0] >= 0) & (warped_pts[..., 0] <= self.shape[0] - 1) &\
                          (warped_pts[..., 1] >= 0) & (warped_pts[..., 1] <= self.shape[1] - 1) & \
                          ~nan_vals
            return warped_pts, tracked_pts
        else:
            return warped_pts

    def matrix(self, dof: int = None, method: str = None, masked: bool = None) -> np.ndarray:
        """Fits a transformation matrix to the flow field using OpenCV functions

        :param dof: Int describing the degrees of freedom in the transformation matrix fitted, defaults to 8. Options:
            4: Partial affine transform: rotation, translation, scaling
            6: Affine transform: rotation, translation, scaling, shearing
            8: Projective transform, estimates a homography
        :param method: Method used to fit the transformations matrix by OpenCV, defaults to 'ransac'. Options:
            'lms': Least mean squares
            'ransac': RANSAC-based robust method
            'lmeds': Least-Median robust method
        :param masked: Boolean determining whether the flow mask is used to ignore flow locations where the mask is
            False. Defaults to True
        :return: Numpy array of shape 3-3 of the transformation matrix
        """

        # Input validation
        dof = 8 if dof is None else dof
        if dof not in [4, 6, 8]:
            raise ValueError("Error fitting transformation matrix to flow: Dof needs to be 4, 6 or 8")
        method = 'ransac' if method is None else method
        if method not in ['lms', 'ransac', 'lmeds']:
            raise ValueError("Error fitting transformation matrix to flow: "
                             "Method needs to be 'lms', 'ransac', or 'lmeds'")
        masked = True if masked is None else masked
        if not isinstance(masked, bool):
            raise TypeError("Error fitting transformation matrix to flow: Masked needs to be boolean")

        # Get the two point arrays
        if self.ref == 't':
            dst_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            src_pts = dst_pts - self.vecs
        else:  # ref is 's'
            src_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            dst_pts = src_pts + self.vecs
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)

        # Mask if required
        if masked:
            src_pts = src_pts[self.mask.ravel()]
            dst_pts = dst_pts[self.mask.ravel()]

        if dof in [4, 6] and method == 'lms':
            method = 'ransac'
            warnings.warn("Method 'lms' (least mean squares) not supported for fitting a transformation matrix with 4 "
                          "or 6 degrees of freedom to the flow - defaulting to 'ransac'")

        dof_lookup = {
            4: cv2.estimateAffinePartial2D,
            6: cv2.estimateAffine2D,
            8: cv2.findHomography
        }

        method_lookup = {
            'lms': 0,
            'ransac': cv2.RANSAC,
            'lmeds': cv2.LMEDS
        }

        # Fit matrix
        if dof in [4, 6]:
            matrix = np.eye(3)
            matrix[:2] = dof_lookup[dof](src_pts, dst_pts, method=method_lookup[method])[0]
        else:
            matrix = dof_lookup[dof](src_pts, dst_pts, method=method_lookup[method])[0]
        return matrix

    def visualise(
        self,
        mode: str,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        range_max: float = None
    ) -> np.ndarray:
        """Returns a flow visualisation as a numpy array containing an rgb / bgr / hsv img of the same shape as the flow

        :param mode: Output mode, options: 'rgb', 'bgr', 'hsv'
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
            the flow magnitudes. Defaults to the 99th percentile of the current flow field
        :return: Numpy array containing the flow visualisation as an rgb / bgr / hsv image of the same shape as the flow
        """

        show_mask = False if show_mask is None else show_mask
        show_mask_borders = False if show_mask_borders is None else show_mask_borders
        if not isinstance(show_mask, bool):
            raise TypeError("Error visualising flow: Show_mask needs to be boolean")
        if not isinstance(show_mask_borders, bool):
            raise TypeError("Error visualising flow: Show_mask_borders needs to be boolean")

        f = self.vecs.copy()  # Necessary, as otherwise the flow outside this function can be affected (not immutable)

        # Threshold the flow: very small numbers can otherwise lead to issues when calculating mag / angle
        f[(-self._threshold < f) & (f < self._threshold)] = 0

        # Colourise the flow
        hsv = np.zeros((f.shape[0], f.shape[1], 3), 'f')
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1], angleInDegrees=True)
        hsv[..., 0] = ang / 2
        hsv[..., 2] = 255

        # Add mask if required
        if show_mask:
            hsv[np.invert(self.mask), 2] = 180

        # Scale flow
        if range_max is None:
            if np.percentile(mag, 99) > 0:  # Use 99th percentile to avoid extreme outliers skewing the scaling
                range_max = np.percentile(mag, 99)
            elif np.max(mag):  # If the 99th percentile is 0, use the actual maximum instead
                range_max = np.max(mag)
            else:  # If the maximum is 0 too (i.e. the flow field is entirely 0)
                range_max = 1
        if not isinstance(range_max, (float, int)):
            raise TypeError("Error visualising flow: Range_max needs to be an integer or a float")
        if range_max <= 0:
            raise ValueError("Error visualising flow: Range_max needs to be larger than zero")
        hsv[..., 1] = np.clip(mag * 255 / range_max, 0, 255)

        # Add mask borders if required
        if show_mask_borders:
            contours, hierarchy = cv2.findContours((255 * self.mask).astype('uint8'),
                                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(hsv, contours, -1, (0, 0, 0), 1)

        # Process and return the flow visualisation
        if mode == 'hsv':
            return np.round(hsv).astype('uint8')
        elif mode == 'rgb' or mode == 'bgr':
            h = hsv[..., 0] / 180
            s = hsv[..., 1] / 255
            v = hsv[..., 2] / 255
            # Credit to stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
            i = np.int_(h * 6.)
            f = h * 6. - i
            i = np.ravel(i)
            t = np.ravel(1. - f)
            f = np.ravel(f)
            i %= 6
            c_list = (1 - np.ravel(s) * np.vstack([np.zeros_like(f), np.ones_like(f), f, t])) * np.ravel(v)
            # 0:v 1:p 2:q 3:t
            order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
            rgb = c_list[order[i], np.arange(np.prod(h.shape))[:, None]].reshape(*h.shape, 3)
            rgb = np.round(rgb * 255).astype('uint8')
            if mode == 'bgr':
                return rgb[..., ::-1]
            else:
                return rgb
        else:
            raise ValueError("Error visualising flow: Mode needs to be either 'bgr', 'rgb', or 'hsv'")

    def visualise_arrows(
        self,
        grid_dist: int,
        img: np.ndarray = None,
        scaling: Union[float, int] = None,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        colour: tuple = None
    ) -> np.ndarray:
        """Visualises the flow as arrowed lines, in BGR mode

        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to 20
        :param img: Numpy array with the background image to use (in BGR mode), defaults to black
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in visualise()
        :return: Numpy array of the flow visualised as arrowed lines, of the same shape as the flow, in BGR
        """

        # Validate arguments
        grid_dist = 20 if grid_dist is None else grid_dist
        if not isinstance(grid_dist, int):
            raise TypeError("Error visualising flow arrows: Grid_dist needs to be an integer value")
        if not grid_dist > 0:
            raise ValueError("Error visualising flow arrows: Grid_dist needs to be an integer larger than zero")
        if img is None:
            img = np.zeros(self.shape[:2] + (3,), 'uint8')
        if not isinstance(img, np.ndarray):
            raise TypeError("Error visualising flow arrows: Img needs to be a numpy array")
        if not img.ndim == 3 or img.shape[:2] != self.shape or img.shape[2] != 3:
            raise ValueError("Error visualising flow arrows: "
                             "Img needs to have 3 channels and the same shape as the flow")
        if scaling is not None:
            if not isinstance(scaling, (float, int)):
                raise TypeError("Error visualising flow arrows: Scaling needs to be a float or an integer")
            if scaling <= 0:
                raise ValueError("Error visualising flow arrows: Scaling needs to be larger than zero")
        show_mask = False if show_mask is None else show_mask
        show_mask_borders = False if show_mask_borders is None else show_mask_borders
        if not isinstance(show_mask, bool):
            raise TypeError("Error visualising flow: Show_mask needs to be boolean")
        if not isinstance(show_mask_borders, bool):
            raise TypeError("Error visualising flow: Show_mask_borders needs to be boolean")
        if colour is not None:
            if not isinstance(colour, tuple):
                raise TypeError("Error visualising flow: Colour needs to be a tuple")
            if len(colour) != 3:
                raise ValueError("Error visualising flow arrows: Colour list or tuple needs to have length 3")

        # Thresholding
        f = self.vecs.copy()
        f[(-self._threshold < f) & (f < self._threshold)] = 0

        # Make points
        x, y = np.mgrid[:f.shape[0] - 1:grid_dist, :f.shape[1] - 1:grid_dist]
        i_pts = np.dstack((x, y))
        i_pts_flat = np.reshape(i_pts, (-1, 2)).astype('i')
        f_at_pts = f[i_pts_flat[..., 0], i_pts_flat[..., 1]]
        flow_mags, ang = cv2.cartToPolar(f_at_pts[..., 0], f_at_pts[..., 1], angleInDegrees=True)
        if scaling is None:
            scaling = grid_dist / np.percentile(flow_mags, 99)
        flow_mags *= scaling
        f *= scaling
        colours = None
        tip_size = 3.5
        if colour is None:
            hsv = np.full((1, ang.shape[0], 3), 255, 'uint8')
            hsv[0, :, 0] = np.round(ang[:, 0] / 2)
            colours = np.squeeze(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        for i_num, i_pt in enumerate(i_pts_flat):
            if flow_mags[i_num] > 0.5:  # Only draw if the flow length rounds to at least one pixel
                e_pt = np.round(i_pt + f[i_pt[0], i_pt[1]][::-1]).astype('i')
                c = tuple(int(item) for item in colours[i_num]) if colour is None else colour
                tip_length = tip_size / flow_mags[i_num]
                cv2.arrowedLine(img, (i_pt[1], i_pt[0]), (e_pt[1], e_pt[0]), c,
                                thickness=1, tipLength=tip_length, line_type=cv2.LINE_AA)
            img[i_pt[0], i_pt[1]] = [0, 0, 255]

        # Show mask and mask borders if required
        if show_mask:
            img[~self.mask] = np.round(0.5 * img[~self.mask]).astype('uint8')
        if show_mask_borders:
            mask_as_img = np.array(255 * self.mask, 'uint8')
            contours, hierarchy = cv2.findContours(mask_as_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
        return img

    def show(self, wait: int = None, show_mask: bool = None, show_mask_borders: bool = None):
        """Shows the flow in a cv2 window

        :param wait: Integer determining how long to show the flow for, in ms. Defaults to 0, which means show until
            window closed or process terminated
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether flow mask border is visualised, defaults to False
        """

        wait = 0 if wait is None else wait
        if not isinstance(wait, int):
            raise TypeError("Error showing flow: Wait needs to be an integer")
        if wait < 0:
            raise ValueError("Error showing flow: Wait needs to be an integer larger than zero")
        img = self.visualise('bgr', show_mask, show_mask_borders)
        cv2.imshow('Visualise and show flow', img)
        cv2.waitKey(wait)

    def show_arrows(
        self,
        wait: int = None,
        grid_dist: int = None,
        img: np.ndarray = None,
        scaling: Union[float, int] = None,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        colour: tuple = None
    ):
        """Shows the flow in a cv2 window, visualised with arrows

        :param wait: Integer determining how long to show the flow for, in ms. Defaults to 0, which means show until
            window closed or process terminated
        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to 20
        :param img: Numpy array with the background image to use (in BGR mode), defaults to black
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in visualise()
        """

        wait = 0 if wait is None else wait
        if not isinstance(wait, int):
            raise TypeError("Error showing flow: Wait needs to be an integer")
        if wait < 0:
            raise ValueError("Error showing flow: Wait needs to be an integer larger than zero")
        img = self.visualise_arrows(grid_dist, img, scaling, show_mask, show_mask_borders, colour)
        cv2.imshow('Visualise and show flow', img)
        cv2.waitKey(wait)

    def target_area(self) -> np.ndarray:
        """Finds the valid area in the target image

        Given source image, flow, and target image created by warping the source with the flow, the valid area is a
        boolean mask that is True wherever the value in the target stems from warping a value from the source, and
        False where no valid information is known. Pixels that are False in this valid area will often be black (or
        'empty') in the warped target image, but not necessarily, due to warping artefacts etc. Even when they are all
        empty, the valid area allows a distinction between pixels that are black due to no actual information being
        available at this position, and pixels that are black due to black pixel values having been warped to that
        location by the flow.

        :return: Valid area in the target image
        """

        if self.ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Warping this mask to the target image
            # gives a boolean mask of which positions in the target image are valid, i.e. have been filled by values
            # warped there from the source by flow vectors that were themselves valid:
            # area = F{source & mask}, where: source & mask = mask, because: source = True everywhere
            area = apply_flow(self.vecs, self.mask.astype('f'), self.ref)
            area = np.round(area).astype('bool')
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, warping a test array that
            # is true everywhere, ANDed with the flow mask, will yield a boolean mask of valid positions in the target
            # image, i.e. positions that have been filled by values warped there from the source by flow vectors that
            # were themselves valid:
            # area = F{source} & mask, where: source = True everywhere
            area = apply_flow(self.vecs, np.ones(self.shape), self.ref)
            area = np.round(area).astype('bool')
            area &= self.mask
        return area

    def source_area(self) -> np.ndarray:
        """Finds the area in the source image that will end up being valid in the target image after warping

        Given source image, flow, and target image created by warping the source with the flow, the 'source area' is a
        boolean mask that is True wherever the value in the source will end up somewhere in the valid target area, and
        False where the value in the source will either be warped outside of the target image, or not be warped at all
        due to a lack of valid flow vectors connecting to this position.

        :return: Area in the source image valid in target image after warping
        """

        if self.ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping (equal to self.target_area()),
            # warping a test array that is True everywhere from target to source with the inverse of the flow, ANDed
            # with the flow mask, will yield a boolean mask of valid positions in the source image:
            # area = F.inv{target} & mask, where target = True everywhere
            area = apply_flow(-self.vecs, np.ones(self.shape), 't')
            # Note: this is equal to: area = self.invert('t').apply(np.ones(self.shape)), but more efficient as there
            # is no unnecessary warping of the mask
            area = np.round(area).astype('bool')
            area &= self.mask
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping (equal to self.target_area()),
            # warping the flow mask from target to source with the inverse of the flow will yield a boolean mask of
            # valid positions in the source image:
            # area = F.inv{target & mask}, where target & mask = mask, because target = True everywhere
            area = apply_flow(-self.vecs, self.mask.astype('f'), 's')
            # Note: this is equal to: area = self.invert('s').apply(self.mask.astype('f')), but more efficient as there
            # is no unnecessary warping of the mask
            area = np.round(area).astype('bool')
        # Note: alternative way of seeing this: self.source_area() = self.invert(<other ref>).target_area()
        return area
