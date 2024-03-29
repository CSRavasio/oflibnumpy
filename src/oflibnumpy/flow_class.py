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
# This file is part of oflibnumpy. It contains the custom flow class and makes use of functions defined in utils.

from __future__ import annotations

import math
from typing import Union, Tuple
import warnings
import cv2
import numpy as np
from scipy.interpolate import griddata
from .utils import get_valid_ref, get_valid_padding, validate_shape, apply_flow, threshold_vectors, \
    from_matrix, from_transforms, load_kitti, load_sintel, load_sintel_mask, resize_flow, is_zero_flow, track_pts


FlowAlias = 'Flow'


class Flow(object):
    # """Custom flow class, based on numpy arrays, with three attributes: vectors :attr:`vecs`, reference :attr:`ref`,
    # and mask :attr:`mask`.
    #
    # """
    _vecs: np.ndarray
    _mask: np.ndarray
    _ref: str

    def __init__(self, flow_vectors: np.ndarray, ref: str = None, mask: np.ndarray = None) -> FlowAlias:
        """Flow object constructor. For a more detailed explanation of the arguments, see the class attributes
        :attr:`vecs`, :attr:`ref`, and :attr:`mask`.

        :param flow_vectors: Numpy array of shape :math:`(H, W, 2)` containing the flow vector in OpenCV convention:
            ``flow_vectors[..., 0]`` are the horizontal, ``flow_vectors[..., 1]`` are the vertical vector components,
            defined as positive when pointing to the right / down.
        :param ref: Flow reference, either ``t`` for "target", or ``s`` for "source". Defaults to ``t``
        :param mask: Numpy array of shape :math:`(H, W)` containing a boolean mask indicating where the flow vectors
            are valid. Defaults to ``True`` everywhere.
        """
        self.vecs = flow_vectors
        self.ref = ref
        self.mask = mask

    @property
    def vecs(self) -> np.ndarray:
        """Flow vectors, a numpy array of shape :math:`(H, W, 2)`. The last dimension contains the flow
        vectors. These are in the order horizontal component first, vertical component second (OpenCV convention). They
        are defined as positive towards the right and the bottom, meaning the origin is located in the left upper
        corner of the :math:`H \\times W` flow field area.

        :return: Flow vectors as numpy array of shape :math:`(H, W, 2)` and type ``float32``
        """

        return self._vecs

    @vecs.setter
    def vecs(self, input_vecs: np.ndarray):
        """Sets flow vectors, after checking validity

        :param input_vecs: Numpy array of shape :math:`(H, W, 2)`
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
        """Flow reference, a string: either ``s`` for "source" or ``t`` for "target". This determines whether the
        regular grid of shape :math:`(H, W)` associated with the flow vectors should be understood as the source of the
        vectors (which then point to any other position), or the target of the vectors (whose start point can then be
        any other position). The flow reference ``t`` is the default, meaning the regular grid refers to the
        coordinates the pixels whose motion is being recorded by the vectors end up at.

        Applying a flow with reference ``s`` is known as "forward" warping, while reference ``t`` corresponds to what
        is termed "backward" or "reverse" warping.

        .. caution::

            The :meth:`~oflibnumpy.Flow.apply` method for warping an image is significantly faster with a flow in ``t``
            reference. The reason is that this requires interpolating unstructured points from a regular grid, while
            reference ``s`` requires interpolating a regular grid from unstructured points. The former uses the fast
            OpenCV :func:`remap` function, the latter is much more operationally complex and relies on the SciPy
            :func:`griddata` function.

        .. caution::

            The :meth:`~oflibnumpy.Flow.track` method for tracking points is significantly faster with a flow in ``s``
            reference, again due to not requiring a call to SciPy's :func:`griddata` function.

        .. tip::

            If some algorithm :func:`get_flow` is set up to calculate a flow field with reference ``t`` (or ``s``) as in
            ``flow_one_ref = get_flow(img1, img2)``, it is very simple to obtain the flow in reference ``s`` (or ``t``)
            instead: simply call the algorithm with the images in the reversed order, and multiply the resulting flow
            vectors by -1: ``flow_other_ref = -1 * get_flow(img2, img1)``

        :return: Flow reference, as string of value ``t`` or ``s``
        """

        return self._ref

    @ref.setter
    def ref(self, input_ref: str = None):
        """Sets flow reference, after checking validity

        :param input_ref: Flow reference as string of value ``t`` or ``s``. Defaults to ``t``
        """

        self._ref = get_valid_ref(input_ref)

    @property
    def mask(self) -> np.ndarray:
        """Flow mask as a numpy array of shape :math:`(H, W)` and type ``bool``. This array indicates, for each
        flow vector, whether it is considered "valid". As an example, this allows for masking of the flow based on
        object segmentations. It is also necessary to keep track of which flow vectors are valid when different flow
        fields are combined, as those operations often lead to undefined (partially or fully unknown) points in the
        given :math:`H \\times W` area where the flow vectors are either completely unknown, or will not have valid
        values.

        :return: Flow mask as numpy array of shape :math:`(H, W)` and type ``bool``
        """

        return self._mask

    @mask.setter
    def mask(self, input_mask: np.ndarray = None):
        """Sets flow mask, after checking validity

        :param input_mask: numpy array of shape :math:`(H, W)` and type ``bool``, matching flow vector array of shape
            :math:`(H, W, 2)`
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
        """Shape (resolution) :math:`(H, W)` of the flow, corresponding to the first two dimensions of the flow
        vector array of shape :math:`(H, W, 2)`

        :return: Tuple of the shape (resolution) :math:`(H, W)` of the flow object
        """

        return self._vecs.shape[:2]

    @classmethod
    def zero(cls, shape: Union[list, tuple], ref: str = None, mask: np.ndarray = None) -> FlowAlias:
        """Flow object constructor, zero everywhere

        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array of shape :math:`(H, W)` and type ``bool`` indicating where the flow vectors are valid.
            Defaults to ``True`` everywhere
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
    ) -> FlowAlias:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as numpy array of shape :math:`(3, 3)`
        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array of shape :math:`(H, W)` and type ``bool`` indicating where the flow vectors are valid.
            Defaults to ``True`` everywhere
        :return: Flow object
        """

        flow_vectors = from_matrix(matrix, shape, ref)
        return cls(flow_vectors, ref, mask)

    @classmethod
    def from_transforms(
        cls,
        transform_list: list,
        shape: Union[list, tuple],
        ref: str = None,
        mask: np.ndarray = None
    ) -> FlowAlias:
        """Flow object constructor, based on list of transforms

        :param transform_list: List of transforms to be turned into a flow field, where each transform is expressed as
            a list of [``transform name``, ``transform value 1``, ... , ``transform value n``]. Supported options:

            - Transform ``translation``, with values ``horizontal shift in px``, ``vertical shift in px``
            - Transform ``rotation``, with values ``horizontal centre in px``, ``vertical centre in px``,
              ``angle in degrees, counter-clockwise``
            - Transform ``scaling``, with values ``horizontal centre in px``, ``vertical centre in px``,
              ``scaling fraction``
        :param shape: List or tuple of the shape :math:`(H, W)` of the flow field
        :param ref: Flow reference, string of value ``t`` ("target") or ``s`` ("source"). Defaults to ``t``
        :param mask: Numpy array of shape :math:`(H, W)` and type ``bool`` indicating where the flow vectors are valid.
            Defaults to ``True`` everywhere
        :return: Flow object
        """

        flow_vectors = from_transforms(transform_list, shape, ref)
        return cls(flow_vectors, ref, mask)

    @classmethod
    def from_kitti(cls, path: str, load_valid: bool = None) -> FlowAlias:
        """Loads the flow field contained in KITTI ``uint16`` png images files, optionally including the valid pixels.
        Follows the official instructions on how to read the provided .png files on the
        `KITTI optical flow dataset website`_.

        .. _KITTI optical flow dataset website: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

        :param path: String containing the path to the KITTI flow data (``uint16``, .png file)
        :param load_valid: Boolean determining whether the valid pixels are loaded as the flow :attr:`mask`. Defaults
            to ``True``
        :return: A flow object corresponding to the KITTI flow data, with flow reference :attr:`ref` ``s``.
        """

        load_valid = True if load_valid is None else load_valid
        if not isinstance(load_valid, bool):
            raise TypeError("Error loading flow from KITTI data: Load_valid needs to be boolean")

        data = load_kitti(path)
        if load_valid:
            return cls(data[..., :2], 's', data[..., 2].astype('bool'))
        else:
            return cls(data[..., :2], 's')

    @classmethod
    def from_sintel(cls, path: str, inv_path: str = None) -> FlowAlias:
        """Loads the flow field contained in Sintel .flo byte files, including the invalid pixels if required. Follows
        the official instructions provided alongside the .flo data on the `Sintel optical flow dataset website`_.

        .. _Sintel optical flow dataset website: http://sintel.is.tue.mpg.de/

        :param path: String containing the path to the Sintel flow data (.flo byte file, little Endian)
        :param inv_path: String containing the path to the Sintel invalid pixel data (.png, black and white)
        :return: A flow object corresponding to the Sintel flow data, with flow reference :attr:`ref` ``s``
        """

        flow = load_sintel(path)
        mask = None if inv_path is None else load_sintel_mask(inv_path)
        return cls(flow, 's', mask)

    def copy(self) -> FlowAlias:
        """Copy a flow object by constructing a new one with the same vectors :attr:`vecs`, reference :attr:`ref`, and
        mask :attr:`mask`

        :return: Copy of the flow object
        """

        return Flow(self._vecs, self._ref, self._mask)

    def __str__(self) -> str:
        """Enhanced string representation of the flow object, containing the flow reference :attr:`ref` and shape
        :attr:`shape`

        :return: String representation
        """

        info_string = "Flow object, reference {}, shape {}*{}; ".format(self._ref, *self.shape)
        info_string += self.__repr__()
        return info_string

    def __getitem__(self, item: Union[int, list, slice]) -> FlowAlias:
        """Mimics ``__getitem__`` of a numpy array, returning a new flow object cut accordingly

        Will throw an error if ``mask.__getitem__(item)`` or ``vecs.__getitem__(item)`` (corresponding to
        ``mask[item]`` and ``vecs[item]``) throw an error. Also throws an error if sliced :attr:`vecs` or :attr:`mask`
        are not suitable to construct a new flow object with, e.g. if the number of dimensions is too low.

        :param item: Slice used to select a part of the flow
        :return: New flow object cut as a corresponding numpy array would be cut
        """

        return Flow(self._vecs.__getitem__(item), self._ref, self._mask.__getitem__(item))

    def __add__(self, other: Union[np.ndarray, FlowAlias]) -> FlowAlias:
        """Adds a flow object or a numpy array to a flow object

        .. caution::
            This is **not** equal to applying the two flows sequentially. For that, use
            :meth:`~oflibnumpy.Flow.combine_with` with ``mode`` set to ``3``.

        .. caution::
            If this method is used to add two flow objects, there is no check on whether they have the same reference
            :attr:`ref`.

        :param other: Flow object or numpy array corresponding to the addend. Adding a flow object will adjust the mask
            of the resulting flow object to correspond to the logical union of the augend / addend masks
        :return: New flow object corresponding to the sum
        """

        if not isinstance(other, (np.ndarray, Flow)):
            raise TypeError("Error adding to flow: Addend is not a flow object or a numpy array")
        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error adding to flow: Augend and addend flow objects are not the same shape")
            else:
                vecs = self._vecs + other._vecs
                mask = np.logical_and(self._mask, other._mask)
                return Flow(vecs, self._ref, mask)
        if isinstance(other, np.ndarray):
            if self.shape != other.shape[:2] or other.ndim != 3 or other.shape[2] != 2:
                raise ValueError("Error adding to flow: Addend numpy array needs to have the same shape as the flow "
                                 "object, 3 dimensions overall, and a channel length of 2")
            else:
                vecs = self._vecs + other
                return Flow(vecs, self._ref, self._mask)

    def __sub__(self, other: Union[np.ndarray, FlowAlias]) -> FlowAlias:
        """Subtracts a flow object or a numpy array from a flow object

        .. caution::
            This is **not** equal to subtracting the effects of applying flow fields to an image. For that, use
            :meth:`~oflibnumpy.Flow.combine_with` with ``mode`` set to ``1`` or ``2``.

        .. caution::
            If this method is used to subtract two flow objects, there is no check on whether they have the same
            reference :attr:`ref`.

        :param other: Flow object or numpy array corresponding to the subtrahend. Subtracting a flow object will adjust
            the mask of the resulting flow object to correspond to the logical union of the minuend / subtrahend masks
        :return: New flow object corresponding to the difference
        """

        if not isinstance(other, (np.ndarray, Flow)):
            raise TypeError("Error subtracting from flow: Subtrahend is not a flow object or a numpy array")
        if isinstance(other, Flow):
            if self.shape != other.shape:
                raise ValueError("Error subtracting from flow: "
                                 "Minuend and subtrahend flow objects are not the same shape")
            else:
                vecs = self._vecs - other._vecs
                mask = np.logical_and(self._mask, other._mask)
                return Flow(vecs, self._ref, mask)
        if isinstance(other, np.ndarray):
            if self.shape != other.shape[:2] or other.ndim != 3 or other.shape[2] != 2:
                raise ValueError("Error subtracting from flow: Subtrahend numpy array needs to have the same shape as "
                                 "the flow object, 3 dimensions overall, and a channel length of 2")
            else:
                vecs = self._vecs - other
                return Flow(vecs, self._ref, self._mask)

    def __mul__(self, other: Union[float, int, list, np.ndarray]) -> FlowAlias:
        """Multiplies a flow object with a single number, a list, or a numpy array

        :param other: Multiplier, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - an array of the same shape :math:`(H, W)` as the flow object
            - an array of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the product
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs * float(other), self._ref, self._mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error multiplying flow: Multiplier list not length 2")
                return Flow(self._vecs * np.array(other)[np.newaxis, np.newaxis, :], self._ref, self._mask)
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
                return Flow(self._vecs * other, self._ref, self._mask)
            else:
                raise TypeError("Error multiplying flow: Multiplier cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __truediv__(self, other: Union[float, int, list, np.ndarray]) -> FlowAlias:
        """Divides a flow object by a single number, a list, or a numpy array

        :param other: Divisor, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - an array of the same shape :math:`(H, W)` as the flow object
            - an array of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the quotient
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs / float(other), self._ref, self._mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error dividing flow: Divisor list not length 2")
                return Flow(self._vecs / np.array(other)[np.newaxis, np.newaxis, :], self._ref, self._mask)
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
                return Flow(self._vecs / other, self._ref, self._mask)
            else:
                raise TypeError("Error dividing flow: Divisor cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __pow__(self, other: Union[float, int, bool, list, np.ndarray]) -> FlowAlias:
        """Exponentiates a flow object by a single number, a list, or a numpy array

        :param other: Exponent, options:

            - can be converted to a float
            - a list of shape :math:`(2)`
            - an array of the same shape :math:`(H, W)` as the flow object
            - an array of the same shape :math:`(H, W, 2)` as the flow vectors
        :return: New flow object corresponding to the power
        """

        try:  # other is int, float, or can be converted to it
            return Flow(self._vecs ** float(other), self._ref, self._mask)
        except TypeError:
            if isinstance(other, list):
                if len(other) != 2:
                    raise ValueError("Error exponentiating flow: Exponent list not length 2")
                return Flow(self._vecs ** np.array(other)[np.newaxis, np.newaxis, :], self._ref, self._mask)
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
                return Flow(self._vecs ** other, self._ref, self._mask)
            else:
                raise TypeError("Error exponentiating flow: Exponent cannot be converted to float, "
                                "or isn't a list or numpy array")

    def __neg__(self) -> FlowAlias:
        """Returns a new flow object with all the flow vectors inverted

        .. caution::
            This is **not** equal to inverting the transformation a flow field corresponds to! For that, use
            :meth:`~oflibnumpy.Flow.invert`.

        :return: New flow object with inverted flow vectors
        """

        return self * -1

    def resize(self, scale: Union[float, int, list, tuple]) -> FlowAlias:
        """Resize a flow field, scaling the flow vectors values :attr:`vecs` accordingly.

        :param scale: Scale used for resizing, options:

            - Integer or float of value ``scaling`` applied both vertically and horizontally
            - List or tuple of shape :math:`(2)` with values ``[vertical scaling, horizontal scaling]``
        :return: New flow object scaled as desired
        """

        resized_flow = resize_flow(self._vecs, scale)
        if isinstance(scale, (float, int)):
            scale = [scale, scale]
        resized_mask = cv2.resize(self._mask.astype('f'), None, fx=scale[1], fy=scale[0])
        # Note: scale can be used with no validity checks because already validated in resize_flows
        return Flow(resized_flow, self._ref, np.round(resized_mask))

    def pad(self, padding: Union[list, tuple] = None, mode: str = None) -> FlowAlias:
        """Pad the flow with the given padding. Padded flow :attr:`vecs` values are either constant (set to ``0``),
        correspond to the edge values of the flow field, or are a symmetric mirroring of the existing flow values.
        Padded :attr:`mask` values are set to ``False``.

        :param padding: List or tuple of shape :math:`(4)` with padding values ``[top, bot, left, right]``
        :param mode: String of the numpy padding mode for the flow vectors, with options ``constant`` (fill value
            ``0``), ``edge``, ``symmetric`` (see documentation for :func:`numpy.pad`). Defaults to ``constant``
        :return: New flow object with the padded flow field
        """

        mode = 'constant' if mode is None else mode
        if mode not in ['constant', 'edge', 'symmetric']:
            raise ValueError("Error padding flow: Mode should be one of "
                             "'constant', 'edge', 'symmetric', but instead got '{}'".format(mode))
        padding = get_valid_padding(padding, "Error padding flow: ")
        padded_vecs = np.pad(self._vecs, (tuple(padding[:2]), tuple(padding[2:]), (0, 0)), mode=mode)
        padded_mask = np.pad(self._mask, (tuple(padding[:2]), tuple(padding[2:])))
        return Flow(padded_vecs, self._ref, padded_mask)

    def apply(
        self,
        target: Union[np.ndarray, FlowAlias],
        target_mask: np.ndarray = None,
        return_valid_area: bool = None,
        consider_mask: bool = None,
        padding: Union[list, tuple] = None,
        cut: bool = None
    ) -> Union[Union[np.ndarray, FlowAlias], Tuple[Union[np.ndarray, FlowAlias], np.ndarray]]:
        """Apply the flow to a target, which can be a numpy array or a Flow object itself. If the flow shape
        :math:`(H_{flow}, W_{flow})` is smaller than the target shape :math:`(H_{target}, W_{target})`, a list of
        padding values needs to be passed to localise the flow in the larger :math:`H_{target} \\times W_{target}` area.

        The valid image area that can optionally be returned is ``True`` where the image values in the function output:

        1) have been affected by flow vectors. If the flow has a reference :attr:`ref` value of ``t`` ("target"),
           this is always ``True`` as the target image by default has a corresponding flow vector at each pixel
           location in :math:`H \\times W`. If the flow has a reference :attr:`ref` value of ``s`` ("source"), this
           is only ``True`` for some parts of the image: some target image pixel locations in :math:`H \\times W`
           would only be reachable by flow vectors originating outside of the source image area, which is impossible
           by definition
        2) have been affected by flow vectors that were themselves valid, as determined by the flow mask

        .. caution::

            The parameter `consider_mask` relates to whether the invalid flow vectors in a flow field with reference
            ``s`` are removed before application (default behaviour) or not. Doing so results in a smoother flow field,
            but can cause artefacts to arise where the outline of the area returned by
            :meth:`~oflibnumpy.Flow.valid_target` is not a convex hull. For a more detailed explanation with an
            illustrative example, see the section ":ref:`Applying a Flow`" in the usage documentation.

        :param target: Numpy array of shape :math:`(H, W)` or :math:`(H, W, C)`, or a flow object of shape
            :math:`(H, W)` to which the flow should be applied, where :math:`H` and :math:`W` are equal or larger
            than the corresponding dimensions of the flow itself
        :param target_mask: Optional numpy array of shape :math:`(H, W)` that indicates which part of the target is
            valid (only relevant if `target` is a numpy array). Only impacts the valid area returned when
            ``return_valid_area = True``. Defaults to ``True`` everywhere
        :param return_valid_area: Boolean determining whether the valid image area is returned (only if the target is a
            numpy array), defaults to ``False``. The valid image area is returned as a boolean numpy array of shape
            :math:`(H, W)`.
        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 's'``). Results in smoother outputs, but more artefacts. Defaults to
            ``True``
        :param padding: List or tuple of shape :math:`(4)` with padding values ``[top, bottom, left, right]``. Required
            if the flow and the target don't have the same shape. Defaults to ``None``, which means no padding needed
        :param cut: Boolean determining whether the warped target is returned cut from :math:`(H_{target}, W_{target})`
            to :math:`(H_{flow}, W_{flow})`, in the case that the shapes are not the same. Defaults to ``True``
        :return: The warped target of the same shape :math:`(C, H, W)` and type as the input (rounded if necessary),
            and optionally the valid area of the flow as a boolean array of shape :math:`(H, W)`
        """

        return_valid_area = False if return_valid_area is None else return_valid_area
        if not isinstance(return_valid_area, bool):
            raise TypeError("Error applying flow: Return_valid_area needs to be a boolean")
        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        cut = True if cut is None else cut
        if not isinstance(cut, bool):
            raise TypeError("Error applying flow: Cut needs to be a boolean")
        if padding is None:
            if self.shape[0] != target.shape[0] or self.shape[1] != target.shape[1]:
                raise ValueError("Error applying flow: Flow shape does not match target shape")
        if padding is not None:
            padding = get_valid_padding(padding, "Error applying flow: ")
            if self.shape[0] + np.sum(padding[:2]) != target.shape[0] or \
                    self.shape[1] + np.sum(padding[2:]) != target.shape[1]:
                raise ValueError("Error applying flow: Padding values do not match flow and target shape difference")

        # Type check, prepare arrays
        return_dtype = None
        return_2d = False
        if isinstance(target, Flow):
            return_flow = True
            t = target._vecs
            mask = target._mask
        elif isinstance(target, np.ndarray):
            return_flow = False
            if target.ndim == 3:
                t = target
            elif target.ndim == 2:
                return_2d = True
                t = target[..., np.newaxis]
            else:
                raise ValueError("Error applying flow: Target needs to have the shape H-W (2 dimensions) "
                                 "or H-W-C (3 dimensions)")
            if target_mask is None:
                mask = np.ones(t.shape[:2], 'b')
            else:
                if not isinstance(target_mask, np.ndarray):
                    raise TypeError("Error applying flow: Target_mask needs to be a numpy ndarray")
                if target_mask.shape != target.shape[:2]:
                    raise ValueError("Error applying flow: Target_mask needs to match the target shape")
                if target_mask.dtype != bool:
                    raise TypeError("Error applying flow: Target_mask needs to have dtype 'bool'")
                if not return_valid_area:
                    warnings.warn("Warning applying flow: a mask is passed, but return_valid_area is False - so the "
                                  "mask passed will not affect the output, but possibly make the function slower.")
                mask = target_mask
            return_dtype = target.dtype
        else:
            raise ValueError("Error applying flow: Target needs to be either a flow object, or a numpy ndarray")

        # Concatenate the flow vectors with the mask if required, so they are warped in one step
        if return_flow or return_valid_area:
            # if self.ref == 't': Just warp the mask, which self.vecs are valid taken into account after warping
            if self._ref == 's':
                # Warp the target mask after ANDing with flow mask to take into account which self.vecs are valid
                if mask.shape != self.shape:
                    # If padding in use, mask can be smaller than self.mask
                    tmp = mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]].copy()
                    mask[...] = False
                    mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]] = \
                        tmp & self._mask
                else:
                    mask = mask & self._mask
            t = np.concatenate((t, mask[..., np.newaxis]), axis=-1)

        # Determine flow to use for warping, and warp
        if padding is None:
            if not target.shape[:2] == self.shape[:2]:
                raise ValueError("Error applying flow: Flow and target have to have the same shape")
            warped_t = apply_flow(self._vecs, t, self._ref, self._mask if consider_mask else None)
        else:
            mode = 'constant' if self._ref == 't' else 'edge'
            # Note: this mode is very important: irrelevant for flow with reference 't' as this by definition covers
            # the area of the target image, so 'constant' (defaulting to filling everything with 0) is fine. However,
            # for flows with reference 's', if locations in the source image with some flow vector border padded
            # locations with flow zero, very strange interpolation artefacts will result, both in terms of the image
            # being warped, and the mask being warped. By padding with the 'edge' mode, large gradients in flow vector
            # values at the edge of the original flow area are avoided, as are interpolation artefacts.
            flow = self.pad(padding, mode=mode)
            warped_t = apply_flow(flow._vecs, t, flow._ref, flow._mask if consider_mask else None)

        # Cut if necessary
        if padding is not None and cut:
            warped_t = warped_t[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]]

        # Extract and finalise mask if required
        if return_flow or return_valid_area:
            mask = warped_t[..., -1] == 1

            # if self.ref == 's': Valid self.vecs already taken into account by ANDing with self.mask before warping
            if self._ref == 't':
                # Still need to take into account which self.vecs are actually valid by ANDing with self.mask
                if mask.shape != self._mask.shape:
                    # If padding is in use, but warped_t has not been cut: AND with self.mask inside the flow area, and
                    # set everything else to False as not warped by the flow
                    t = mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]].copy()
                    mask[...] = False
                    mask[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]] = t & self._mask
                else:
                    mask = mask & self._mask

        # Return as correct type
        if return_flow:
            return Flow(warped_t[:, :, :2], target._ref, mask)
        else:
            if return_valid_area:
                warped_t = warped_t[:, :, :-1]
            if np.issubdtype(return_dtype, np.integer):
                warped_t = np.round(warped_t)
            if return_2d:
                warped_t = warped_t[:, :, 0]
            if return_valid_area:
                return warped_t.astype(return_dtype), mask
            else:
                return warped_t.astype(return_dtype)

    def switch_ref(self, mode: str = None) -> FlowAlias:
        """Switch the reference :attr:`ref` between ``s`` ("source") and ``t`` ("target")

        .. caution::

            Do not use ``mode=invalid`` if avoidable: it does not actually change any flow values, and the resulting
            flow object, when applied to an image, will no longer yield the correct result.

        :param mode: Mode used for switching, available options:

            - ``invalid``: just the flow reference attribute is switched without any flow values being changed. This
              is functionally equivalent to simply assigning ``flow.ref = 't'`` for a "source" flow or
              ``flow.ref = 's'`` for a "target" flow
            - ``valid`` (default): the flow field is switched to the other coordinate reference, with flow vectors
              recalculated accordingly
        :return: New flow object with switched coordinate reference
        """

        mode = 'valid' if mode is None else mode
        if mode == 'valid':
            if self.is_zero(thresholded=False):  # In case the flow is 0, no further calculations are necessary
                return self.switch_ref(mode='invalid')
            else:
                if self._ref == 's':
                    switched_ref_flow = self.apply(self)  # apply_to is done s-based; see window pic 08/04/19
                    switched_ref_flow._ref = 't'
                    return switched_ref_flow
                elif self._ref == 't':
                    flow_copy_s = self.switch_ref(mode='invalid')  # so apply_to is ref-s; see window pic 08/04/19
                    return (-flow_copy_s).apply(flow_copy_s)
        elif mode == 'invalid':
            if self._ref == 's':
                return Flow(self._vecs, 't', self._mask)
            elif self._ref == 't':
                return Flow(self._vecs, 's', self._mask)
        else:
            raise ValueError("Error switching flow reference: Mode not recognised, should be 'valid' or 'invalid'")

    def invert(self, ref: str = None) -> FlowAlias:
        """Inverting a flow: `img`\\ :sub:`1` -- `f` --> `img`\\ :sub:`2` becomes `img`\\ :sub:`1` <-- `f` --
        `img`\\ :sub:`2`. The smaller the input flow, the closer the inverse is to simply multiplying the flow by -1.

        :param ref: Desired reference of the output field, defaults to the reference of original flow field
        :return: New flow object, inverse of the original
        """

        ref = self._ref if ref is None else get_valid_ref(ref)
        if self._ref == 's':
            if ref == 's':
                return self.apply(-self)
            elif ref == 't':
                return Flow(-self._vecs, 't', self._mask)
        elif self._ref == 't':
            if ref == 's':
                return Flow(-self._vecs, 's', self._mask)
            elif ref == 't':
                return self.invert('s').switch_ref()

    def track(
        self,
        pts: np.ndarray,
        int_out: bool = None,
        get_valid_status: bool = None,
        s_exact_mode: bool = None
    ) -> np.ndarray:
        """Warp input points with the flow field, returning the warped point coordinates as integers if required

        .. tip::
            Calling :meth:`~oflibnumpy.Flow.track` on a flow field with reference :attr:`ref` ``s`` ("source") is
            significantly faster (as long as `s_exact_mode` is not set to ``True``), as this does not require a call to
            :func:`scipy.interpolate.griddata`.

        :param pts: Numpy array of shape :math:`(N, 2)` containing the point coordinates. ``pts[:, 0]`` corresponds to
            the vertical coordinate, ``pts[:, 1]`` to the horizontal coordinate
        :param int_out: Boolean determining whether output points are returned as rounded integers, defaults to
            ``False``
        :param get_valid_status: Boolean determining whether an array of shape :math:`(N, 2)` is returned, which
            contains the status of each point. This corresponds to applying :meth:`~oflibnumpy.Flow.valid_source` to the
            point positions, and returns ``True`` for the points that 1) tracked by valid flow vectors, and 2) end up
            inside the flow area of :math:`H \\times W`. Defaults to ``False``
        :param s_exact_mode: Boolean determining whether the necessary flow interpolation will be done using
            :func:`scipy.interpolate.griddata`, if the flow has the reference :attr:`ref` value of ``s`` ("source").
            Defaults to ``False``, which means a less exact, but around 2 orders of magnitude faster bilinear
            interpolation method will be used. This is recommended for normal point tracking applications.
        :return: Numpy array of warped ('tracked') points, and optionally a numpy array of the point tracking status
        """

        # Validate inputs
        get_valid_status = False if get_valid_status is None else get_valid_status
        if not isinstance(get_valid_status, bool):
            raise TypeError("Error tracking points: Get_tracked needs to be a boolean")

        warped_pts = track_pts(flow=self._vecs, ref=self._ref, pts=pts, int_out=int_out, s_exact_mode=s_exact_mode)
        if get_valid_status:
            status_array = self.valid_source()[np.round(pts[..., 0]).astype('i'),
                                               np.round(pts[..., 1]).astype('i')]
            return warped_pts, status_array
        else:
            return warped_pts

    def matrix(self, dof: int = None, method: str = None, masked: bool = None) -> np.ndarray:
        """Fit a transformation matrix to the flow field using OpenCV functions

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
        :param masked: Boolean determining whether the flow mask is used to ignore flow locations where the mask
            :attr:`mask` is ``False``. Defaults to ``True``
        :return: Numpy array of shape :math:`(3, 3)` containing the transformation matrix
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
        if self._ref == 't':
            dst_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            src_pts = dst_pts - self._vecs
        else:  # ref is 's'
            src_pts = np.stack(np.mgrid[:self.shape[0], :self.shape[1]], axis=-1)[..., ::-1]
            dst_pts = src_pts + self._vecs
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)

        # Mask if required
        if masked:
            src_pts = src_pts[self._mask.ravel()]
            dst_pts = dst_pts[self._mask.ravel()]

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
        """Visualises the flow as an rgb / bgr / hsv image, optionally showing the outline of the flow mask :attr:`mask`
        as a black line, and the invalid areas greyed out.

        :param mode: Output mode, options: ``rgb``, ``bgr``, ``hsv``
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
            the flow magnitudes. Defaults to the 99th percentile of the flow field magnitudes
        :return: Numpy array of shape :math:`(H, W, 3)` containing the flow visualisation
        """

        show_mask = False if show_mask is None else show_mask
        show_mask_borders = False if show_mask_borders is None else show_mask_borders
        if not isinstance(show_mask, bool):
            raise TypeError("Error visualising flow: Show_mask needs to be boolean")
        if not isinstance(show_mask_borders, bool):
            raise TypeError("Error visualising flow: Show_mask_borders needs to be boolean")

        f = threshold_vectors(self._vecs)
        # Threshold the flow: very small numbers can otherwise lead to issues when calculating mag / angle

        # Colourise the flow
        hsv = np.zeros((f.shape[0], f.shape[1], 3), 'f')
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1], angleInDegrees=True)
        hsv[..., 0] = np.mod(ang, 360) / 2
        hsv[..., 2] = 255

        # Add mask if required
        if show_mask:
            hsv[np.invert(self._mask), 2] = 180

        # Scale flow
        if range_max is None:
            if np.percentile(mag, 99) > 0:  # Use 99th percentile to avoid extreme outliers skewing the scaling
                range_max = float(np.percentile(mag, 99))
            elif np.max(mag):  # If the 99th percentile is 0, use the actual maximum instead
                range_max = float(np.max(mag))
            else:  # If the maximum is 0 too (i.e. the flow field is entirely 0)
                range_max = 1
        if not isinstance(range_max, (float, int)):
            raise TypeError("Error visualising flow: Range_max needs to be an integer or a float")
        if range_max <= 0:
            raise ValueError("Error visualising flow: Range_max needs to be larger than zero")
        hsv[..., 1] = np.clip(mag * 255 / range_max, 0, 255)

        # Add mask borders if required
        if show_mask_borders:
            contours, hierarchy = cv2.findContours((255 * self._mask).astype('uint8'),
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
        grid_dist: int = None,
        img: np.ndarray = None,
        scaling: Union[float, int] = None,
        show_mask: bool = None,
        show_mask_borders: bool = None,
        colour: tuple = None,
        thickness: int = None,
    ) -> np.ndarray:
        """Visualises the flow as arrowed lines, optionally showing the outline of the flow mask :attr:`mask` as a black
        line, and the invalid areas greyed out.

        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
            ``20``
        :param img: Numpy array with the background image to use (in BGR mode), defaults to white
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
            :meth:`~oflibnumpy.Flow.visualise`
        :param thickness: Integer of the flow arrow thickness, larger than zero. Defaults to ``1``
        :return: Numpy array of shape :math:`(H, W, 3)` containing the flow visualisation, in ``bgr`` colour space
        """

        # Validate arguments
        grid_dist = 20 if grid_dist is None else grid_dist
        if not isinstance(grid_dist, int):
            raise TypeError("Error visualising flow arrows: Grid_dist needs to be an integer value")
        if grid_dist > min(self.shape) // 2:
            print("Warning: grid_dist in visualise_arrows is '{}', which is too large for a flow field of shape "
                  "({}, {}). grid_dist will be reset to '{}'."
                  .format(grid_dist, *self.shape, min(self.shape) // 2))
            grid_dist = min(self.shape) // 2
        if not grid_dist > 0:
            raise ValueError("Error visualising flow arrows: Grid_dist needs to be an integer larger than zero")
        if img is None:
            img = np.full(self.shape[:2] + (3,), 255, 'uint8')
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
        thickness = 1 if thickness is None else thickness
        if not isinstance(thickness, int):
            raise TypeError("Error visualising flow: Thickness needs to be an integer")
        if thickness <= 0:
            raise ValueError("Error visualising flow: Thickness needs to be a integer larger than zero")

        # Thresholding
        f = threshold_vectors(self._vecs)

        # Make points
        x, y = np.mgrid[grid_dist//2:f.shape[0] - 1:grid_dist, grid_dist//2:f.shape[1] - 1:grid_dist]
        i_pts = np.dstack((x, y))
        i_pts_flat = np.reshape(i_pts, (-1, 2)).astype('i')
        f_at_pts = f[i_pts_flat[..., 0], i_pts_flat[..., 1]]
        flow_mags, ang = cv2.cartToPolar(f_at_pts[..., 0], f_at_pts[..., 1], angleInDegrees=True)
        if scaling is None:
            scaling = grid_dist / np.percentile(flow_mags, 99)
        flow_mags *= scaling
        f *= scaling
        colours = None
        tip_size = math.sqrt(thickness) * 3.5  # Empirical value
        if colour is None:
            hsv = np.full((1, ang.shape[0], 3), 255, 'uint8')
            hsv[0, :, 0] = np.round(np.mod(ang[:, 0], 360) / 2)
            colours = np.squeeze(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        for i_num, i_pt in enumerate(i_pts_flat):
            if flow_mags[i_num] > 0.5:  # Only draw if the flow length rounds to at least one pixel
                c = tuple(int(item) for item in colours[i_num]) if colour is None else colour
                tip_length = float(tip_size / flow_mags[i_num])
                if self.ref == 's':
                    e_pt = np.round(i_pt + f[i_pt[0], i_pt[1]][::-1]).astype('i')
                    cv2.arrowedLine(img, (i_pt[1], i_pt[0]), (e_pt[1], e_pt[0]), c,
                                    thickness=1, tipLength=tip_length, line_type=cv2.LINE_AA)
                else:  # self.ref == 't'
                    e_pt = np.round(i_pt - f[i_pt[0], i_pt[1]][::-1]).astype('i')
                    cv2.arrowedLine(img, (e_pt[1], e_pt[0]), (i_pt[1], i_pt[0]), c,
                                    thickness=thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
            img[i_pt[0], i_pt[1]] = [0, 0, 255]

        # Show mask and mask borders if required
        if show_mask:
            img[~self._mask] = np.round(0.5 * img[~self._mask]).astype('uint8')
        if show_mask_borders:
            mask_as_img = np.array(255 * self._mask, 'uint8')
            contours, hierarchy = cv2.findContours(mask_as_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
        return img

    def show(self, wait: int = None, show_mask: bool = None, show_mask_borders: bool = None):
        """Shows the flow in an OpenCV window using :meth:`~oflibnumpy.Flow.visualise`

        :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
            it will be shown until the window is closed, or the process is terminated
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether flow mask border is visualised, defaults to ``False``
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
        """Shows the flow in an OpenCV window using :meth:`~oflibnumpy.Flow.visualise_arrows`

        :param wait: Integer determining how long to show the flow for, in milliseconds. Defaults to ``0``, which means
            it will be shown until the window is closed, or the process is terminated
        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation, defaults to
            ``20``
        :param img: Numpy array with the background image to use (in BGR colour space), defaults to black
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to ``False``
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to ``False``
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in
            :meth:`~oflibnumpy.Flow.visualise`
        """

        wait = 0 if wait is None else wait
        if not isinstance(wait, int):
            raise TypeError("Error showing flow: Wait needs to be an integer")
        if wait < 0:
            raise ValueError("Error showing flow: Wait needs to be an integer larger than zero")
        img = self.visualise_arrows(grid_dist, img, scaling, show_mask, show_mask_borders, colour)
        cv2.imshow('Visualise and show flow', img)
        cv2.waitKey(wait)

    def valid_target(self, consider_mask: bool = None) -> np.ndarray:
        """Find the valid area in the target domain

        Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
        with the flow. The valid area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
        the value in the target img stems from warping a value from the source, and ``False`` where no valid information
        is known.

        Pixels that are ``False`` will often be black (or 'empty') in the warped target image - but not necessarily, due
        to warping artefacts etc. The valid area also allows a distinction between pixels that are black due to no
        actual information being available at this position (validity ``False``), and pixels that are black due to black
        pixel values having been warped to that (valid) location by the flow.

        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 's'``, analogous to :meth:`~oflibnumpy.Flow.apply`). Results in smoother
            outputs, but more artefacts. Defaults to ``True``
        :return: Boolean numpy array of the same shape :math:`(H, W)` as the flow
        """

        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Warping this mask to the target image
            # gives a boolean mask of which positions in the target image are valid, i.e. have been filled by values
            # warped there from the source by flow vectors that were themselves valid:
            # area = F{source & mask}, where: source & mask = mask, because: source = True everywhere
            area = apply_flow(self._vecs, self._mask.astype('f'), 's', self._mask if consider_mask else None)
            area = area == 1  # Matching behaviour of mask setting in .apply()
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, warping a test array that
            # is true everywhere, ANDed with the flow mask, will yield a boolean mask of valid positions in the target
            # image, i.e. positions that have been filled by values warped there from the source by flow vectors that
            # were themselves valid:
            # area = F{source} & mask, where: source = True everywhere
            area = apply_flow(self._vecs, np.ones(self.shape), 't')
            area = area == 1  # Matching behaviour of mask setting in .apply()
            area &= self._mask
        return area

    def valid_source(self, consider_mask: bool = None) -> np.ndarray:
        """Finds the area in the source domain that will end up being valid in the target domain (see
        :meth:`~oflibnumpy.Flow.valid_target`) after warping

        Given a source image and a flow, both of shape :math:`(H, W)`, the target image is created by warping the source
        with the flow. The source area is then a boolean numpy array of shape :math:`(H, W)` that is ``True`` wherever
        the value in the source will end up somewhere inside the valid target area, and ``False`` where the value in the
        source will either be warped outside of the target image, or not be warped at all due to a lack of valid flow
        vectors connecting to this position.

        :param consider_mask: Boolean determining whether the flow vectors are masked before application (only relevant
            for flows with reference ``ref = 't'`` as their inverse flow will be applied, using the reference ``s``;
            analogous to :meth:`~oflibnumpy.Flow.apply`). Results in smoother outputs, but more artefacts. Defaults
            to ``True``
        :return: Boolean numpy array of the same shape :math:`(H, W)` as the flow
        """

        consider_mask = True if consider_mask is None else consider_mask
        if not isinstance(consider_mask, bool):
            raise TypeError("Error applying flow: Consider_mask needs to be a boolean")
        if self._ref == 's':
            # Flow mask in 's' flow refers to valid flow vecs in the source image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping a test array that is True everywhere from target to source with the inverse of the flow, ANDed
            # with the flow mask, will yield a boolean mask of valid positions in the source image:
            # area = F.inv{target} & mask, where target = True everywhere
            area = apply_flow(-self._vecs, np.ones(self.shape), 't')
            # Note: this is equal to: area = self.invert('t').apply(np.ones(self.shape)), but more efficient as there
            # is no unnecessary warping of the mask
            area = area == 1  # Matching behaviour of mask setting in .apply()
            area &= self._mask
        else:  # ref is 't'
            # Flow mask in 't' flow refers to valid flow vecs in the target image. Therefore, to find the area in the
            # source image that will end up being valid in the target image after warping, equal to self.valid_target(),
            # warping the flow mask from target to source with the inverse of the flow will yield a boolean mask of
            # valid positions in the source image:
            # area = F.inv{target & mask}, where target & mask = mask, because target = True everywhere
            area = apply_flow(-self._vecs, self._mask.astype('f'), 's', self._mask if consider_mask else None)
            # Note: this is equal to: area = self.invert('s').apply(self.mask.astype('f')), but more efficient as there
            # is no unnecessary warping of the mask
            area = area == 1  # Matching behaviour of mask setting in .apply()
        # Note: alternative way of seeing this: self.valid_source() = self.invert(<other ref>).valid_target()
        return area

    def get_padding(self) -> list:
        """Determine necessary padding from the flow field:

        - When the flow reference :attr:`ref` has the value ``t`` ("target"), this corresponds to the padding needed in
          a source image which ensures that every flow vector in :attr:`vecs` marked as valid by the
          mask :attr:`mask` will find a value in the source domain to warp towards the target domain. I.e. any invalid
          locations in the area :math:`H \\times W` of the target domain (see :meth:`~oflibnumpy.Flow.valid_target`) are
          purely due to no valid flow vector being available to pull a source value to this target location, rather than
          no source value being available in the first place.
        - When the flow reference :attr:`ref` has the value ``s`` ("source"), this corresponds to the padding needed for
          the flow itself, so that applying it to a source image will result in no input image information being lost in
          the warped output, i.e each input image pixel will come to lie inside the padded area.

        :return: A list of shape :math:`(4)` with the values ``[top, bottom, left, right]``
        """

        v = threshold_vectors(self._vecs)
        # Threshold to avoid very small flow values (possible artefacts) triggering a need for padding
        if self._ref == 's':
            v *= -1
        x, y = np.mgrid[:self.shape[0], :self.shape[1]]
        v[..., 0] -= y
        v[..., 1] -= x
        v *= -1
        padding = [
            max(-np.min(v[self._mask, 1]), 0),
            max(np.max(v[self._mask, 1]) - (self.shape[0] - 1), 0),
            max(-np.min(v[self._mask, 0]), 0),
            max(np.max(v[self._mask, 0]) - (self.shape[1] - 1), 0)
        ]
        padding = [int(np.ceil(p)) for p in padding]
        return padding

    def is_zero(self, thresholded: bool = None, masked: bool = None) -> bool:
        """Check whether all flow vectors (where :attr:`mask` is ``True``) are zero. Optionally, a threshold flow
        magnitude value of ``1e-3`` is used. This can be useful to filter out motions that are equal to very small
        fractions of a pixel, which might just be a computational artefact to begin with.

        :param thresholded: Boolean determining whether the flow is thresholded, defaults to ``True``
        :param masked: Boolean determining whether the flow is masked with :attr:`mask`, defaults to ``True``
        :return: ``True`` if the flow field is zero everywhere, otherwise ``False``
        """

        masked = True if masked is None else masked
        if not isinstance(masked, bool):
            raise TypeError("Error checking whether flow is zero: Masked needs to be a boolean")

        f = self._vecs[self._mask][np.newaxis, ...] if masked else self._vecs
        return is_zero_flow(f, thresholded)

    def combine_with(self, flow: FlowAlias, mode: int, thresholded: bool = None) -> FlowAlias:
        """Function that returns the result of the combination of two flow objects of the same shape :attr:`shape` and
        reference :attr:`ref`

        .. tip::
           All of the flow field combinations in this function rely on some combination of the
           :meth:`~oflibnumpy.Flow.apply`, :meth:`~oflibnumpy.Flow.invert`, and :meth:`~oflibnumpy.Flow.combine_with`
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
        :math:`flow_1 ⊕ flow_2 = flow_3`, where :math:`⊕` is a non-commutative flow composition operation.
        This can be visualised with the start / end points of the flows as follows:

        .. code-block::

            S = Start point    S1 = S3 ─────── f3 ────────────┐
            E = End point       │                             │
            f = flow           f1                             v
                                └───> E1 = S2 ── f2 ──> E2 = E3

        The main difficulty in combining flow fields is that it would be incorrect to simply add up or subtract
        flow vectors at one location in the flow field area :math:`H \\times W`. This appears to work given e.g.
        a translation to the right, and a translation downwards: the result will be the linear combination of the
        two vectors, or a translation towards the bottom right. However, looking more closely, it becomes evident
        that this approach isn't actually correct: A pixel that has been moved from `S1` to `E1` by the first flow
        field `f1` is then moved from that location by the flow vector of the flow field `f2` that corresponds to
        the new pixel location `E1`, *not* the original location `S1`. If the flow vectors are the same everywhere
        in the field, the difference will not be noticeable. However, if the flow vectors of `f2` vary throughout
        the field, such as with a rotation around some point, it will!

        In this case (corresponding to calling :func:`f1.combine_with(f2, mode=3)<combine_with>`), and if the
        flow reference :attr:`ref` is ``s`` ("source"), the solution is to first apply the inverse of `f1` to `f2`,
        essentially linking up each location `E1` back to `S1`, and *then* to add up the flow vectors. Analogous
        observations apply for the other permutations of flow combinations and reference :attr:`ref` values.

        .. note::
           This is consistent with the observation that two translations are commutative in their application -
           the order does not matter, and the vectors can simply be added up at every pixel location -, while a
           translation followed by a rotation is not the same as a rotation followed by a translation: adding up
           vectors at each pixel cannot be the correct solution as there wouldn't be a difference based on the
           order of vector addition.

        :param flow: Flow object to combine with
        :param mode: Integer determining how the input flows are combined, where the number corresponds to the
            position in the formula :math:`flow_1 ⊕ flow_2 = flow_3`:

            - Mode ``1``: `self` corresponds to :math:`flow_2`, `flow` corresponds to :math:`flow_3`, the result will
              be :math:`flow_1`
            - Mode ``2``: `self` corresponds to :math:`flow_1`, `flow` corresponds to :math:`flow_3`, the result will
              be :math:`flow_2`
            - Mode ``3``: `self` corresponds to :math:`flow_1`, `flow` corresponds to :math:`flow_2`, the result will
              be :math:`flow_3`
        :param thresholded: Boolean determining whether flows are thresholded during an internal call to
            :meth:`~oflibnumpy.Flow.is_zero`, defaults to ``False``
        :return: New flow object
        """

        # Check input validity
        if not isinstance(flow, Flow):
            raise TypeError("Error combining flows: Flow need to be of type 'Flow'")
        if not self.shape == flow.shape:
            raise ValueError("Error combining flows: Flow fields need to have the same shape")
        if not self.ref == flow.ref:
            raise ValueError("Error combining flows: Flow fields need to have the same reference")
        if mode not in [1, 2, 3]:
            raise ValueError("Error combining flows: Mode needs to be 1, 2 or 3")
        thresholded = False if thresholded is None else thresholded
        if not isinstance(thresholded, bool):
            raise TypeError("Error combining flows: Thresholded needs to be a boolean")

        # Check if one input is zero, return early if so
        if self.is_zero(thresholded=thresholded):
            # if mode == 1:  # Flows are in order (desired_result, self=0, flow)
            #     return flow
            # elif mode == 2:  # Flows are in order (self=0, desired_result, flow)
            #     return flow
            # elif mode == 3:  # Flows are in order (self=0, flow, desired_result)
            #     return flow
            # Above code simplifies to:
            return flow
        elif flow.is_zero(thresholded=thresholded):
            if mode == 1:  # Flows are in order (desired_result, self, flow=0)
                return self.invert()
            elif mode == 2:  # Flows are in order (self, desired_result, flow=0)
                return self.invert()
            elif mode == 3:  # Flows are in order (self, flow=0, desired_result)
                return self

        result = None
        if mode == 1:  # Flows are in order (desired_result, self, flow)
            if self._ref == flow._ref == 's':
                # Explanation: f1 is (f3 minus f2), when S2 is moved to S3, achieved by applying f2 to move S2 to E3,
                # then inverted(f3) to move from E3 to S3.
                # F1_s = F3_s - F2_s.combine_with(F3_s^-1_s, 3){F2_s}
                # result = flow - self.combine_with(flow.invert(), mode=3).apply(self)
                #
                # Alternative: this should take an equivalent amount of time (same number of griddata calls), but is
                # slightly faster in tests
                # F1_s = F3_s - F2_s-as-t.combine_with(F3_s^-1_t, 3){F2_s}
                # result = flow - self.switch_ref().combine_with(flow.invert('t'), mode=3).apply(self)
                # To avoid call to combine_with and associated overhead, do the corresponding operations directly:
                flow_inv_t = flow.invert('t')
                result = flow - (flow_inv_t + flow_inv_t.apply(self.switch_ref())).apply(self)
            elif self._ref == flow._ref == 't':
                # Explanation: currently no native implementation to ref 't', so just "translated" from ref 's'
                # F1_t = (F3_t-as-s - F2_t.combine_with(F3_t^-1_t, 3){F2_t-as-s})_as-t
                # result = flow.switch_ref() - self.combine_with(flow.invert(), mode=3).apply(self.switch_ref())
                # result = result.switch_ref()
                #
                # Alternative: saves one call to griddata. However, test shows barely a difference - not sure why
                # F1_t = (F3_t-as-s - F2_t-as-s.combine_with(F3_t^-1_s, 3){F2_t-as-s})_as-t
                # self_s = self.switch_ref()
                # result = flow.switch_ref() - self_s.combine_with(flow.invert('s'), mode=3).apply(self_s)
                # result = result.switch_ref()
                # To avoid call to combine_with and associated overhead, do the corresponding operations directly:
                self_s = self.switch_ref()
                result = flow.switch_ref() - (self_s + self_s.invert(ref='t').apply(flow.invert('s'))).apply(self_s)
                result = result.switch_ref()
        elif mode == 2:  # Flows are in order (self, desired_result, flow)
            if self._ref == flow._ref == 's':
                # Explanation: f2 is (f3 minus f1), when S1 = S3 is moved to S2, achieved by applying f1
                # F2_s = F1_s{F3_s - F1_s}
                result = self.apply(flow - self)
            elif self._ref == flow._ref == 't':
                # Strictly "translated" version from the ref 's' case:
                # F2_t = F1_t{F3_t-as-s - F1_t-as-s}_as-t)
                # result = (self.apply(flow.switch_ref() - self.switch_ref())).switch_ref()
                #
                # Improved version cutting down on operational complexity
                # F3 - F1, where F1 has been resampled to the source positions of F3.
                coord_1 = np.copy(-self.vecs)
                coord_1[:, :, 0] += np.arange(coord_1.shape[1])
                coord_1[:, :, 1] += np.arange(coord_1.shape[0])[:, np.newaxis]
                coord_1_flat = np.reshape(coord_1, (-1, 2))
                vecs_with_mask = np.concatenate((self.vecs, self.mask[..., np.newaxis]), axis=-1)
                vals_flat = np.reshape(vecs_with_mask, (-1, 3))
                coord_3 = np.copy(-flow.vecs)
                coord_3[:, :, 0] += np.arange(coord_3.shape[1])
                coord_3[:, :, 1] += np.arange(coord_3.shape[0])[:, np.newaxis]
                vals_resampled = griddata(coord_1_flat, vals_flat,
                                          (coord_3[..., 0], coord_3[..., 1]),
                                          method='linear', fill_value=0)
                result = flow - Flow(vals_resampled[..., :-1], 't', vals_resampled[..., -1] > .99)
        elif mode == 3:  # Flows are in order (self, flow, desired_result)
            if self._ref == flow._ref == 's':
                # Explanation: f3 is (f1 plus f2), when S2 is moved to S1, achieved by applying inverted(f1)
                # F3_s = F1_s + (F1_s)^-1_t{F2_s}
                # Note: instead of inverting the ref-s flow field to a ref-s flow field (slow) which is then applied
                #   to the other flow field (also slow), it is inverted to a ref-t flow field (fast) which is then
                #   also much faster to apply to the other flow field.
                result = self + self.invert(ref='t').apply(flow)
            elif self._ref == flow._ref == 't':
                # Explanation: f3 is (f2 plus f1), with f1 pulled towards the f2 grid by applying f2 to f1.
                # F3_t = F2_t + F2_t{F1_t}
                result = flow + flow.apply(self)

        return result
