from __future__ import annotations
from typing import Union
import cv2
import numpy as np
from .utils import get_valid_ref, get_valid_padding, flow_from_matrix, matrix_from_transform
from .flow_operations import apply_flow


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

        :param input_mask: bool numpy array of size H-W (self.shape), matching flow vectors with size H-W-2
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
    def zero(cls, size: list, ref: str = None, mask: np.ndarray = None) -> Flow:
        """Flow object constructor, zero everywhere

        :param size: List [H, W] of flow field size
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

        return cls(np.zeros((size[0], size[1], 2)), ref, mask)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, size: list, ref: str = None, mask: np.ndarray = None) -> Flow:
        """Flow object constructor, based on transformation matrix input

        :param matrix: Transformation matrix to be turned into a flow field, as Numpy array 3-3
        :param size: List [H, W] of flow field size
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

        ref = get_valid_ref(ref)
        if ref == 's':
            # Coordinates correspond to the meshgrid of the original ('s'ource) image. They are transformed according
            # to the transformation matrix. The start points are subtracted from the end points to yield flow vectors.
            flow_vectors = flow_from_matrix(matrix, size)
            return cls(flow_vectors, ref, mask)
        elif ref == 't':
            # Coordinates correspond to the meshgrid of the warped ('t'arget) image. They are inversely transformed
            # according to the transformation matrix. The end points, which correspond to the flow origin for the
            # meshgrid in the warped image, are subtracted from the start points to yield flow vectors.
            flow_vectors = -flow_from_matrix(np.linalg.pinv(matrix), size)
            return cls(flow_vectors, ref, mask)

    @classmethod
    def from_transforms(cls, transform_list: list, size: list, ref: str = None, mask: np.ndarray = None) -> Flow:
        """Flow object constructor, zero everywhere.

        :param transform_list: List of transforms to be turned into a flow field. Options for each transform in list:
            ['translation', horizontal shift in px, vertical shift in px]
            ['rotation', horizontal centre in px, vertical centre in px, angle in degrees, counter-clockwise]
            ['scaling', horizontal centre in px, vertical centre in px, scaling fraction]
        :param size: List [H, W] of flow field size
        :param ref: Flow referencce, 't'arget or 's'ource. Defaults to 't'
        :param mask: Numpy array H-W containing a boolean mask indicating where the flow vectors are valid. Defaults to
            True everywhere.
        :return: Flow object
        """

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
        matrix = np.identity(3)
        for transform in reversed(transform_list):
            matrix = matrix @ matrix_from_transform(transform[0], transform[1:])
        return cls.from_matrix(matrix, size, ref, mask)

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

    def __add__(self, other: Flow) -> Flow:
        """Adds flow objects

        Note: this is NOT equal to applying the two flows sequentially. For that, use combine_flows(flow1, flow2, None).
        The function also does not check whether the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object corresponding to the addend
        :return: Flow object corresponding to the sum
        """

        if not isinstance(other, Flow):
            raise TypeError("Error adding to flow: Addend is not a flow object")
        if not self.shape == other.shape:
            raise ValueError("Error adding to flow: Augend and addend are not the same size")
        vecs = self.vecs + other.vecs
        mask = np.logical_and(self.mask, other.mask)
        return Flow(vecs, self.ref, mask)

    def __sub__(self, other: Flow) -> Flow:
        """Subtracts flow objects.

        Note: this is NOT equal to subtracting the effects of applying flow fields to an image. For that, used
        combine_flows(flow1, None, flow2) or combine_flows(None, flow1, flow2). The function also does not check whether
        the two flow objects have the same reference.

        DO NOT USE if you're not certain about what you're aiming to achieve.

        :param other: Flow object corresponding to the subtrahend
        :return: Flow object corresponding to the difference
        """

        if not isinstance(other, Flow):
            raise TypeError("Error subtracting from flow: Subtrahend is not a flow object")
        if not self.shape == other.shape:
            raise ValueError("Error subtracting from flow: Minuend and subtrahend are not the same size")
        vecs = self.vecs - other.vecs
        mask = np.logical_and(self.mask, other.mask)
        return Flow(vecs, self.ref, mask)

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

    def pad(self, padding: list = None) -> Flow:
        """Pads the flow with the given padding, inserting 0 values

        :param padding: [top, bot, left, right] list of padding values
        :return: Padded flow
        """

        padding = get_valid_padding(padding, "Error padding flow: ")
        padded_vecs = np.pad(self.vecs, (tuple(padding[:2]), tuple(padding[2:]), (0, 0)))
        padded_mask = np.pad(self.mask, (tuple(padding[:2]), tuple(padding[2:])))
        return Flow(padded_vecs, self.ref, padded_mask)

    def apply(self, target: Union[np.ndarray, Flow], padding: list = None, cut: bool = None) -> Union[np.ndarray, Flow]:
        """Applies the flow to the target, which can be a numpy array or a Flow object.

        :param target: Numpy array or flow object the flow should be applied to
        :param padding: If flow applied only covers part of the target; [top, bot, left, right]; default None
        :param cut: If padding is given, whether the input is returned as cut to size of flow; default True
        :return: An object of the same type as the input (numpy array, or flow)
        """

        if not isinstance(cut, bool) and cut is not None:
            raise TypeError("Error applying flow: Cut needs to be a boolean or convert to one")
        cut = False if cut is None else cut

        # Determine whether the target is a flow object or not, if so, get actual array to warp
        if isinstance(target, Flow):
            return_flow = True
            # So the flow vectors and mask are warped in one step
            t = np.concatenate((target.vecs, target.mask[..., np.newaxis]), axis=-1)
        else:
            return_flow = False
            if not isinstance(target, np.ndarray):
                raise ValueError("Error applying flow: Target needs to be either a flow object, or a numpy ndarray")
            t = target

        # Determine flow to use for warping, and warp
        if padding is None:
            if not target.shape[:2] == self.shape[:2]:
                raise ValueError("Error applying flow: Flow and target have to have the same shape")
            warped_t = apply_flow(self.vecs, t, self.ref, self.mask)
        else:
            padding = get_valid_padding(padding, "Error applying flow: ")
            if self.shape[0] + np.sum(padding[:2]) != target.shape[0] or \
                    self.shape[1] + np.sum(padding[2:]) != target.shape[1]:
                raise ValueError("Error applying flow: Padding values do not match flow and target size difference")
            flow = self.pad(padding)
            warped_t = apply_flow(flow.vecs, t, flow.ref, flow.mask)

        # Cut if necessary
        if padding is not None and cut:
            warped_t = warped_t[padding[0]:padding[0] + self.shape[0], padding[2]:padding[2] + self.shape[1]]

        # Return as correct type
        if return_flow:
            return Flow(warped_t[..., :2], target.ref, np.round(warped_t[..., 2]).astype('bool'))
        else:
            return warped_t

    def visualise(
            self,
            mode: str,
            show_mask: bool = None,
            show_mask_borders: bool = None,
            range_max: float = None
    ) -> np.ndarray:
        """Returns a flow visualisation as a numpy array containing an rgb / bgr / hsv image of the same size as the flow

        :param mode: Output mode, options: 'rgb', 'bgr', 'hsv'
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param range_max: Maximum vector magnitude expected, corresponding to the HSV maximum Value of 255 when scaling
            the flow magnitudes. Defaults to the 99th percentile of the current flow field
        :return: Numpy array containing the flow visualisation as an rgb / bgr / hsv image of the same size as the flow
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
        range_max = np.percentile(mag, 99) if range_max is None else range_max
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
        """Visualises the flow as arrowed lines
        """Visualises the flow as arrowed lines, in BGR mode

        :param grid_dist: Integer of the distance of the flow points to be used for the visualisation
        :param img: Numpy array with the background image to use, defaults to black
        :param img: Numpy array with the background image to use (in BGR mode), defaults to black
        :param scaling: Float or int of the flow line scaling, defaults to scaling the 99th percentile of arrowed line
            lengths to be equal to twice the grid distance (empirical value)
        :param show_mask: Boolean determining whether the flow mask is visualised, defaults to False
        :param show_mask_borders: Boolean determining whether the flow mask border is visualised, defaults to False
        :param colour: Tuple of the flow arrow colour, defaults to hue based on flow direction as in visualise()
        :return: Numpy array of the flow visualised as arrowed lines, of the same size as the flow
        :return: Numpy array of the flow visualised as arrowed lines, of the same size as the flow, in BGR
        """

        # Validate arguments
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
