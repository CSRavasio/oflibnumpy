oflibnumpy
==========
A handy python **o**\ ptical **f**\ low **lib**\ rary, based on **NumPy** arrays, that enables
the manipulation and combination of flow fields while keeping track of valid areas (see "Usage").

.. code-block:: python

    import oflibnumpy as of
    # Make a flow field and display it
    shape = (300, 400)
    flow = of.Flow.from_transforms([['rotation', 200, 150, -30]], shape)
    flow.show()

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_rotation.png?token=AG5S37QEFXP3HTKIN4E4TT3ATPBRG
  :width: 200
  :alt: Visualisation of optical flow representing a rotation

.. code-block:: python

    # Combine sequentially with another flow field, display the result
    flow_2 = of.Flow.from_transforms([['translation', 40, 0]], shape)
    result = of.combine_flows(flow, flow_2, mode=3)
    result.show(show_mask=True, show_mask_borders=True)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_translated_rotation.png?token=AG5S37VDNMWDNFKKMNXTVVTATPBP4
  :width: 200
  :alt: Visualisation of optical flow representing a rotation, translated to the right

.. code-block:: python

    result.show_arrows(show_mask=True, show_mask_borders=True)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_translated_rotation_arrows.png?token=AG5S37WIKWJFZDDVRJPUE2TATPBOQ
  :width: 200
  :alt: Visualisation of optical flow representing a rotation, translated to the right

It is mostly code written from scratch, but also contains useful wrappers for specific functions from libraries such as
OpenCV's ``remap``, to integrate them with the custom flow field class introduced by oflibnumpy. Features:

- Provides a custom flow field class for both backwards and forwards ('source' / 'target' based) flow fields
- Provides a number of class methods to create flow fields from lists of affine transforms, or a transformation matrix
- Provides a number of functions to resize the flow field, visualise it, warp images, find necessary image padding
- Allows for three different types of flow field combination operations
- Keeps track of valid flow field areas through said operations


Usage & Documentation
---------------------
The library is fully documented, with usage examples, at


Installation
------------
Oflibnumpy is based on Python>=3.7. Install it by running:

.. code-block::

    pip install oflibnumpy


Contribution & Support
----------------------
- Issue Tracker: https://github.com/RViMLab/oflibnumpy/issues
- Source Code: https://github.com/RViMLab/oflibnumpy


License
-------
Author: Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's College London
(KCL), supervised by:

- Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
  Imaging Sciences (BMEIS) at King's College London (KCL)
- Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK

This code is licensed under the GNU General Public License (GPL), Version 3.
