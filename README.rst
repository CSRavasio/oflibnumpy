Introduction
============
**Oflibnumpy:** a handy python **o**\ ptical **f**\ low **lib**\ rary, based on **NumPy** arrays, that enables
the manipulation and combination of flow fields while keeping track of valid areas (see "Usage"). It is mostly code
written from scratch, but also contains useful wrappers for specific functions from libraries such as OpenCV's
``remap``, to integrate them with the custom flow field class introduced by oflibnumpy. Features:

- Provides a custom flow field class for both backwards and forwards ('source' / 'target' based) flow fields
- Provides a number of class methods to create flow fields from lists of affine transforms, or a transformation matrix
- Provides a number of functions to resize the flow field, visualise it, warp images, find necessary image padding
- Allows for three different types of flow field combination operations
- Keeps track of valid flow field areas through said operations
- Provides alternative functions to avoid the explicit use of the custom flow class, with slightly limited functionality

Note there is an equivalent flow library called Oflibpytorch, mostly based on PyTorch tensors. Its
`code is available on Github`_, and the `documentation is accessible on ReadTheDocs`_.

.. _code is available on Github:  https://github.com/RViMLab/oflibpytorch
.. _documentation is accessible on ReadTheDocs: https://oflibpytorch.rtfd.io


Usage & Documentation
---------------------
A user's guide as well as full documentation of the library is available at ReadTheDocs_. Some quick examples:

.. _ReadTheDocs: https://oflibnumpy.rtfd.io

.. code-block:: python

    import oflibnumpy as of

    shape = (300, 400)
    transform = [['rotation', 200, 150, -30]]

    # Make a flow field and display it
    flow = of.Flow.from_transforms(transform, shape)
    flow.show()

    # Alternative option without using the custom flow class
    flow = of.from_transforms(transform, shape, 't')
    of.show_flow(flow)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_rotation.png
  :width: 50%

**Above:** Visualisation of optical flow representing a rotation

.. code-block:: python

    # Combine sequentially with another flow field, display the result
    flow_2 = of.Flow.from_transforms([['translation', 40, 0]], shape)
    result = flow.combine_with(flow_2, mode=3)
    result.show(show_mask=True, show_mask_borders=True)

    # Alternative option without using the custom flow class
    flow_2 = of.from_transforms([['translation', 40, 0]], shape, 't')
    result = of.combine_flows(flow, flow_2, mode=3, ref='t')
    of.show_flow(result)  # Note: no way to show the valid flow area (see documentation)

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_translated_rotation.png
  :width: 50%

**Above:** Visualisation of optical flow representing a rotation, translated to the right, using the custom flow class

.. code-block:: python

    result.show_arrows(show_mask=True, show_mask_borders=True)

    # Alternative option without using the custom flow class
    of.show_flow_arrows(result, 't')  # Note: again no way to show the valid flow area

.. image:: https://raw.githubusercontent.com/RViMLab/oflibnumpy/main/docs/_static/flow_translated_rotation_arrows.png
  :width: 50%

**Above:** Visualisation of optical flow representing a rotation, translated to the right, using the custom flow class


Installation
------------
Oflibnumpy is based on Python>=3.7. Install it by running:

.. code-block::

    pip install oflibnumpy


Testing
------------
Oflibnumpy contains a large number of tests to verify it is working as intended. Use the command line to navigate
to ``oflibnumpy/tests`` and run the following code:

.. code-block::

    python -m unittest discover .

The tests will take several minutes to run. Successful completion will be marked with ``OK``.


Contribution & Support
----------------------
- Source Code: https://github.com/RViMLab/oflibnumpy
- Issue Tracker: https://github.com/RViMLab/oflibnumpy/issues


License
-------
Copyright (c) 2021 Claudio S. Ravasio, PhD student at University College London (UCL), research assistant at King's
College London (KCL), supervised by:

- Dr Christos Bergeles, PI of the Robotics and Vision in Medicine (RViM) lab in the School of Biomedical Engineering &
  Imaging Sciences (BMEIS) at King's College London (KCL)
- Prof Lyndon Da Cruz, consultant ophthalmic surgeon, Moorfields Eye Hospital, London UK

This code is licensed under the `MIT License`_.

.. _MIT License: https://opensource.org/licenses/MIT

If you use this code, please acknowledge us with the following citation:

.. code-block:: plaintext

    @article{ravasio_oflib,
      title     = {oflibnumpy {\&} oflibpytorch: Optical Flow Handling and Manipulation in Python},
      author    = {Ravasio, Claudio S. and Da Cruz, Lyndon and Bergeles, Christos},
      journal   = {Journal of Open Research Software (JORS)},
      year      = 2021,
      volume    = {9},
      publisher = {Ubiquity Press, Ltd.},
      doi       = {10.5334/jors.380}
    }