oflibnumpy
==========
A handy python **o**\ ptical **f**\ low **lib**\ rary, based on **NumPy** arrays, that enables
the manipulation of flow fields:

.. code-block:: python

    import oflibnumpy as of

    # Make a flow field
    shape = (100, 200)
    flow = of.Flow.from_transforms(['rotation', 50, 100, -30], shape)

    # Show the flow field
    flow.show()

    # Combine sequentially with another flow field
    flow_2 = of.Flow.from_transforms(['scaling', 50, 100, 0.8], shape)
    result = of.combine_flows(flow, flow_2, mode=3)

It is mostly code written from scratch, but also contains useful wrappers for specific functions from libraries such as
OpenCV's ``remap``, to integrate them with the custom flow field class introduced by oflibnumpy.


Features
--------
- Provides a custom flow field class for both backwards and forwards ('source' / 'target' based) flow fields
- Provides a number of class methods to create flow fields from lists of affine transforms, or a transformation matrix
- Provides a number of functions to resize the flow field, visualise it, warp images, find necessary image padding
- Allows for three different types of flow field combination operations
- Keeps track of valid flow field areas through said operations


Installation
------------
Install oflibnumpy by running:

.. code-block::

    pip install oflibnumpy


Contribution & Support
----------------------
- Issue Tracker: https://github.com/RViMLab/oflibnumpy/issues
- Source Code: https://github.com/RViMLab/oflibnumpy


License
-------
This code is licensed under the GNU General Public License (GPL), Version 3.
