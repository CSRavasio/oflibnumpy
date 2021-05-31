Documentation
=============
While this documentation aims to go beyond a simple listing of parameters and instead attempts to explain some of the
principles behind the functions, please see the section ":ref:`Usage`" for more details and usage examples including
code and flow field visualisations.


Flow Constructors and Operators
-------------------------------
.. autoclass:: oflibnumpy.Flow
    :members: zero, from_matrix, from_transforms, vecs, ref, mask, shape, copy, is_zero
    :special-members: __str__, __getitem__, __add__, __sub__, __mul__, __truediv__, __pow__, __neg__

    .. automethod:: __init__

Manipulating the Flow
---------------------
.. currentmodule:: oflibnumpy
.. automethod:: Flow.resize
.. automethod:: Flow.pad
.. automethod:: Flow.invert
.. automethod:: Flow.switch_ref

Applying the Flow
-----------------
.. currentmodule:: oflibnumpy
.. automethod:: Flow.apply
.. automethod:: Flow.track

Evaluating the Flow
-------------------
.. currentmodule:: oflibnumpy
.. automethod:: Flow.matrix
.. automethod:: Flow.valid_target
.. automethod:: Flow.valid_source
.. automethod:: Flow.get_padding

Visualising the Flow
--------------------
.. currentmodule:: oflibnumpy
.. automethod:: Flow.visualise
.. automethod:: Flow.visualise_arrows
.. automethod:: Flow.show
.. automethod:: Flow.show_arrows

Flow Operations
---------------
.. autofunction:: oflibnumpy.visualise_definition
.. autofunction:: oflibnumpy.combine_flows
