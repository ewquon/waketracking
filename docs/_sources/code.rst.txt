**************
Code Reference
**************

``dataloaders`` module
======================

This module includes all data processing routines to arbitrary scalar or vector data into a standard form. 

.. automodule:: waketracking.dataloaders
    :members:

``waketrackers`` module
=======================

This module defines generic wake-tracking objects with general helper routines. To access the Tracker objects, **the** ``track`` **function should be used**. The classes here provide common functionality but should not be used directly.

.. automodule:: waketracking.waketrackers
    :members:

``Tracker`` modules
===================

These modules (that have the \*WakeTracker suffix) implement the actual wake tracking algorithms.

.. automodule:: waketracking.gaussian
    :members:

.. automodule:: waketracking.contour
    :members:

``contour_functions`` module
============================

This is a helper module for processing contour paths identified by ``matplotlib._cntr.Cntr``.

.. automodule:: waketracking.contour_functions
    :members:

