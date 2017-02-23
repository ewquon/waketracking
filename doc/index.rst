.. WakeTracking documentation master file, created by
   sphinx-quickstart on Wed Feb 22 17:33:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Wake Tracking
====================

Author: `Eliot Quon <mailto:eliot.quon@nrel.gov>`_

This Python module provides a set of tools for:

1. Processing regularly sampled data, either from simulation or measurements; and
2. Identifying rotor wake trajectories by a variety of methods.

Motivation for this work comes from the need for a general and robust methodology for wake identification in both simulated and measured data. Originally, these tools were designed with wind turbine wakes in mind, but they may readily extended to other applications as well. 

This module was developed as part of an NREL Laboratory Directory Research and Development (LDRD) funded project, for the purpose of extracting wake modeling parameters. The PI for the LDRD project, FAST.Farm, is `Jason Jonkman <mailto:jason.jonkman@nrel.gov>`_. More information about FAST.Farm can be found at...

Reference journal article...

.. toctree::
   :maxdepth: 2

   methodology

   usage

   samples

   code

.. Indices and tables
   ==================
   
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
