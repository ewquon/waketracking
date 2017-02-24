*****
Usage
*****

Data Formats
============

talk about data loaders class here

regular data

template, etc


Raw Data
--------

OpenFOAM array data
-------------------


Wake Processing
===============

#. Import appropriate modules.

.. code-block:: python

    from waketracking.dataloaders import foam_ensight_array
    from waketracking.waketrackers import track

#. First load the data using the appropriate data loader (if necessary), as discussed in the previous section.

.. code-block:: python

    arraydata = foam_ensight_array(
            outputDir='postProcessing/array.3D',
            prefix='array.3D_U')

#. Since the input data may be three-dimensional, extract a slice of the data (if necessary).

.. code-block:: python

    y,z,u = arraydata.sliceI(0)         # slice by index
    y,z,u = arraydata.slice_at(x=1000.) # slice at location

#. Perform the wake analysis. The *track* function may be called without a specified method to return a list of available tracking methods.

.. code-block:: python

    wake = track(\*args,\*\*kwargs, method='ConstantArea')


