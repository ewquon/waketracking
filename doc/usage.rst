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

1. First load the data using the appropriate data loader, as discussed in the previous section.

.. code-block:: python

    arraydata = waketracking.dataloader.foam_ensight_array(
            outputDir='postProcessing/array.3D',
            prefix='array.3D_U')

2. Since the input data may be three-dimensional, extract a slice of the data (if necessary).

.. code-block:: python

    y,z,u = arraydata.sliceI(0)         # slice by index
    y,z,u = arraydata.slice_at(x=1000.) # slice at location

3. Process


