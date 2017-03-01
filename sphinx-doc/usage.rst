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

        # Returns a planar coordinates (x,y,z) and time-varying velocities (u)
        #   on a plane with size (Nh,Nv), Nh and Nv are the number of
        #   horizontal and vertical points respectively.
        x,y,z,u = arraydata.sliceI(0)         # slice by index
        x,y,z,u = arraydata.slice_at(x=1000.) # slice at location

#. Create the wake tracking object. The ``track`` function may be called without a specified method to return a list of available tracking methods.

    .. code-block:: python

        # All outputs will go into the $prefix directory
        wake = track(x,y,z,u,prefix='processedWake',method='ConstantArea')

#. Perform the wake tracking.

    .. code-block:: python

        # Remove the wind shear (optional)
        # --------------------------------
        # Take the time average of the last 300 samples; subtract out the
        #  average of the left- and right-most time-averaged profiles (i.e. the
        #  fringes of the sampling plane).
        wake.removeShear(Navg=-300)

        # Extract the wake centers
        # ------------------------
        # Results returned in the rotor-aligned sampling plane by default.
        # For rotor with axis aligned with the x-direction, the rotor-algned
        # frame and the inertial frames are identical. (Note that the 'inertial'
        # option returns three arrays corresponding to x, y, and z.
        wake.findCenters(12500.,writeTrajectories='trajectory.dat')

#. Visualize the results.

    .. code-block:: python

        # writes out 'processedWake/snapshots/wakeVelocityDeficit_{:d}.png'
        wake.saveSnapshots(outdir='snapshots',seriesname='wakeVelocityDeficit')

