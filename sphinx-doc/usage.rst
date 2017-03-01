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
        wake = track(x,y,z,u,
                     horzRange=(1250,1750),
                     vertRange=(0,250),
                     prefix='processedWake',
                     method='ConstantArea')

#. Perform the wake tracking. Removing the wind shear is an optional step, but typically performed to facilitate the wake identification. One way to do this is to take a simple average of the last ``Navg`` steps, then take a spatial average of the profiles from the fringes of the sampling plane. 

   Results are returned in the rotor-aligned sampling plane by default. For a rotor with axis aligned with the x-direction, the rotor-aligned and intertial frames are identical. Note that the 'rotor-aligned' option returns two arrays xh,xv while the 'inertial' option returns three arrays corresponding x,y,z.

    .. code-block:: python

        wake.removeShear(Navg=-300)
        targetValue = 12500. # approx rotor area for the ConstantArea method
        wake.findCenters(targetValue,
                         trajectoryFile='trajectory.dat',
                         outlinesFile='outlines.pkl')

#. Visualize the results.

    .. code-block:: python

        # writes out 'processedWake/snapshots/wakeVelocityDeficit_{:d}.png'
        wake.saveSnapshots(outdir='snapshots',seriesname='wakeVelocityDeficit')

