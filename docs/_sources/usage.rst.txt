*****
Usage
*****

Data Formats
============

The waketracking module includes the ``dataloaders`` module for processing differently formatted data into a general form for use with the ``waketrackers`` module. The data are assumed to be regularly sampled (i.e., on a structured grid), otherwise some interpolation will be needed within a custom dataloader. 

One of the purposes of the dataloaders module is to provide a consistent data format for the ``Tracker`` modules to use. Even though the tracking operates on a two-dimensional plane, coordinates in three dimension are retained to handle cases in which the sampling planes are yawed. The user is not expected to directly interact with the data; the inputs should be some data structure depending on the data source (e.g. OpenFOAM), and the outputs will be a time history of the wake center for the selected (sliced) plane. Outputs are calculated in the rotor-aligned frame (flow and plane normal are in the +x direction, with the origin is located at the plane center) or the inertial frame (originally sampled coordinates). 

Raw Data
--------
This dataloader assumes that the data is of the form (y,z,u) stored in a CSV file, one time step per file. The sampled points are then sorted and arranged into a regular grid. The shape of the sampled grid is needed as input.

There is no support for simultaneously reading in multiple time steps (i.e., reading in a time series) at this point.

Pandas DataFrame
----------------
Data are provided in dataframe(s), a data structured based on numpy ndarrays provided by the ``pandas`` package. These data are series of the form (y,z,u).

OpenFOAM array data
-------------------
This dataloader loads a time series wherein the data are saved in time subdirectories, with one time step per subdirectory. Within each subdirectory, there is a solution file and a .mesh file in the Ensight format.

The mesh is assumed to be identical for all time steps. Dimensions may be specified or guessed from the data.


Wake Processing
===============

#. Import appropriate modules.

    .. code-block:: python
    
        from waketracking.dataloaders import foam_ensight_array
        from waketracking.waketrackers import track

#. First load the data using the appropriate data loader (if necessary), as discussed in the previous section. *The 'prefix' keyword is only needed the first time the data is processed. After the initial processing, a numpy npz archive is stored in* ``outputDir``. *Subsequent calls to sampled_data objects will read existing npz archives and the original is not needed.*

    .. code-block:: python
    
        arraydata = foam_ensight_array(
                outputDir='postProcessing/array.3D',
                prefix='array.3D_U' # optional after first time
                )

#. Since the input data may be three-dimensional, extract a slice of the data.

    .. code-block:: python

        # Returns planar coordinates (x,y,z) and time-varying velocities
        #   (u) on a plane with size (Nh,Nv), Nh and Nv are the number
        #   of horizontal and vertical points respectively.
        x,y,z,u = arraydata.sliceI(0)         # slice by index
        x,y,z,u = arraydata.slice_at(x=1000.) # slice at location

#. Create the wake tracking object. The ``track`` function may be called without a specified method to return a list of available tracking methods.

    .. code-block:: python

        # All outputs will go into the $prefix directory
        wake = track(x,y,z,u,
                     horzRange=(1250,1750), #optional
                     vertRange=(0,250), #optional
                     prefix='processedWake',
                     method='ConstantArea')

#. Perform the wake tracking. Removing the wind shear is an optional step, but typically performed to facilitate the wake identification. One way to do this is to take a simple average of the last ``Navg`` steps, then take a spatial average of the profiles from the fringes of the sampling plane. 

   Results are returned in the rotor-aligned sampling plane by default. For a rotor with axis aligned with the x-direction, the rotor-aligned and intertial frames are identical. Note that the 'rotor-aligned' option returns two arrays xh,xv while the 'inertial' option returns three arrays corresponding x,y,z.

    .. code-block:: python

        wake.removeShear(Navg=-300) # average over _last_ 300 samples
        targetValue = 12500. # method-dependent value
        wake.findCenters(targetValue,
                         trajectoryFile='trajectory.dat',
                         outlinesFile='outlines.pkl')

        yr,zr = wake.trajectoryIn('rotor-aligned')
        x,y,z = wake.trajectoryIn('inertial')

#. Visualize the results.

    .. code-block:: python

        # writes out 'processedWake/snapshots/wakeVelocityDeficit_*.png'
        wake.saveSnapshots(outdir='snapshots',
                           seriesname='wakeVelocityDeficit')

