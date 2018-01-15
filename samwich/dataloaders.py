from __future__ import print_function
import sys
import os

import numpy as np

from samwich.timeseries import TimeSeries

class sampled_data(object):
    """Generic regularly sampled data object"""

    def __init__(self,
            outputdir='.', prefix=None,
            NX=1, NY=None, NZ=None, datasize=3,
            npzdata='arrayData.npz',
            interp_holes=False
            ):
        """Attempts to load processed data with shape
        (Ntimes,NX,NY,NZ,datasize).

        I/O and processing of the data should take place in __init__ of
        the derived class.

        All inherited readers should call this generic data reader for
        consistency. The resulting data object should contain:

        * ts: TimeSeries object with information regarding the location
              of the data (None for raw data)
        * Ntimes: Number of output time directories
        * NX,NY,NZ: Number of points in the x,y,z directions
        * datasize: Dimension of the data (scalar=1, vector=3)
        * x,y,z: Arrays with shape (NX,NY,NZ)
        * data: Array with shape (Ntimes,NX,NY,NZ,datasize)

        Parameters
        ----------
        outputdir : string
            Path to directory containing time subdirectories.
        prefix : string, optional
            Data file prefix.
        NX,NY,NZ : integer
            Dimensions of data, which depending on the reader may be 
            detected or read from the data file.
        datasize : integer
            Describes the type of data (scalar=1, vector=3).
        npzdata : string
            The compressed numpy data file to load from and save to.
        interp_holes : boolean, optional
            Attempt to interpolate data onto a regular grid in case
            the input data has sampling errors. This depends on the
            np.unique function to identify coordinates.
        """
        self.outputdir = outputdir
        self.prefix = prefix

        self.Ntimes = 0
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.datasize = datasize

        self.ts = None
        self.x = None
        self.y = None
        self.z = None
        self.data = None
        self.npzdata = npzdata
        self.data_read_from = None

        self.interp_holes = interp_holes
        if interp_holes and NX > 1:
            raise ValueError('Interpolation of holes only implemented for planar data')

        savepath = os.path.join(outputdir,npzdata)
        if not os.path.isfile(savepath):
            return
        # attempt to load previously processed data
        try:
            savedarrays = np.load(savepath)
        except IOError:
            savedarrays = dict()
        # attempt to process loaded data
        try:
            self.x = savedarrays['x']
            self.y = savedarrays['y']
            self.z = savedarrays['z']
            assert(self.x.shape == self.y.shape == self.z.shape)

            self.data = savedarrays['data']
            try:
                self.Ntimes = self.data.shape[0]
                self.NX, self.NY, self.NZ = self.x.shape
                self.datasize = self.data.shape[4]
                if self.data.shape[4] == datasize \
                        and self.data.shape[1] == self.NX \
                        and self.data.shape[2] == self.NY \
                        and self.data.shape[3] == self.NZ:
                    print('Loaded compressed array data from',savepath)
                    self.data_read_from = savepath
            except ValueError:
                print('Mismatched data')
        except KeyError:
            print('Could not read',savepath)

    def __repr__(self):
        if self.datasize==1:
            s = 'Scalar data array'
        elif self.datasize==3:
            s = 'Vector data array'
        else:
            s = str(self.datasize)+'-D vector data array'
        s += ' with shape ({:d},{:d},{:d})'.format(self.NX,self.NY,self.NZ)
        if self.ts is not None:
            s += ' in a series with ' + str(self.ts)
        if self.data_read_from is not None:
            s += ' read from ' + self.data_read_from
        return s

    def _slice(self,i0=None,i1=None,j0=None,j1=None,k0=None,k1=None):
        """Note: This only extracts slices of the array, no
                 interpolation is performed to perform actual slicing
                 through the computational domain. The regularly
                 sampled data should be properly rotated to the rotor-
                 aligned frame.
        """
        if i0 is not None and i0==i1:
            print('Slicing data at i={} x={}'.format(i0,np.mean(self.x[i0,:,:])))
            x0 = self.x[i0,j0:j1,k0:k1]
            x1 = self.y[i0,j0:j1,k0:k1]
            x2 = self.z[i0,j0:j1,k0:k1]
            if self.datasize==1:
                u = self.data[:,i0,:,:,0]
            else:
                u = self.data[:,i0,:,:,:]
        elif j0 is not None and j0==j1:
            print('Slicing data at j={} y={}'.format(j0,np.mean(self.y[:,j0,:])))
            x0 = self.x[i0:i1,j0,k0:k1]
            x1 = self.y[i0:i1,j0,k0:k1]
            x2 = self.z[i0:i1,j0,k0:k1]
            if self.datasize==1:
                u = self.data[:,:,j0,:,0]
            else:
                u = self.data[:,:,j0,:,:]
        elif k0 is not None and k0==k1:
            print('Slicing data at k={} z={}'.format(k0,np.mean(self.z[:,:,k0])))
            x0 = self.x[i0:i1,j0:j1,k0]
            x1 = self.y[i0:i1,j0:j1,k0]
            x2 = self.z[i0:i1,j0:j1,k0]
            if self.datasize==1:
                u = self.data[:,:,:,k0,0]
            else:
                u = self.data[:,:,:,k0,:]
        else:
            raise IndexError('Slicing ranges ambiguous: '+str([i0,i1,j0,j1,k0,k1]))
        return x0,x1,x2,u

    def sliceI(self,i=0):
        """Return slice through the dimension 0.

        This is probably the only slicing that makes sense...

        Returns
        -------
        xh,xv : ndarray
            Planar coordinaates--horizontal and vertical--with the
            dimensions (Nh,Nv).
        u : ndarray
            Velocity array with dimensions (Ntimes,Nh,Nv,datasize).
        """
        if i >= 0 and i < self.NX:
            return self._slice(i0=i,i1=i)
        else:
            raise IndexError('I={:d} outside of range [0,{:d})'.format(i,self.NX))

    def sliceJ(self,j=0):
        """Return slice through the dimension 1

        Warning: Depending on the data sampling set up, this slicing
        probably does not make sense.
        """
        if j >= 0 and j < self.NY:
            return self._slice(j0=j,j1=j)
        else:
            raise IndexError('J={:d} outside of range [0,{:d})'.format(j,self.NY))

    def sliceK(self,k=0):
        """Return slice through the dimension 2

        Warning: Depending on the data sampling set up, this slicing
        probably does not make sense.
        """
        if k >= 0 and k < self.NZ:
            return self._slice(k0=k,k1=k)
        else:
            raise IndexError('K={:d} outside of range [0,{:d})'.format(k,self.NZ))

    def slice_at(self,x=None,y=None,z=None):
        """Create a set of 2D data near/at the specified slice location.

        Returns
        -------
        * x0,x1,x2 : ndarray
            Sampling grid with dimensions (N1,N2); coordinates are in
            the Cartesian reference frame.
        * u : ndarray
            Velocity array with dimensions (Ntimes,N1,N2,datasize).
        """
        if x is not None:
            xmid = self.x[:,self.NY/2,self.NZ/2]
            i0 = np.argmin(np.abs(xmid-x))
            return self.sliceI(i0)
        elif y is not None:
            ymid = self.y[self.NX/2,:,self.NZ/2]
            j0 = np.argmin(np.abs(ymid-y))
            return self.sliceJ(j0)
        elif z is not None:
            zmid = self.z[self.NX/2,self.NY/2,:]
            k0 = np.argmin(np.abs(zmid-z))
            return self.sliceK(k0)
        else:
            raise AttributeError('Need to specify x, y, or z location')

class _template_sampled_data_format(sampled_data):
    """TEMPLATE for other data readers
    
    Inherits superclass sampled_data.
    """
    def __init__(self,*args,**kwargs):
        """DESCRIPTION HERE

        """
        super(self.__class__,self).__init__(*args,**kwargs)

        # get time series
        datafile = 'FILENAME.DAT'
        self.ts = TimeSeries(self.outputdir,datafile)

        # set convenience variables
        NX = self.NX
        NY = self.NY
        NZ = self.NZ

        # read mesh
        self.x = None
        self.y = None
        self.z = None

        # read data
        self.data = None
        self.data_read_from = None


class rawdata(sampled_data):
    """Raw data, e.g., in csv format.

    See superclass sampled_data for more information.
    """
    def __init__(self,fname,NY,NZ=None,
                 skiprows=1,delimiter=','):
        """Reads a single snapshot from the specified file. Data are
        expected to be in xh, xv, and u columns, with xh/xv being the
        horizontal and vertical sample positions in an inertial frame
        of reference and u being the velocity normal to the sampling
        plane.

        Parameters
        ----------
        fname : string
            Path to file.
        NY : integer
            Number of horizontal points.
        NZ : integer, optional
            Number of vertical points; if omitted, assumed equal to NY.
        skiprows : integer, optional
            Number of rows to skip when calling np.loadtxt.
        delimiter : string, optional
            String to use as delimiter when calling np.loadtxt.
        """
        #super(self.__class__,self).__init__(*args,**kwargs)
        if NZ is None:
            NZ = NY
        self.NX = 1  # single plane
        self.NY = NY
        self.NZ = NZ
        self.datasize = 1  # scalar

        self.ts = None # not a time series
        self.Ntimes = 1

        data = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)
        y = data[:,0]
        z = data[:,1]
        u = data[:,2]

        order = np.lexsort((z,y))

        self.x = np.zeros((1,NY,NZ))
        self.y = y[order].reshape((1,NY,NZ))
        self.z = z[order].reshape((1,NY,NZ))
        self.data = u[order].reshape((1,1,NY,NZ,1))  # shape == (Ntimes,NX,NY,NZ,datasize)
        self.data_read_from = None

class planar_data(sampled_data):
    """Pre-processed data, in 2D arrays.

    See superclass sampled_data for more information.
    """
    def __init__(self,datadict,center_x=False,center_y=True):
        """Takes data stored in a dictionary with keys:
            'x', 'y', 'z', 'u', 'v', 'w'
        and returns a sampled_data object. 'x', 'v', and 'w' are
        optional.

        Parameters
        ----------
        datadict : dict
            Dictionary containing 2D arrays.
        center_x : boolean
            Shift center of plane to x=0.
        center_y : boolean
            Shift center of plane to y=0.
        """
        #super(self.__class__,self).__init__(*args,**kwargs)
        self.NX = 1  # single plane
        self.NY, self.NZ = datadict['u'].shape
        self.datasize = 3  # vector

        self.ts = None # not a time series
        self.Ntimes = 1

        self.y = datadict['y'].reshape((1,self.NY,self.NZ))
        self.z = datadict['z'].reshape((1,self.NY,self.NZ))
        try:
            self.x = datadict['x'].reshape((1,self.NY,self.NZ))
        except KeyError:
            self.x = np.zeros((1,self.NY,self.NZ))

        self.data = np.zeros((1,1,self.NY,self.NZ,3))  # shape == (Ntimes,NX,NY,NZ,datasize)
        self.data[0,0,:,:,0] = datadict['u']
        try:
            self.data[0,0,:,:,1] = datadict['v']
        except KeyError: pass
        try:
            self.data[0,0,:,:,2] = datadict['w']
        except KeyError: pass
        self.data_read_from = None

        if center_x:
            self.x -= np.mean(self.x)
        if center_y:
            self.y -= np.mean(self.y)

class pandas_dataframe(sampled_data):
    """Raw data from pandas dataframe(s)
    
    See superclass sampled_data for more information.
    """

    def __init__(self,frames,NY=None,NZ=None,xr=None,refineFactor=None):
        """Reads a single time instance from one or more scans provided
        in pandas' DataFrame format. Data are assumed to be scalar
        fields.

        Parameters
        ----------
        frames : DataFrame, list, or tuple
            Pandas frames containing scan data.
        NY,NZ : integer, optional
            Number of points in each scan direction.
        xr : ndarray, optional
            Range gate distances; if None, then equal unit spacing is
            assumed.
        refineFactor : integer, optional
            Refinement factor for super-resolving (by cubic
            interpolation) the field in the lateral and vertical
            directions.
        """
        self.ts = None # not a time series
        self.Ntimes = 1

        if isinstance(frames,(list,tuple)):
            self.NX = len(frames)
        else:
            self.NX = 1
            frames = [frames]
            xr = [0]
        if xr is None:
            xr = np.arange(self.NX)
        else:
            assert(len(xr) == self.NX)
            print('Specified range gates: {}'.format(xr))
        if NY is None:
            yrange = list(set(frames[0].y.as_matrix()))
            yrange.sort()
            NY = len(yrange)
            print('Detected y: {} {}'.format(NY,yrange))
        if NZ is None:
            zrange = list(set(frames[0].z.as_matrix()))
            zrange.sort()
            NZ = len(zrange)
            print('Detected z: {} {}'.format(NZ,zrange))

        if refineFactor is None:
            refineFactor = 1
        elif refineFactor > 1:
            from scipy.interpolate import RectBivariateSpline
            refineFactor = int(refineFactor)
            print('Refining input dataframe by factor of {}'.format(refineFactor))
        self.NY = refineFactor * NY
        self.NZ = refineFactor * NZ

        xarray = np.ones((self.NX,self.NY,self.NZ))
        for i,xi in enumerate(xr):
            xarray[i,:,:] *= xi
        self.x = xarray
        
        # sort and interpolate data
        ydata = [ df.y.as_matrix() for df in frames ]
        zdata = [ df.z.as_matrix() for df in frames ]
        udata = [ df.u.as_matrix() for df in frames ]
        self.y = np.zeros((self.NX,self.NY,self.NZ))
        self.z = np.zeros((self.NX,self.NY,self.NZ))
        self.data = np.zeros((1,self.NX,self.NY,self.NZ,1))  # shape == (Ntimes,NX,NY,NZ,datasize)
        for i in range(self.NX):
            order = np.lexsort((zdata[i],ydata[i]))
            ygrid = ydata[i][order].reshape((NY,NZ))
            zgrid = zdata[i][order].reshape((NY,NZ))
            ugrid = udata[i][order].reshape((NY,NZ))
            if refineFactor > 1:
                y0,y1 = np.min(ygrid),np.max(ygrid)
                z0,z1 = np.min(zgrid),np.max(zgrid)
                interpGrid = RectBivariateSpline(ygrid[:,0],
                                                 zgrid[0,:],
                                                 ugrid) # default: 3rd order (cubic)
                ygrid,zgrid = np.meshgrid(np.linspace(y0,y1,self.NY),
                                          np.linspace(z0,z1,self.NZ),
                                          indexing='ij')
                ugrid = interpGrid(ygrid[:,0],zgrid[0,:])
            self.y[i,:,:] = ygrid
            self.z[i,:,:] = zgrid
            self.data[0,i,:,:,0] = ugrid
        self.datasize = 1


#------------------------------------------------------------------------------
# Sampled data cleanup
#
def interp_holes_2d(y,z,verbose=True):
    y0 = y.ravel()
    z0 = z.ravel()
    Norig = len(y0)

    # check for unique points, TODO: may need tolerance for nearly coincident values
    y_uni = np.unique(y)
    z_uni = np.unique(z)
    NY = len(y_uni)
    NZ = len(z_uni)
    if verbose:
        print('Found unique y: {} {}'.format(NY,y_uni))
        print('Found unique z: {} {}'.format(NZ,z_uni))
    # check spacings
    dy = np.diff(y_uni)
    dz = np.diff(z_uni)
    assert(np.max(dy)-np.min(dy) < 0.1) # all spacings should be ~equal
    assert(np.max(dz)-np.min(dz) < 0.1)

    # create the grid we want
    ynew = np.zeros((1,NY,NZ))
    znew = np.zeros((1,NY,NZ))
    ytmp,ztmp = np.meshgrid(y_uni, z_uni, indexing='ij')
    ynew[0,:,:] = ytmp
    znew[0,:,:] = ztmp
    y = ynew.ravel(order='F') # points increase in y, then z
    z = znew.ravel(order='F')
    assert(y[1]-y[0] > 0)

    # find holes
    if verbose: print('Looking for holes in mesh...')
    hole_indices = [] # in new array
    idx_old = 0
    Nholes = 0
    Ndup = 0
    data_map = np.zeros(Norig,dtype=int) # mapping of raveled input array (w/ holes) to new array
    for idx_new in range(NY*NZ):
        if y[idx_new] != y0[idx_old] or z[idx_new] != z0[idx_old]:
            print('  hole at {} {}'.format(y[idx_new],z[idx_new]))
            hole_indices.append(idx_new)
            Nholes += 1
        else:
            data_map[idx_old] = idx_new
            idx_old += 1
            if idx_old >= Norig:
                continue
            # handle duplicate points (not sure why this happens in OpenFOAM sampling...)
            while y[idx_new] == y0[idx_old] and z[idx_new] == z0[idx_old]:
                Ndup += 1
                print('  duplicate point at {} {}'.format(y[idx_new],z[idx_new]))
                data_map[idx_old] = idx_new # map to the same point in the new grid
                idx_old += 1
    assert(idx_old == Norig) # all points mapped
    if verbose:
        print('  {} holes, {} duplicate points'.format(Nholes,Ndup))

    hole_locations = np.stack((y[hole_indices],z[hole_indices])).T

    return ynew, znew, data_map, hole_locations, hole_indices

#------------------------------------------------------------------------------

class foam_ensight_array(sampled_data):
    """OpenFOAM array sampling data in Ensight format
    
    See superclass sampled_data for more information.
    """

    def __init__(self,*args,**kwargs):
        """Reads time series data from subdirectories in ${outputdir}.
        Each time subdirectory should contain a file named
        '${prefix}.000.U'.

        If NY or NZ are set to None, then the array dimensions 
        will be guessed from the data.

        The .mesh files are assumed identical (the mesh is only read
        once from the first directory)
        """
        super(self.__class__,self).__init__(*args,**kwargs)

        if self.prefix is None:
            if self.data_read_from is not None:
                # we already have data that's been read in...
                print("Note: 'prefix' not specified, time series was not read.")
                return
            else:
                raise AttributeError("'prefix' needs to be specified")

        # get time series
        try:
            datafile = self.prefix+'.000.U'
            self.ts = TimeSeries(self.outputdir,datafile,verbose=False)
        except AssertionError:
            if self.data_read_from is not None:
                print('Note: Data read but time series information is unavailable.')
                print('      Proceed at your own risk.')
                return
            else:
                raise IOError('Data not found in '+self.outputdir)

        if self.data_read_from is not None:
            # Previously saved $npzdata was read in super().__init__
            if self.Ntimes == self.ts.Ntimes:
                return
            else:
                print('{} has {} data series, expected {}'.format(
                    self.data_read_from,self.Ntimes,self.ts.Ntimes))

        self.Ntimes = self.ts.Ntimes

        # set convenience variables
        NX = self.NX
        NY = self.NY
        NZ = self.NZ

        # read mesh
        with open(os.path.join(self.ts.dirList[0],self.prefix+'.mesh'),'r') as f:
            for _ in range(8):  # skip header
                f.readline()
            N = int(f.readline())
            xdata = np.zeros(3*N)
            for i in range(3*N):
                xdata[i] = float(f.readline())

        self.x = xdata[:N]
        self.y = xdata[N:2*N]
        self.z = xdata[2*N:3*N]
        print('x range : {} {}'.format(np.min(self.x),np.max(self.x)))
        print('y range : {} {}'.format(np.min(self.y),np.max(self.y)))
        print('z range : {} {}'.format(np.min(self.z),np.max(self.z)))

        # detect NY,NZ if necessary for planar input
        if NY is None or NZ is None:
            assert(NX==1)
            if self.interp_holes:
                interp_points = np.stack((self.y.ravel(),self.z.ravel())).T
                Norig = N
                self.y, self.z, data_map, hole_locations, hole_indices = interp_holes_2d(self.y, self.z)
                # at this point, self.y and self.z have changed
                NX,NY,NZ = self.y.shape
                N = NX*NY*NZ
                # need to update self.x to match self.y and .z in shape
                self.x = self.x[0] * np.ones((NY,NZ))
            else:
                for NY in np.arange(2,N+1):
                    NZ = int(N/NY)
                    if NZ == float(N)/NY:
                        if np.all(self.y[:NY] == self.y[NY:2*NY]):
                            break
                print('Detected NY,NZ = {} {}'.format(NY,NZ))
                if (NZ == 1) or not (NZ == int(N/NY)):
                    print('  Warning: There may be holes in the mesh...')
                    print('           Try running with interp_holes=True')
                assert(N == NX*NY*NZ)
            self.NY = NY
            self.NZ = NZ

        self.x = self.x.reshape((NX,NY,NZ),order='F')
        self.y = self.y.reshape((NX,NY,NZ),order='F')
        self.z = self.z.reshape((NX,NY,NZ),order='F')

        # read data
        data = np.zeros((self.Ntimes,NX,NY,NZ,self.datasize))
        for itime,fname in enumerate(self.ts):
            sys.stderr.write('\rProcessing frame {:d}'.format(itime))
            #sys.stderr.flush()

            if self.interp_holes and Norig < N:
                from scipy.interpolate import LinearNDInterpolator
                u = np.loadtxt(fname,skiprows=4).reshape((self.datasize,Norig))
                interp_values = u.T
                u = np.zeros((self.datasize,N)) # raveled
                # fill new array with known values
                for idx_old,idx_new in enumerate(data_map):
                    # if duplicate points exist, the last recorded value at a
                    #   location will be used
                    u[:,idx_new] = interp_values[idx_old,:]
                # interpolate at holes
                interpfunc = LinearNDInterpolator(interp_points, interp_values)
                uinterp = interpfunc(hole_locations)
                for i in range(3):
                    u[i,hole_indices] = uinterp[:,i]
                # write out new ensight files for debugging
#                pre = fname[:-len('.000.U')]
#                with open(pre+'_NEW.mesh','w') as f:
#                    f.write('foo\nbar\nnode id assign\nelement id assign\npart\n1\ninternalMesh\ncoordinates\n')
#                    f.write(str(N)+'\n')
#                    for xi in self.x.ravel(order='F'):
#                        f.write(' {:g}\n'.format(xi))
#                    for yi in self.y.ravel(order='F'):
#                        f.write(' {:g}\n'.format(yi))
#                    for zi in self.z.ravel(order='F'):
#                        f.write(' {:g}\n'.format(zi))
#                    f.write('point\n')
#                    f.write(str(N)+'\n')
#                    for i in range(1,N+1):
#                        f.write(str(i)+'\n')
#                with open(pre+'_NEW.000.U','w') as f:
#                    f.write('vector\npart\n1\ncoordinates\n')
#                    for i in range(3):
#                        for j in range(N):
#                            f.write(' {:g}\n'.format(u[i,j]))
#                with open(pre+'.case','r') as f1, open(pre+'_NEW.case','w') as f2:
#                    for line in f1:
#                        if self.prefix in line:
#                            f2.write(line.replace(self.prefix,self.prefix+'_NEW'))
#                        else:
#                            f2.write(line)

            else:
                u = np.loadtxt(fname,skiprows=4).reshape((self.datasize,N))

            for i in range(self.datasize):
                data[itime,:,:,:,i] = u[i,:].reshape((NX,NY,NZ),order='F')

        sys.stderr.write('\n')
        self.data = data
        self.data_read_from = os.path.join(self.outputdir,'*',datafile)

        # save data
        if self.npzdata:
            savepath = os.path.join(self.outputdir,self.npzdata)
            try:
                np.savez_compressed(savepath,x=self.x,y=self.y,z=self.z,data=self.data)
                print('Saved compressed array data to',savepath)
            except IOError as e:
                print('Problem saving array data to',savepath)
                errstr = str(e)
                if 'requested' in errstr and errstr.endswith('written'):
                    print('IOError:',errstr)
                    print('Possible known filesystem issue!')
                    print('  Try adding TMPDIR=/scratch/$USER to your environment, or another')
                    print('  path to use for temporary storage that has more available space.')
                    print('  (see https://github.com/numpy/numpy/issues/5336)')


class foam_ensight_array_series(sampled_data):
    """OpenFOAM array sampling data in Ensight format.

    New output format has a single output directory containing a series of .U
    files with a single associated .case and .mesh file.
    
    See superclass sampled_data for more information.
    """

    def __init__(self,*args,**kwargs):
        """Reads time series data from ${prefix}.case file ${outputdir}.
        The output directory should contain ${prefix}.mesh and solution
        samples named ${prefix}.#####.U

        Note: This reader does not use the TimeSeries object.

        If NY or NZ are set to None, then the array dimensions 
        will be guessed from the data.
        """
        super(self.__class__,self).__init__(*args,**kwargs)

        if self.prefix is None:
            self.prefix = os.path.split(self.outputdir)[-1] + '_U'

        # get time series from case file (if available)
        casefile = os.path.join(self.outputdir, self.prefix + '.case')
        Ntimes = -1
        if os.path.isfile(casefile):
            index_start = 0
            index_incr = 0
            with open(casefile,'r') as f:
                f.readline() # FORMAT
                f.readline() # type:
                f.readline() # <blank>
                f.readline() # GEOMETRY
                meshfile = f.readline().split()[-1] # model:
                assert(meshfile == self.prefix + '.mesh')
                f.readline() # <blank>
                f.readline() # VARIABLE
                f.readline() # vector per node:
                f.readline() # TIME
                f.readline() # time set:
                Ntimes = int(f.readline().split()[-1]) # number of steps:
                index_start = int(f.readline().split()[-1]) # filename start number:
                index_incr = int(f.readline().split()[-1]) # filename increment:
                f.readline() # time values:
                tlist = [ float(val) for val in f.readlines() ] # read all remaining lines
            assert(Ntimes > 0)
            assert(Ntimes == len(tlist))
            self.t = np.array(tlist)

            assert(index_incr > 0)
            filelist = [ os.path.join(self.outputdir, self.prefix + '.' + str(idx) + '.U')
                            for idx in index_start+index_incr*np.arange(Ntimes) ]

        if self.data_read_from is not None:
            # Previously saved $npzdata was read in super().__init__
            if Ntimes < 0 or self.Ntimes == Ntimes:
                # no case file to compare against OR number of times read matches casefile "number of steps"
                # ==> we're good, no need to process all data again
                return
            else:
                print('{} has {} data series, expected {}'.format(
                        self.data_read_from,self.Ntimes,Ntimes))

        self.Ntimes = Ntimes

        # set convenience variables
        NX = self.NX
        NY = self.NY
        NZ = self.NZ

        # read mesh
        with open(os.path.join(self.outputdir,meshfile),'r') as f:
            for _ in range(8):  # skip header
                f.readline()
            N = int(f.readline())
            xdata = np.zeros(3*N)
            for i in range(3*N):
                xdata[i] = float(f.readline())

        self.x = xdata[:N]
        self.y = xdata[N:2*N]
        self.z = xdata[2*N:3*N]
        print('x range : {} {}'.format(np.min(self.x),np.max(self.x)))
        print('y range : {} {}'.format(np.min(self.y),np.max(self.y)))
        print('z range : {} {}'.format(np.min(self.z),np.max(self.z)))

        # detect NY,NZ if necessary for planar input
        if NY is None or NZ is None:
            assert(NX==1)
            if self.interp_holes:
                interp_points = np.stack((self.y.ravel(),self.z.ravel())).T
                Norig = N
                self.y, self.z, data_map, hole_locations, hole_indices = interp_holes_2d(self.y, self.z)
                # at this point, self.y and self.z have changed
                NX,NY,NZ = self.y.shape
                N = NX*NY*NZ
                # need to update self.x to match self.y and .z in shape
                self.x = self.x[0] * np.ones((NY,NZ))
            else:
                for NY in np.arange(2,N+1):
                    NZ = int(N/NY)
                    if NZ == float(N)/NY:
                        if np.all(self.y[:NY] == self.y[NY:2*NY]):
                            break
                print('Detected NY,NZ = {} {}'.format(NY,NZ))
                if (NZ == 1) or not (NZ == int(N/NY)):
                    print('  Warning: There may be holes in the mesh...')
                    print('           Try running with interp_holes=True')
                assert(N == NX*NY*NZ)
            self.NY = NY
            self.NZ = NZ

        self.x = self.x.reshape((NX,NY,NZ),order='F')
        self.y = self.y.reshape((NX,NY,NZ),order='F')
        self.z = self.z.reshape((NX,NY,NZ),order='F')

        # read data
        data = np.zeros((self.Ntimes,NX,NY,NZ,self.datasize))
        for itime,fname in enumerate(filelist):
            sys.stderr.write('\rProcessing frame {:d}'.format(itime))
            #sys.stderr.flush()

            if self.interp_holes and Norig < N:
                from scipy.interpolate import LinearNDInterpolator
                u = np.loadtxt(fname,skiprows=4).reshape((self.datasize,Norig))
                interp_values = u.T
                u = np.zeros((self.datasize,N)) # raveled
                # fill new array with known values
                for idx_old,idx_new in enumerate(data_map):
                    # if duplicate points exist, the last recorded value at a
                    #   location will be used
                    u[:,idx_new] = interp_values[idx_old,:]
                # interpolate at holes
                interpfunc = LinearNDInterpolator(interp_points, interp_values)
                uinterp = interpfunc(hole_locations)
                for i in range(3):
                    u[i,hole_indices] = uinterp[:,i]

            else:
                u = np.loadtxt(fname,skiprows=4).reshape((self.datasize,N))

            for i in range(self.datasize):
                data[itime,:,:,:,i] = u[i,:].reshape((NX,NY,NZ),order='F')

        sys.stderr.write('\n')
        self.data = data
        self.data_read_from = casefile

        # save data
        if self.npzdata:
            savepath = os.path.join(self.outputdir,self.npzdata)
            try:
                np.savez_compressed(savepath,x=self.x,y=self.y,z=self.z,data=self.data)
                print('Saved compressed array data to',savepath)
            except IOError as e:
                print('Problem saving array data to',savepath)
                errstr = str(e)
                if 'requested' in errstr and errstr.endswith('written'):
                    print('IOError:',errstr)
                    print('Possible known filesystem issue!')
                    print('  Try adding TMPDIR=/scratch/$USER to your environment, or another')
                    print('  path to use for temporary storage that has more available space.')
                    print('  (see https://github.com/numpy/numpy/issues/5336)')


