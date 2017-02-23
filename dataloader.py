import os

import numpy as np

from timeseries import TimeSeries

class sampled_data(object):
    """ Generic regularly sampled data object """

    def __init__(self,
            outputDir='.', prefix=None,
            NX=1, NY=None, NZ=None, datasize=3,
            npzdata='arrayData.npz'
            ):
        """ Attempts to read data with shape (Ntimes,NX,NY,NZ,datasize)
        The $npzdata keyword indicates the compressed npz file to load
        from and save to.

        All inherited readers should call this generic data reader for
        consistency. The object should contain:
        * ts: TimeSeries object with information regarding the location
              of the data (None for raw data)
        * Ntimes: Number of output time directories
        * NX,NY,NZ: Number of points in the x,y,z directions
        * datasize: Dimension of the data (scalar=1, vector=3)
        * x,y,z: Arrays with shape (NX,NY,NZ)
        * data: Array with shape (Ntimes,NX,NY,NZ,datasize)
        """
        self.outputDir = outputDir
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
        self.dataReadFrom = None

        # attempt to load previously processed data
        savepath = os.path.join(outputDir,npzdata)
        try:
            savedarrays = np.load(savepath)
        except IOError:
            savedarrays = dict()
        try:
            self.x = savedarrays['x']
            self.y = savedarrays['y']
            self.z = savedarrays['z']
            self.data = savedarrays['data']
            assert(self.x.shape == self.y.shape == self.z.shape)
            try:
                self.NX, self.NY, self.NZ = self.x.shape
                self.datasize = self.data.shape[4]
                if self.data.shape[4] == datasize \
                        and self.data.shape[1] == self.NX \
                        and self.data.shape[2] == self.NY \
                        and self.data.shape[3] == self.NZ:
                    print 'Loaded compressed array data from',savepath
                    self.dataReadFrom = savepath
            except ValueError:
                print 'Mismatched data'
        except KeyError:
            print savepath,'was not read'

    def __repr__(self):
        if self.datasize==1:
            s = 'Scalar data array'
            s += ' with shape ({:d},{:d},{:d})'.format(self.NX,self.NY,self.NZ)
            if self.ts is not None:
                s += ' in a series with ' + str(self.ts)
        elif self.datasize==3:
            s = 'Vector data array'
            s += ' with shape ({:d},{:d},{:d})'.format(self.NX,self.NY,self.NZ)
            if self.ts is not None:
                s += ' in a series with ' + str(self.ts)
        else:
            s = str(self.datasize)+'-D vector data array'
            s += ' with shape ({:d},{:d},{:d})'.format(self.NX,self.NY,self.NZ)

        if self.ts is not None:
            s += ' in a series with ' + str(self.ts)

        return s

    def _slice(self,i0=None,i1=None,j0=None,j1=None,k0=None,k1=None):
        if i0 is not None and i0==i1:
            print 'Slicing data at i=',i0
            print '  x ~=',np.mean(self.x[i0,:,:])
            x1 = self.y[i0,:,:]
            x2 = self.z[i0,:,:]
            u = self.data[:,i0,:,:,:]
        elif j0 is not None and j0==j1:
            print 'Slicing data at j=',j0
            print '  y ~=',np.mean(self.y[:,j0,:])
            x1 = self.x[:,j0,:]
            x2 = self.z[:,j0,:]
            u = self.data[:,:,j0,:,:]
        elif k0 is not None and k0==k1:
            print 'Slicing data at k=',k0
            print '  z ~=',np.mean(self.z[:,:,k0])
            x1 = self.x[:,:,k0]
            x2 = self.y[:,:,k0]
            u = self.data[:,:,:,k0,:]
        return x1,x2,u

    def slice_at(self,xs=None,ys=None,zs=None):
        """ Returns a set of 2D data (x1,x2,u) nearest to the specified
        slice location.

        * x1,x2: has dimensions (N1,N2)
        * u: has dimensions (Ntimes,N1,N2,datasize)

        Example: for an x-slice (xs is not None), N1=NY and N2=NZ
        """
        if xs is not None:
            i0 = i1 = 0
        elif ys is not None:
            j0 = j1 = 0
        elif zs is not None:
            k0 = k1 = 0
        return _slice(i0,i1,j0,j1,k0,k1)

class foam_ensight_array(sampled_data):
    """ OpenFOAM array sampling data in Ensight format
    
    Inherits superclass sampled_data.
    """

    def __init__(self,*args,**kwargs):
        """ Reads time series data from subdirectories in $outputDir.
        Each time subdirectory should contain a file named $prefix.000.U

        If NY or NZ are set to None, then the array dimensions 
        will be guessed from the data.

        The .mesh files are assumed identical (the mesh is only read
        once from the first directory)
        """
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.dataReadFrom is not None:
            return
        # get time series
        datafile = self.prefix+'.000.U'
        self.ts = TimeSeries(self.outputDir,datafile)
        self.Ntimes = self.ts.Ntimes

        # set convenience variables
        NX = self.NX
        NY = self.NY
        NZ = self.NZ

        # read mesh
        with open(os.path.join(self.ts.dirList[0],self.prefix+'.mesh'),'r') as f:
            for _ in range(8): # skip header
                f.readline()
            N = int(f.readline())
            xdata = np.zeros(3*N)
            for i in range(3*N):
                xdata[i] = float(f.readline())

        self.x = xdata[:N]
        self.y = xdata[N:2*N]
        self.z = xdata[2*N:3*N]
        print 'x range :',np.min(self.x),np.max(self.x)
        print 'y range :',np.min(self.y),np.max(self.y)
        print 'z range :',np.min(self.z),np.max(self.z)

        # detect NY,NZ if necessary
        if NY is None or NZ is None:
            assert(NX==1)
            # detect NY and NZ
            for NY in np.arange(2,N+1):
                NZ = N/NY
                if NZ == float(N)/NY:
                    if np.all(self.y[:NY] == self.y[NY:2*NY]):
                        break
            print 'Detected NY,NZ =',NY,NZ
            self.NX = NX
            self.NY = NY
            self.NZ = NZ
        assert(N == NX*NY*NZ)

        self.x = self.x.reshape((NX,NY,NZ),order='F')
        self.y = self.y.reshape((NX,NY,NZ),order='F')
        self.z = self.z.reshape((NX,NY,NZ),order='F')

        # read data
        data = np.zeros((self.Ntimes,NX,NY,NZ,self.datasize))
        for itime,fname in enumerate(self.ts):
            print 'Processing frame',itime
            u = np.loadtxt(fname,skiprows=4).reshape((self.datasize,N))
            data[itime,:,:,:,0] = u[0,:].reshape((NX,NY,NZ))
            data[itime,:,:,:,1] = u[1,:].reshape((NX,NY,NZ))
            data[itime,:,:,:,2] = u[2,:].reshape((NX,NY,NZ))
        self.data = data
        self.dataReadFrom = os.path.join(self.outputDir,'*',datafile)

        # save data
        savepath = os.path.join(self.outputDir,self.npzdata)
        try:
            np.savez_compressed(savepath,x=self.x,y=self.y,z=self.z,data=self.data)
            print 'Saved compressed array data to',savepath
        except IOError:
            print 'Problem saving array data to',savepath

