from __future__ import print_function
import os
import sys
import importlib
import inspect
import pickle  # to archive path objects containing wake outlines
from collections import OrderedDict

import numpy as np
from scipy import ndimage  # to perform moving average, filtering
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatch
from matplotlib.animation import FuncAnimation, writers
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from samwich.contour_functions import Contours

python_version = sys.version_info[0]

#==============================================================================

def track(*args,**kwargs):
    """Returns the specified waketracker object. If no method is
    specified, then a list of available Tracker objects is printed.

    All arguments and keyword arguments are passed to the resultant
    object.

    Parameters
    ----------
    method : string
        Should correspond to a Tracker class object.
    """
    tracker_list = {}
    module_path = os.path.dirname(__file__)
    module_name = os.path.split(module_path)[-1]

    for rootdir,subdirs,filelist in os.walk(module_path):
        trimdir = rootdir.replace(module_path,module_name)  # trim path
        pyroot = trimdir.replace(os.sep,'.')  # change to python format
        for filename in filelist:
            if filename.endswith('Tracker.py'):
                submodule = pyroot + '.' + filename[:-3]
                mod = importlib.import_module(submodule)
                for name,cls in inspect.getmembers(mod,inspect.isclass):
                    # can have more than one tracker class per module file
                    if cls.__module__ == mod.__name__:
                        tracker_list[name] = cls

    method = kwargs.pop('method',None)
    if method not in tracker_list.keys():
        print("Need to specify 'method' as one of:")
        for name,tracker in tracker_list.items():
            print('  {:s} ({:s})'.format(name,tracker.__module__))
        return list(tracker_list.keys())
    else:
        tracker = tracker_list[method]
        if kwargs.get('verbose',True):
            print('Selected Tracker: {}\n'.format(tracker.__name__))
        if len(args) > 0:
            return tracker(*args,**kwargs)
        else:
            return tracker

#==============================================================================

class WakeTracker(object):
    """A general class for wake tracking operations.

    Tracking algorithms are expected to be implemented as children of
    this class.
    """

    def __init__(self,*args,**kwargs):
        """Process structured data in rotor-aligned frames of reference.
        Arguments may be in the form:

            WakeTracker(x, y, z, u, ...)

        or

            WakeTracker((x,y,z,u), ...)

        The latter form is useful if the datatracker object's slice
        function is called inline.

        Parameters
        ----------
        x,y,z : ndarray
            Sampling grid coordinates in a Cartesian reference frame,
            with shape (Nh,Nv). Used to calculate xh and xv, the
            horizontal and vertical grid coordinates in the rotor-
            aligned frame.
        u : ndarray
            Instantaneous velocity array with shape
            (Ntimes,Nh,Nv,datasize) or (Ntimes,Nh,Nv) for which
            datasize=1 (i.e., scalar field) is assumed. In the scalar
            case, the velocity is assumed normal to the sampling plane;
            in the vector case, the planar normal velocity is
            calculated assuming that the sampling plane is only yawed
            (and not tilted).
        load3D : bool, optional
            If True, load the 3D velocity vector (with v and w
            components); otherwise, just load u. (Default: False)
        horz_range,vert_range : tuple, optional
            Range of points in the horizontal and vertical directions,
            respectively, in the rotor-aligned sampling plane through
            which to search for the wake center
        prefix : string, optional
            Root output directory to save processed data and images.
        verbose : boolean, optional
            Screen output verbosity.
        """
        self.verbose = kwargs.get('verbose',True)
        if len(args) == 0:
            print('Need to specify x,y,z,u!')
            return

        # set initial/default values
        self.wake_tracked = False
        self.shear_removal = None
        self.Navg = None  # for removing shear
        self._plot_initialized = False

        self.prefix = kwargs.get('prefix','.')
        if not os.path.isdir(self.prefix):
            if self.verbose: print('Creating output dir: {}'.format(self.prefix))
            os.makedirs(self.prefix)

        # check and store sampling mesh
        if isinstance(args[0], (list,tuple)):
            xdata = args[0][0]
            ydata = args[0][1]
            zdata = args[0][2]
            udata = args[0][3]
        else:
            xdata = args[0]
            ydata = args[1]
            zdata = args[2]
            udata = args[3]
        assert(xdata.shape == ydata.shape == zdata.shape)
        self.x = xdata.astype(float,copy=True)
        self.y = ydata.astype(float,copy=True)
        self.z = zdata.astype(float,copy=True)
        self.Nh, self.Nv = self.x.shape

        # calculate sampling plane unit normal
        yvec = (self.x[-1, 0] - self.x[ 0,0],
                self.y[-1, 0] - self.y[ 0,0],
                self.z[-1, 0] - self.z[ 0,0])
        zvec = (self.x[-1,-1] - self.x[-1,0],
                self.y[-1,-1] - self.y[-1,0],
                self.z[-1,-1] - self.z[-1,0])
        norm = np.cross(yvec,zvec)
        self.norm = norm / np.sqrt(norm.dot(norm))
        if self.verbose:
            print('Sampling plane normal vector: {}'.format(self.norm))
            if not self.norm[2] == 0:
                print('WARNING: sampling plane is tilted?')

        # calculate horizontal unit vector
        self.vert = np.array((0,0,1))
        self.horz = np.cross(-self.norm, self.vert)

        # setup planar coordinates
        # d: downwind (x)
        # h: horizontal (y)
        # v: vertical (z)
        #self.xh = y  # rotor axis aligned with Cartesian axes
        self.xv = self.z
        ang = np.arctan2(self.norm[1],self.norm[0])  # ang>0: rotating from x-dir to y-dir
        self.yaw = ang

        # get sampling plane centers
        self.x0 = (np.max(self.x) + np.min(self.x))/2
        self.y0 = (np.max(self.y) + np.min(self.y))/2
        self.z0 = (np.max(self.z) + np.min(self.z))/2  # not used (rotation about z only)
        if self.verbose:
            print('  identified plane center at: {} {} {}'.format(self.x0,self.y0,self.z0))

        # clockwise rotation (seen from above)
        # note: downstream coord, xd, not used for tracking
        self.xd =  np.cos(ang)*(self.x-self.x0) + np.sin(ang)*(self.y-self.y0)
        self.xh = -np.sin(ang)*(self.x-self.x0) + np.cos(ang)*(self.y-self.y0)
        if self.verbose:
            print('  rotated to rotor-aligned axes (about z): {} deg'.format(ang*180./np.pi))

        self.xh_range = self.xh[:,0]
        # throw an error if the data are not in the right order, which should be (y,z) and not (z,y)
        if (len(np.unique(self.xh_range))==1):
            print('WARNING: data appears to be of shape (z,y) which will not work! Transpose it to get (y,z).')
            return
        self.xv_range = self.xv[0,:]

        # check plane yaw
        # note: rotated plane should be in y-z with min(x)==max(x)
        #       and tracking should be performed using the xh-xv coordinates
        xd_diff = np.max(self.xd) - np.min(self.xd)
        if self.verbose:
            print('  rotation error: {}'.format(xd_diff))
        if np.abs(xd_diff) > 1e-6:
            print('WARNING: problem with rotation to rotor-aligned frame?')

        # set dummy values in case wake tracking algorithm breaks down
        self.xh_fail = self.xh_range[0]
        self.xv_fail = self.xv_range[0]

        # set up search range
        self.hrange = kwargs.get('horz_range',(-1e9,1e9))
        self.vrange = kwargs.get('vert_range',(-1e9,1e9))
        self.jmin = np.argmin(np.abs(self.hrange[0]-self.xh_range-self.y0))
        self.kmin = np.argmin(np.abs(self.vrange[0]-self.xv_range))
        self.jmax = np.argmin(np.abs(self.hrange[1]-self.xh_range-self.y0))
        self.kmax = np.argmin(np.abs(self.vrange[1]-self.xv_range))
        self.xh_min = self.xh_range[self.jmin]
        self.xv_min = self.xv_range[self.kmin]
        self.xh_max = self.xh_range[self.jmax]
        self.xv_max = self.xv_range[self.kmax]
        if self.verbose:
            print('  horizontal search range: {} {}'.format(self.xh_min,self.xh_max))
            print('  vertical search range: {} {}'.format(self.xv_min,self.xv_max))

        # check and calculate instantaneous velocities including shear,
        # u_tot
        load3D = kwargs.get('load3D',False)
        assert(len(udata.shape) in (3,4))
        assert((self.Nh,self.Nv) == udata.shape[1:3])
        self.Ntimes = udata.shape[0]
        if len(udata.shape)==3:
            # SCALAR data
            self.datasize = 1
            self.u_tot = udata.astype(float,copy=True)  # shape=(Ntimes,Nh,Nv)
            if load3D:
                print('Warning: velocity data does not have 3 components')
        else:
            # VECTOR data (assumed)
            self.datasize = udata.shape[3]
            assert(self.datasize == 3)
            if all(self.norm == [1,0,0]):
                self.u_tot = udata[:,:,:,0].astype(float,copy=True)
                if load3D:
                    self.v_tot = udata[:,:,:,1]
                    self.w_tot = udata[:,:,:,2]
            else:
                # calculate velocities in the sampling plane frame of reference
                self.u_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
                if load3D:
                    self.v_tot = np.zeros(udata.shape)
                    self.w_tot = np.zeros(udata.shape)
                for itime in range(self.Ntimes):
                    for ih in range(self.Nh):
                        for iv in range(self.Nv):
                            # normal velocity
                            self.u_tot[itime,ih,iv] = \
                                    udata[itime,ih,iv,:].dot(self.norm)
                            if load3D:
                                # horizontal velocity
                                self.v_tot[itime,ih,iv] = udata[itime,ih,iv,:].dot(self.horz)
                                # vertical velocity (expect norm[2]==0, vert=[0,0,1])
                                self.w_tot[itime,ih,iv] = udata[itime,ih,iv,2]

        self.u = self.u_tot  # in case input u already has shear removed
        if load3D:
            self.v = self.v_tot
            self.w = self.w_tot

        self.xwake = np.zeros(self.Ntimes)
        self.ywake = np.zeros(self.Ntimes)
        self.zwake = np.zeros(self.Ntimes)
        self.xh_wake = np.zeros(self.Ntimes)
        self.xv_wake = np.zeros(self.Ntimes)

        self.paths = self.Ntimes*[None]

        if self.verbose:
            print('Number of time frames to process: {}'.format(self.Ntimes))
            print('\n...finished initializing WakeTracker')

    def __repr__(self):
        s = 'Tracking '+str(self.Ntimes)+' sampled planes of '
        if self.datasize==1:
            s += 'scalars'
        elif self.datasize==3:
            s += 'vectors'
        else:
            s += str(self.datasize)+'-D data'
        s += ' with shape ({:d},{:d})'.format(self.Nh,self.Nv)
        return s

    def __str__(self):
        s = 'Sampled planes of '
        if self.datasize==1:
            s += 'scalars'
        elif self.datasize==3:
            s += 'vectors'
        else:
            s += str(self.datasize)+'-D data'
        s += ' with shape ({:d},{:d})\n'.format(self.Nh,self.Nv)
        s += '  Number of frames       : {:d}\n'.format(self.Ntimes)
        s += '  Sampling plane yaw     : {:.1f} deg\n'.format(self.yaw*180/np.pi)
        s += '  Shear removal          : {}\n'.format(self.shear_removal)
        s += '  Wake tracking complete : {}\n'.format(self.wake_tracked)
        return s

    def average_velocity(self,Navg=-300):
        """Calculates moving average using
        scipy.ndimage.uniform_filter1d

        Called by remove_shear()

        Parameters
        ----------
        Navg : integer, optional
            Number of snapshots over which to average. If Navg < 0,
            average from end of series only, otherwise a sliding average
            is used.

        Returns
        -------
        uavg : ndarray
            If Navg < 0, uavg.shape==(Nh,Nv); otherwise a moving average
            is return, with uavg.shape==(Ntimes,Nh,Nv).
        """
        Navg = int(Navg)
        if Navg < 0:
            self.uavg = np.mean(self.u_tot[-Navg:,:,:], axis=0)  # shape=(Nh,Nv)
        elif Navg > 0:
            # see http://stackoverflow.com/questions/22669252/how-exactly-does-the-reflect-mode-for-scipys-ndimage-filters-work
            self.uavg = ndimage.uniform_filter1d(self.u_tot,
                                                 size=Navg, axis=0,
                                                 mode='mirror')
        else:
            # no averaging performed
            Navg = 1
            self.uavg = self.u_tot
        self.Navg = Navg
        return self.uavg

    def remove_shear(self,method=None,Navg=None,wind_profile=None,
                     alpha=None,Uref=None,zref=None):
        """Removes wind shear from data.

        Calculates self.u from self.u_tot.

        Current supported methods:

        - "fringe": Estimate from fringes
        - "powerlaw": Generic wind profile with specified shear
        - "specified": Wind profile specified as either a function or an
            array of heights vs horizontal velocity

        Parameters
        ----------
        method : string, optional
            Specify method to remove wind shear, or None to use data as
            is; some methods may require additional keyword arguments
        Navg : integer, optional
            Number of snapshots to average over to obtain an
            instaneous average (when shear_removal=='default').
            If Navg < 0, average from end of series only, otherwise a
            sliding average is performed.
        wind_profile : array-like or str, optional
            An array of mean velocities normal to the sampling plane,
            at the same heights as the sampling grid. The array may have
            the following shape:
            - 1D: constant U(z)
            - 2D: time-varying profile U(t,z)
            - 3D: time-varying inflow U(t,y,z)
            Alternatively, a filename may be specified from which the
            wind profile will be loaded.
        alpha : float, optional
            Shear exponent for power-law wind profile.
        Uref : float, optional
            Reference velocity for power-law wind profile.
        zref : float, optional
            Reference height for power-law wind profile.
        """
        if self.shear_removal is not None:
            print('remove_shear() was already called, doing nothing.')
            return

        if wind_profile is not None:
            method = 'specified'
        elif (alpha is not None) and (Uref is not None) and (zref is not None):
            method = 'powerlaw'
        self.shear_removal = method

        # determine the wind profile
        if method == 'fringe':
            if self.verbose:
                print('Estimating velocity profile from fringes of sampling plane with Navg={}'.format(Navg))
            uavg = self.average_velocity(Navg) # updates self.uavg
            if Navg < 0:
                self.Uprofile = (uavg[0,:] + uavg[-1,:]) / 2  # shape=(Nv)
            else:
                self.Uprofile = (uavg[:,0,:] + uavg[:,-1,:]) / 2 # shape=(Ntimes,Nv)

        elif method == 'specified':
            if wind_profile is None:
                print('Need to specify wind_profile, shear not removed.')
                return
            if isinstance(wind_profile, str):
                # 2D array (i.e., table with height,windspeed columns) read from file
                wind_profile = np.loadtxt(wind_profile)
                print('Wind profile read from {}'.format(wind_profile))
                try:
                    assert len(wind_profile) == self.Nv
                    assert np.all(wind_profile[:,0] == self.xv[0,:])
                except AssertionError:
                    self.Uprofile = np.interp(self.xv[0,:],
                                              wind_profile[:,0],wind_profile[:,1])
                else:
                    self.Uprofile = wind_profile[:,1]
            else:
                if isinstance(wind_profile, list):
                    # only mean velocities given; must match up
                    assert len(wind_profile) == self.Nv
                    self.Uprofile = np.array(wind_profile)
                elif isinstance(wind_profile, np.ndarray):
                    # 1D or 2D array
                    if len(wind_profile.shape) == 1:
                        # 1D array (only mean velocities given; must match up)
                        # - assume input profile at same heights as the grid
                        assert len(wind_profile) == self.Nv
                        self.Uprofile = wind_profile
                    elif wind_profile.shape[1] == 2:
                        # table with height,windspeed columns
                        if np.all(wind_profile[:,0] == self.xv[0,:]):
                            # 2D array, vertical levels match
                            self.Uprofile = wind_profile[:,1]
                        else:
                            # 2D array but vertical levels are not coincident
                            if self.verbose:
                                print('Interpolating freestream profile from',
                                      wind_profile[:,0],'to',self.xv[0,:])
                            self.Uprofile = np.interp(self.xv[0,:],
                                                      wind_profile[:,0],wind_profile[:,1])
                    else:
                        # 2D windspeed array, U(t,z)
                        # - assume innput profile at same heights as the grid
                        self.Uprofile = wind_profile

                elif isinstance(wind_profile, tuple):
                    # two 1D-ndarrays or lists: height, windspeed
                    assert(len(wind_profile[0]) == len(wind_profile[1]))
                    if np.all(wind_profile[0] == self.xv[0,:]):
                        self.Uprofile = np.array(wind_profile[1])
                    else:
                        # vertical levels are not coincident
                        if self.verbose:
                            print('Interpolating freestream profile from',
                                  wind_profile[0],'to',self.xv[0,:])
                        self.Uprofile = np.interp(self.xv[0,:],
                                                  wind_profile[0],wind_profile[1])
                else:
                    raise TypeError('Unexpected wind profile data type')

        elif method == 'powerlaw':
            assert (alpha is not None) and (Uref is not None) and (zref is not None)
            z = np.mean(self.z, axis=0)
            self.Uprofile = Uref * (z/zref)**alpha

        elif method is not None:
            print('Shear removal method ({}) not supported.'.format(method))
            return

        # actually remove shear now
        self.u = np.copy(self.u_tot)  # u_tot.shape==(Ntimes,Nh,Nv)

        if len(self.Uprofile.shape)==1:
            if self.verbose:
                print('  subtracting out profile (constant in time)')
            # Uprofile.shape==(Nv)
            for k,umean in enumerate(self.Uprofile):
                self.u[:,:,k] -= umean
        elif len(self.Uprofile.shape)==2:
            if self.verbose:
                print('  subtracting out time-varying profile')
            # Uprofile.shape==(Ntimes,Nv)
            for itime in range(self.Ntimes):
                for k,umean in enumerate(self.Uprofile[itime,:]):
                    self.u[itime,:,k] -= umean
        elif len(self.Uprofile.shape)==3:
            if self.verbose:
                print('  subtracting out time-varying field')
            # Uprofile.shape==(Ntimes,Nh,Nv)
            self.u -= self.Uprofile
        else:
            print('  unexpected Uprofile data shape--mean was not removed')

    def apply_filter(self, filtertype='gaussian', N=None, size=None, minsize=2):
        """Apply specified filter to the velocity deficit field

        Parameters
        ----------
        filtertype : str, optional
            Name of scipy.ndimage filter to use
        N : int, optional
            Number of points to include in each dimension
        size : float, optional
            Width of the filter in physical dimensions
        """
        if (N is None) and (size is None):
            raise ValueError('Specify N or size')
        elif size is not None:
            dy = np.diff(self.xh_range)
            dz = np.diff(self.xv_range)
            assert(all(np.abs(dy-dy[0]) < 1e-8))
            assert(all(np.abs(dz-dz[0]) < 1e-8))
            dy = dy[0]
            dz = dz[0]
            ds = (dy + dz) / 2
            N = max(int(size/ds), minsize)
        assert(N is not None)
        if self.verbose:
            print('filter size:',N)
        self.u_orig = self.u.copy()
        filterfun = getattr(ndimage,filtertype+'_filter')
        self.u = np.stack([
            filterfun(self.u[itime,:,:], N) for itime in range(self.Ntimes)
        ])

    def find_centers(self,
                    trajectory_file=None,
                    frame='rotor-aligned'):
        self._plot_initialized = False
        print('{} needs to override this function!'.format(self.__class__.__name))
        #self.wake_tracked = True

    def trajectory_in(self,frame):
        """Returns a tuple with the wake trajectory in the specified frame"""
        if frame == 'inertial':
            return self.xwake, self.ywake, self.zwake
        elif frame == 'rotor-aligned':
            return self.xh_wake, self.xv_wake
        else:
            print('output frame not recognized')

    def calculate_areas(self):
        """Calculate the area enclosed by all paths (whether they be
        identified contours or approximate wake shapes) after the wake
        trajectory has been identified.
        """
        assert(self.wake_tracked)
        cntr = Contours(self.xh[self.jmin:self.jmax+1, self.kmin:self.kmax+1],
                        self.xv[self.jmin:self.jmax+1, self.kmin:self.kmax+1])
        self.Careas = np.array([ cntr.calc_area(coords=path)
                                 for path in self.paths ])

    def fix_trajectory_errors(self,update=False,istart=0,iend=None):
        """Some wake detection algorithms are not guaranteed to provide
        a valid trajectory. By default, the coordinates of failed
        detection points is set to be (min(y),min(z)). This routine
        locates points believed to be problem points and interpolates
        between surrounding points. Piecewise linear interpolation
        (np.interp) is used.

        NOT TESTED IN THIS VERSION

        Parameters
        ----------
        update : boolean, optional
            Set to True to update the ywake and zwake in place.
        istart,iend : integer, optional
            The range of values to correct.

        Returns
        -------
        yw_fix,zw_fix : ndarray
            Cleaned-up coordinates
        """
        idx = np.nonzero(
                (self.xh_wake > self.xh_fail) & (self.xv_wake > self.xv_fail)
                )[0]  # indices of reliable values
        ifix0 = np.nonzero(
                (self.xh_wake == self.xh_fail) & (self.xv_wake == self.xv_fail)
                )[0]  # values that need to be corrected
        if iend==None:
            iend = self.Ntimes
        ifix = ifix0[(ifix0 > istart) & (ifix0 < iend)]  # values to be corrected
        tfix = self.t[ifix]  # times to be corrected

        yw_fix = self.ywake.copy()
        zw_fix = self.zwake.copy()
        yw_fix[ifix] = np.interp(tfix, self.t[idx], self.ywake[idx])
        zw_fix[ifix] = np.interp(tfix, self.t[idx], self.zwake[idx])

        print('Interpolated wake centers {} times (method: {:s})'.format(len(tfix),self.wakeTracking))
        if update:
            self.ywake = yw_fix
            self.zwake = zw_fix

        return yw_fix, zw_fix

    def to_MFoR(self,y_mfor,z_mfor,field='u',
                method='RectBivariateSpline',mask_outside=np.nan):
        """Translate wake to meandering frame of reference (MFoR) and
        interpolate to rectangular grid with specified horizontal and
        vertical coordinates.

        If method is 'RectBivariateSpline' and mask_outside is not None,
        then set points outside the interpolation region to this value.
        """
        if not self.wake_tracked:
            print('Need to perform wake tracking first')
        inputfield = getattr(self, field)
        if np.any(np.isnan(inputfield)) and method=='RectBivariateSpline':
            print('NaNs found -- switching interpolation method to griddata')
            method = 'griddata'
        if hasattr(self, 'xh_mfor') or hasattr(self, 'xv_mfor'):
            assert (hasattr(self,'xh_mfor') and hasattr(self, 'xv_mfor'))
            if (~np.all(y_mfor == self.xh_mfor[:,0]) or \
                    ~np.all(z_mfor == self.xv_mfor[0,:])):
                print('Specified rectangular MFoR grid does not match previous call')
        outputfieldname = field + '_mfor'
        if hasattr(self, outputfieldname):
            print('Overwriting previously transformed',outputfieldname)
        # create fields in meandering frame of reference (_mfor)
        self.xh_mfor, self.xv_mfor = np.meshgrid(y_mfor, z_mfor, indexing='ij')
        outputfield = np.empty((self.Ntimes,len(y_mfor),len(z_mfor)))
        # interpolate to regular grid
        if method == 'RectBivariateSpline':
            # complete data (no nans) on structured grid -- faster calculation
            from scipy.interpolate import RectBivariateSpline
            print('Interpolating with',method)
            for itime in range(self.Ntimes):
                yw, zw = self.xh_wake[itime], self.xv_wake[itime]
                interpfun = RectBivariateSpline(self.xh_range-yw,
                                                self.xv_range-zw,
                                                inputfield[itime,:,:])
                outputfield[itime,:,:] = interpfun(y_mfor, z_mfor, grid=True)
                if mask_outside is not None:
                    outside = (
                        (self.xh_mfor > (self.xh_range[-1] - yw)) | 
                        (self.xh_mfor < (self.xh_range[ 0] - yw)) | 
                        (self.xv_mfor > (self.xv_range[-1] - zw)) | 
                        (self.xv_mfor < (self.xv_range[ 0] - zw))
                    )
                    outputfield[itime,outside] = mask_outside
                sys.stderr.write('\rTransform: frame {:d}'.format(itime))
            sys.stderr.write('\n')
        elif method == 'griddata':
            # some missing data, e.g., scan from lidar
            from scipy.interpolate import griddata
            print('Interpolating with',method)
            output = np.stack((self.xh_mfor.ravel(), self.xv_mfor.ravel()), axis=-1)
            for itime in range(self.Ntimes):
                yw, zw = self.xh_wake[itime], self.xv_wake[itime]
                uw = inputfield[itime,:,:]
                notnan = np.where(np.isfinite(uw))
                points = np.stack((self.xh[notnan].ravel()-yw,
                                   self.xv[notnan].ravel()-zw), axis=-1)
                values = uw[notnan].ravel()
                interpout = griddata(points, values, output)
                outputfield[itime,:,:] = interpout.reshape(self.xh_mfor.shape)
                sys.stderr.write('\rTransform: frame {:d}'.format(itime))
            sys.stderr.write('\n')
        else:
            raise ValueError('Unsupported interpolation method: '+method)
        setattr(self, outputfieldname, outputfield)
        # translate paths to mfor as well
        self.paths_mfor = []
        for itime,(yw,zw) in enumerate(zip(self.xh_wake,self.xv_wake)):
            if self.paths[itime] is not None \
                    and (not self.xh_wake[itime] == self.xh_fail) \
                    and (not self.xv_wake[itime] == self.xv_fail):
                newpath = self.paths[itime].copy()
                newpath[:,0] -= self.xh_wake[itime]
                newpath[:,1] -= self.xv_wake[itime]
                self.paths_mfor.append(newpath)
            else:
                self.paths_mfor.append([])

    def _write_data(self,fname,data):
        """Helper function to write out specified data (e.g., trajectory
        and optimization parameters)
        """
        # setup formatting
        Ndata = len(data)
        fmtlist = ['%d'] + Ndata*['%.18e']
        # arrange data
        data = np.vstack((np.arange(self.Ntimes),data)).T
        # make sure path exists
        fpath = os.path.dirname(fname)
        if not os.path.isdir(fpath):
            if self.verbose: print('Creating data subdirectory:',fpath)
            os.makedirs(fpath)

        np.savetxt(fname, data, fmt=fmtlist, delimiter=',')

    def _read_trajectory(self,fname):
        """Helper function to read trajectory history typically called
        at the beginning of find_centers
        """
        if fname is None:
            return None

        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        if not os.path.isfile(fname):
            # tracking probably hasn't been performed before
            return None

        try:
            data = np.loadtxt(fname,delimiter=',')
        except IOError:
            print('Failed to read',fname)
            return None

        if len(data.shape) == 1:
            Nread = 1
            data = data.reshape((1,len(data)))
        else:
            Nread = len(data)
        if not Nread == self.Ntimes:
            print('Incorrect number of time steps in',fname)
            print('  found',Nread,'but expected',self.Ntimes)
            return None

        # data[:,0] is just an index
        self.xh_wake = data[:,1]
        self.xv_wake = data[:,2]
        self._update_inertial()
        self.wake_tracked = True
        if self.verbose:
            print('Trajectory loaded from',fname)

        return data

    def _write_trajectory(self,fname,*args):
        """Helper function to write trajectory history"""
        if fname is None: return
        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        data = [self.xh_wake, self.xv_wake]
        for arg in args:
            data.append(arg)
        self._write_data(fname,data)
        if self.verbose:
            print('Wrote out trajectory to',fname)

    def _update_inertial(self):
        """Called after loading/calculating a wake trajectory in the
        rotor-aligned frame to calculate the trajectory in the inertial
        frame.
        """
        ang = self.yaw
        xd = np.mean(self.xd)  # assuming no sampling rotation error, self.xd should be constant
        self.xwake = self.x0 + np.cos(ang)*xd - np.sin(ang)*self.xh_wake
        self.ywake = self.y0 + np.sin(ang)*xd + np.cos(ang)*self.xh_wake
        self.zwake = self.xv_wake

    def _init_plot(self,figsize):
        """Set up figure properties here"""
        if self.verbose: print('Initializing plot')

        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        self.fig = plt.figure(figsize=figsize)

        def handle_close(event):
            self._plot_initialized = False
        cid = self.fig.canvas.mpl_connect('close_event', handle_close)

        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])

        self.ax.set_xlim(self.xh_range[0], self.xh_range[-1])
        self.ax.set_ylim(self.xv_range[0], self.xv_range[-1])
        self.ax.axis('scaled')

        self.ax.tick_params(axis='both', labelsize=12, size=10)
        self.ax.set_xlabel(r'$y$ [m]', fontsize=14)
        self.ax.set_ylabel(r'$z$ [m]', fontsize=14)

    def clear_plot(self):
        """DEPRECATED: Use Plotter instead
        
        Resets all saved plot handles and requires reinitialization
        the next time plot_contour is called.
        """
        #if hasattr(self,'fig') and self.fig is not None:
        #    plt.close(self.fig)
        self.fig = None
        self.ax = None
        self._plot_initialized = False

    def plot_contour(self,
                    itime=0,
                    vmin=None,vmax=None,
                    cmap='viridis',
                    markercolor='w',
                    outline=False,
                    figsize=(8,6),
                    writepng=False,outdir='.',seriesname='U',
                    dpi=100):
        """DEPRECATED: Use Plotter instead

        Plot/update contour and center marker in the rotor-aligned
        frame at time ${itime}.

        Parameters
        ----------
        itime : integer
            Index of the wake snapshot to plot.
        vmin,vmax : float, optional
            Range of contour values to plot; if None, then set to min
            and max field values.
        cmap : string, optional
            Colormap for the contour plot.
        markercolor : any matplotlib color, optional
            To plot the detected wake center, otherwise set to None.
        outline : boolean, optional
            If true, plot a representation of the detected wake edge.
        writepng : boolean, optional
            If True, save image to
            ${outdir}/${seriesname}_<timeName>.png
        outdir : string, optional
            Output subdirectory.
        seriesname : string, optional
            Prefix for image series (if writepng==True).
        """
        outdir = os.path.join(self.prefix, outdir)
        if writepng and not os.path.isdir(outdir):
            if self.verbose: print('Creating output subdirectory:',outdir)
            os.makedirs(outdir)

        outline = outline and self.wake_tracked

        if not self._plot_initialized:
            self._init_plot(figsize)  # first time

            if vmin is None:
                vmin = np.nanmin(self.u[itime,:,:])
            if vmax is None:
                vmax = np.nanmax(self.u[itime,:,:])
            self.plot_clevels = np.linspace(vmin, vmax, 100)

            self.plotobj_filledcontours = self.ax.contourf(self.xh, self.xv, self.u[itime,:,:],
                                                           self.plot_clevels, cmap=cmap,
                                                           extend='both')

            # add marker for detected wake center
            if self.wake_tracked and markercolor is not None \
                    and (not self.xh_wake[itime] == self.xh_fail) \
                    and (not self.xv_wake[itime] == self.xv_fail):
                self.plotobj_ctr, = self.ax.plot(self.xh_wake[itime],
                                                 self.xv_wake[itime],
                                                 '+', color=markercolor, alpha=0.5,
                                                 markersize=10,
                                                 markeredgewidth=1.)
                self.plotobj_crc, = self.ax.plot(self.xh_wake[itime],
                                                 self.xv_wake[itime],
                                                 'o', color=markercolor, alpha=0.5,
                                                 markersize=10,
                                                 markeredgewidth=1.,
                                                 markeredgecolor=markercolor )

            # add colorbar
            cb_ticks = np.linspace(vmin, vmax, 11)
            cb = self.fig.colorbar(self.plotobj_filledcontours, ticks=cb_ticks)
            cb.set_label(label=r'$U$ [m/s]',fontsize=14)
            cb.ax.tick_params(labelsize=12)

            # add time annotation
            #self.plotobj_txt = self.ax.text(0.98, 0.97,'t={:.1f}'.format(self.t[itime]),
            #        horizontalalignment='right', verticalalignment='center',
            #        transform=self.ax.transAxes)

            self._plot_initialized = True

        else:
            if outline and hasattr(self,'plotobj_wakeoutline'):
                self.plotobj_wakeoutline.remove()

            # update plot
            for i in range( len(self.plotobj_filledcontours.collections) ):
                self.plotobj_filledcontours.collections[i].remove()
            self.plotobj_filledcontours = self.ax.contourf(
                    self.xh, self.xv, self.u[itime,:,:],
                    self.plot_clevels, cmap=cmap, extend='both')

            if self.wake_tracked and markercolor is not None:
                #print('  marker at {} {}'.format(self.xh_wake[itime],self.xv_wake[itime]))
                self.plotobj_ctr.set_data(self.xh_wake[itime], self.xv_wake[itime])
                self.plotobj_crc.set_data(self.xh_wake[itime], self.xv_wake[itime])

        if outline:
            self.plot_outline(itime)

        self.ax.set_title('itime = {:d}'.format(itime))

        if writepng:
            fname = os.path.join(
                    outdir,
                    '{:s}_{:05d}.png'.format(seriesname,itime)
                    )
            self.fig.savefig(fname, dpi=dpi)
            print('Saved',fname)


    def plot_outline(self,itime=0,ax=None,
            lw=2,ls='-',facecolor='none',edgecolor='w',
            **kwargs):
        """DEPRECATED: Use Plotter instead
        
        Helper function for plotting a representation of the wake
        edge

        Additional plotting style keywords may be specified, e.g.:
            linewidth, linestyle, facecolor, edgecolor,...
        """
        if not self.wake_tracked:
            print('Need to perform wake tracking first')
        #if self.verbose: print('Plotting',self.__class__.__name__,'wake outline')
        lw = kwargs.get('linewidth',lw)
        ls = kwargs.get('linestyle',ls)

        try:
            path = mpath.Path(self.paths[itime])
        except ValueError:
            if self.verbose:
                print('No contour available to plot?')
            return
        self.plotobj_wakeoutline = mpatch.PathPatch(path,
                                                    lw=lw,ls=ls,
                                                    facecolor=facecolor,
                                                    edgecolor=edgecolor,
                                                    **kwargs)
        if ax is None:
            ax = plt.gca()
        ax.add_patch(self.plotobj_wakeoutline)


    def save_snapshots(self,**kwargs):
        """DEPRECATED: Use Plotter instead
        
        Write out all snapshots to ${outdir}.

        See plot_contour for keyword arguments.
        """
        if not self.wake_tracked:
            print('Note: wake tracking has not been performed; wake centers will not be plotted.')
        for itime in range(self.Ntimes):
            self.plot_contour(itime,writepng='True',**kwargs)


    def plot(self,itime=0,name='default',MFoR=False,wake_kwargs={},**kwargs):
        """Create a Plotter object for visualization and generating
        animations. This is a convenience function that is equivalent to
        initializing a Plotter and then adding the current wake
        instance.

        A Plotter object is returned, containing fig and ax.

        Parameters
        ----------
        name : str, optional
            Name of the current wake analysis (to be used in labels).
        MFoR : bool, optional
            If True, visualize in the meandering frame of reference.
        wake_kwargs : dict, optional
            Extra arguments that apply to this particular wake.
        kwargs : dict, optional
            Extra arguments with which to initialize the Plotter object.
        """
        if MFoR:
            p = Plotter(self.xh_mfor, self.xv_mfor, self.u_mfor, MFoR=True, **kwargs)
        else:
            p = Plotter(self.xh, self.xv, self.u, **kwargs)
        p.add(name, self,  **wake_kwargs)
        p.plot(itime)
        return p


    def _read_outlines(self,fname):
        """Helper function to read compressed (pickled) outlines"""
        if (fname is None) or (not self.wake_tracked):
            return None

        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        if not fname.endswith('.pkl'):
            fname += '.pkl'

        try:
            self.paths = pickle.load(open(fname,'rb'))
        except IOError:
            print('Failed to read',fname)
            return None

        if self.verbose:
            print('Read pickled outlines from',fname)

        return self.paths

    def _write_outlines(self,fname):
        """Helper function to write compressed (pickled) outlines"""
        if fname is None: return
        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        if not fname.endswith('.pkl'):
            fname += '.pkl'
        pickle.dump(self.paths,open(fname,'wb'))
        if self.verbose:
            print('Wrote out pickled outlines to',fname)


class ContourWakeTracker(WakeTracker):
    """Class for wake tracking based on (velocity) contours

    Inherits superclass WakeTracker
    """

    def __init__(self,*args,**kwargs):
        if python_version < 3:
            super(ContourWakeTracker,self).__init__(*args,**kwargs)
        else:
            super().__init__(*args,**kwargs)

        self.Clevels = np.zeros(self.Ntimes)
        self.Cfvals = np.zeros(self.Ntimes)

        if self.verbose:
            print('\n...finished initializing ContourWakeTracker')

    def _find_contour_center(self,
                             itime,
                             target_value,
                             weighted_center=True,
                             contour_closure=False,
                             min_contour_points=None,
                             Ntest0=20,Ntest=4,
                             umax=0,
                             tol=0.01,
                             func=None,
                             fields=('u_tot'),
                             vdcheck=True,quick_vdcheck=True,
                             vdcutoff=0,
                             debug=True):
        """Helper function that returns the coordinates of the detected
        wake center. Iteration continues in a binary search fashion
        until the difference in contour values is < 'tol'. This *should*
        be called from ContourWakeTracker.

        NOTE: This not guaranteed to find a global optimum.

        If contour_closure is True, then open contours are closed with
        segments along the boundaries.

        Sets the following attributes:
        * self.xh_wake[itime]
        * self.xv_wake[itime]
        * self.paths[itime]
        * self.Clevels[itime]
        * self.Cfvals[itime]
        """
        Ntest0 = int(Ntest0/2)*2 # even numbers guarantees that the same levels aren't reevaluated
        Ntest = int(Ntest/2)*2 # even numbers guarantees that the same levels aren't reevaluated
        j0,j1 = self.jmin,self.jmax+1
        k0,k1 = self.kmin,self.kmax+1
        usearch = self.u[itime,j0:j1,k0:k1] # velocity deficit contours
        Crange = np.linspace(np.nanmin(usearch), umax, Ntest0+1)[1:]
        interval = Crange[1] - Crange[0]
        if debug: print('starting interval:',interval)

        if vdcheck is not False:
            # need reference velocity deficit value
            if isinstance(vdcheck, np.ndarray):
                umin = np.nanmin(vdcheck[itime,:,:])
            else:
                umin = Crange[0]
            vdcutoff *= umin
            assert (vdcutoff <= 0)
            if debug:
                print('reference/cutoff velocity deficit value:',umin,vdcutoff)

        # search statistics:
        #NtraceCalls = 0
        #NfnEvals = 0
        Nrefine = 0

        if func is None:
            testfields = None
        else:
            try:
                func_params = inspect.signature(func).parameters
            except AttributeError:  # python 2
                func_params = inspect.getargspec(func).args
            assert(len(fields) == len(func_params))
            for fieldname in fields:
                assert(hasattr(self,fieldname))
            testfields = [ getattr(self,fieldname)[itime,j0:j1,k0:k1]
                            for fieldname in fields ]

        Flist = []  # list of evaluated function values
        level = []  # list of candidate contour values
        paths = []  # list of candidate contour paths
        success = True
        cur_opt_val = None
        while Nrefine == 0 or interval > tol:  # go through search at least once
            Nrefine += 1
            if debug: print('refinement cycle {}'.format(Nrefine))

            Cdata = Contours(self.xh[j0:j1,k0:k1],
                             self.xv[j0:j1,k0:k1],
                             usearch,
                             umin=Crange[0], umax=Crange[-1],
                             )

            # BEGIN search loop
            #vvvvvvvvvvvvvvvvvvvvvvvvvvvv
            for Clevel in Crange:
                if debug: print('  testing contour level {}'.format(Clevel))

                cur_path_list = Cdata.get_closed_paths(Clevel,
                                                closure=contour_closure,
                                                min_points=min_contour_points,
                                                verbose=debug)
                if debug:
                    print('    contours found: {}'.format(len(cur_path_list)))

                if (func is None) and (vdcheck is False):
                    # area contours _without_ checking the velocity deficit by 
                    # integrating velocities within contours 
                    # - Note: This is MUCH faster, since we don't have to search
                    #   for interior pts!
                    paths += cur_path_list
                    level += len(cur_path_list)*[Clevel]
                    Flist += [Cdata.calc_area(path) for path in cur_path_list]
                elif (func is None) and quick_vdcheck:
                    # area contours with quick velocity deficit check
                    for path in cur_path_list:
                        # - locate geometric center
                        coords = Cdata.to_coords(path,closed=False,array=True)
                        geoctr = coords.mean(axis=0)
                        rdist2 = (geoctr[0]-self.xh)**2 + (geoctr[1]-self.xv)**2
                        jnear,knear = np.unravel_index(np.argmin(rdist2),
                                                       (self.Nh, self.Nv))
                        if isinstance(vdcheck, np.ndarray):
                            vd_est = vdcheck[itime,jnear,knear]
                        else:
                            vd_est = np.mean(usearch[jnear-1:jnear+2,knear-1:knear+2])
                        if debug:
                            print('    velocity deficit near',
                                  self.xh[jnear,knear], self.xv[jnear,knear],
                                  '~=',vd_est)
                        if vd_est < vdcutoff:
                            paths.append(path)
                            level.append(Clevel)
                            area = Cdata.calc_area(path)
                            Flist.append(area)
                elif func is None:
                    # area contours with rigorous velocity deficit check
                    for path in cur_path_list:
                        area, avgdeficit = \
                                Cdata.integrate_function(path, func=None,
                                                         fields=None,
                                                         vd=usearch)
                        if area is not None and avgdeficit < vdcutoff:
                            paths.append(path)
                            level.append(Clevel)
                            Flist.append(area)
                else:
                    # specified function to calculate flux through contour
                    vd = usearch if vdcheck else None
                    for path in cur_path_list:
                        fval, avgdeficit = \
                                Cdata.integrate_function(path, func,
                                                         testfields,
                                                         vd=vd)
                        #NfnEvals += 1
                        if fval is not None and (avgdeficit < 0 or vd is None):
                            paths.append(path)
                            level.append(Clevel)
                            Flist.append(fval)

            # after testing all the candidate contour values...
            if len(Flist) > 0:
                # found at least one candidate contour
                Ferr = np.abs(np.array(Flist) - target_value)
                idx = np.argmin(Ferr)
                cur_opt_level = level[idx]
                if debug:
                    print('target values',
                          ' (after evaluating all candidate contours):')
                    evals = np.arange(len(level))
                    order = np.argsort(level)
                    print('      level, func_val, num_contour_pts')
                    for i in order:
                        print('  ',evals[i],level[i],Flist[i],len(paths[i]))
                    print('current optimum : {} (level={})'.format(Flist[idx],
                                                                   level[idx]))
            else:
                # no closed contours within our range?
                yc = self.xh_fail
                zc = self.xv_fail
                success = False
                break
            #^^^^^^^^^^^^^^^^^^^^^^^^^^
            # END search loop

            if Flist[idx] == cur_opt_val:
                if debug:
                    print('apparent convergence (optimum unchanged)')
                break
            else:
                cur_opt_val = Flist[idx]

            # update the contour search range
            interval /= 2.
            Crange = np.linspace(cur_opt_level-interval, cur_opt_level+interval, Ntest)

            if debug:
                print('new interval: {}'.format(interval))
                print('new Crange: {}'.format(Crange))

        # end of refinement loop
        info = {
                'tolerance': tol,
                'Nrefine': Nrefine,
                #'NtraceCalls': NtraceCalls,
                #'NfnEvals': NfnEvals,
                'success': success
               }

        if success:
            self.paths[itime] = Cdata.to_coords(paths[idx],closed=True,array=True)
            self.Clevels[itime] = level[idx]  # contour levels
            self.Cfvals[itime] = Flist[idx]  # contour function value

            if not weighted_center == False:
                if weighted_center == True:
                    # absolute value of the velocity deficit
                    func = np.abs
                else:
                    # specified weighting function
                    func = weighted_center
                yc,zc = Cdata.calc_weighted_center(paths[idx], weighting_function=func)
            else:
                # geometric center
                yc = np.mean(paths[idx][:,0])
                zc = np.mean(paths[idx][:,1])

        else:
            # tracking failed!
            self.paths[itime] = []
            self.Clevels[itime] = np.nan
            self.Cfvals[itime] = np.nan
            yc = self.xh_fail
            zc = self.xv_fail

        self.xh_wake[itime] = yc
        self.xv_wake[itime] = zc

        return yc,zc,info

    def _read_trajectory(self,fname):
        """Helper function to read trajectory history typically called
        at the beginning of find_centers
        """
        if python_version < 3:
            data = super(ContourWakeTracker,self)._read_trajectory(fname)
        else:
            data = super()._read_trajectory(fname)
        if data is not None:
            # assume load was successful
            self.Clevels = data[:,3]
            self.Cfvals = data[:,4]
            try:
                self.Careas = data[:,5]
            except IndexError:
                pass
        return data


class Plotter(object):
    """Class for plotting one (or more) wakes and their identified
    centers and outlines.
    """

    varnames = {
        'u': r'$u$ [m/s]',
        'utot': r'$u_{tot}$ [m/s]',
    }

    def __init__(self,y=None,z=None,u=None,MFoR=False,
                 figsize=(8,6),dpi=100,
                 vmin=None,vmax=None,
                 cmap='gray',
                ):
        """
        MFoR : bool, optional
            Whether or not plots will be in the meandering frame of
            reference (MFoR)
        figsize : tuple, optional
            Figure size (width,height)
        dpi : int, optional
            Image resolution
        vmin,vmax : float, optional
            Range of contour values to plot; if None, then set to min
            and max field values.
        cmap : string, optional
            Colormap for the contour plot.
        """
        self.MFoR = MFoR
        self.wakes = OrderedDict()
        self.centers = OrderedDict()
        self.outlines = OrderedDict()
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        if (y is not None) and (z is not None) and (u is not None):
            self.y = y
            self.z = z
            if len(u.shape) == 4:
                # 3-D velocity field
                self.u = u[:,:,:,0] # (Nt,Ny,Nz,3)
            else:
                # x-component of velocity only
                self.u = u
            self.Ntimes = self.u.shape[0]
            self._create_figure()
        else:
            self.y = None
            self.z = None
            self.u = None
            self.Ntimes = -1

    def __str__(self):
        s = 'Stored wakes:\n- '
        s += '\n- '.join(self.wakes.keys())
        return s

    def _cell_centers_to_corners(self):
        # setup grid such that the pcolormesh uses points that correspond to
        # the edges of the colormesh and the centers of each colormesh cell
        # correspond to the actual sampled data
        assert np.all(self.y[:,0] == self.y[:,-1])
        assert np.all(self.z[0,:] == self.z[-1,:])
        y1 = (self.y[1:,0] + self.y[:-1,0]) / 2  # midpoints
        z1 = (self.z[0,1:] + self.z[0,:-1]) / 2
        y0 = [2*self.y[0,0] - y1[0]] # y0 - (y1 - y0)
        z0 = [2*self.z[0,0] - z1[0]] # y0 - (y1 - y0)
        yn = [2*self.y[-1,0] - y1[-1]] # yn + (yn - ym)
        zn = [2*self.z[0,-1] - z1[-1]] # yn + (yn - ym)
        self.y1 = np.concatenate((y0,y1,yn))
        self.z1 = np.concatenate((z0,z1,zn))
        yy,zz = np.meshgrid(self.y1,self.z1,indexing='ij')
        return yy,zz

    def _create_figure(self):
        # create basic plot elements
        self.fig, self.ax = plt.subplots(figsize=self.figsize,
                                         dpi=self.dpi)
        blank = np.empty(self.y.shape)
        blank.fill(np.nan)
        if self.vmin is None:
            self.vmin = np.nanmin(self.u)
        if self.vmax is None:
            self.vmax = np.nanmax(self.u)
        yy,zz = self._cell_centers_to_corners()
        self.bkg = self.ax.pcolormesh(yy, zz, blank, cmap=self.cmap,
                                      vmin=self.vmin, vmax=self.vmax,)
        self.cbar = self.fig.colorbar(self.bkg)
        self.cbar.ax.tick_params(labelsize='x-large')
        self.ax.axis('scaled')
        self.ax.set_xlabel('y [m]')
        self.ax.set_ylabel('z [m]')
        self.fig.tight_layout()

    def add(self,name,wake,color=None,center=True,outline=True,
            marker='+',markersize=10,markerwidth=2,markeralpha=1.0,
            linestyle='-',linewidth=3,linealpha=0.5,
           ):
        """Add wake object to visualize"""
        if self.Ntimes < 0:
            self.y = wake.xh
            self.z = wake.xv
            self.u = wake.u
            self.Ntimes = wake.Ntimes
            self._create_figure()
        else:
            assert wake.Ntimes == self.Ntimes
        # set up styles
        self.wakes[name] = wake
        if color is None:
            color = colors[len(self.wakes.keys())-1]
        # add plot objects
        labelstr = name if not outline else ''
        if center and (not self.MFoR):
            self.centers[name], = self.ax.plot([],[],marker,color=color,
                                               markersize=markersize,
                                               markeredgewidth=markerwidth,
                                               alpha=markeralpha,
                                               label=labelstr)
        else:
            self.centers[name] = None
        if outline:
            self.outlines[name], = self.ax.plot([],[],color=color,
                                                ls=linestyle,
                                                lw=linewidth,
                                                alpha=linealpha,
                                                label=name)
        else:
            self.outlines[name] = None

    def set_range(self,minmax):
        """Set the contour level limits"""
        self.bkg.set_clim(minmax)

    def init_plot(self):
        """For FuncAnimation init_func, to clear axes objects"""
        updated = []
        for name,wake in self.wakes.items():
            if self.centers[name] is not None:
                self.centers[name].set_data([],[]) 
                updated.append(self.centers[name])
            if self.outlines[name] is not None:
                self.outlines[name].set_data([],[]) 
                updated.append(self.outlines[name])
        return tuple(updated)

    def plot(self,itime=0,var='u',wakes=None,verbose=False,
            xlim=None,ylim=None,
             **kwargs):
        """Plot selected wakes (or by default, `wakes=None` for all
        wakes) at the specified time"""
        self.update(itime,var,wakes,verbose,**kwargs)
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        # return figure so that it displays within notebook
        return self.fig

    def update(self,itime=0,var='u',wakes=None,verbose=False,**kwargs):
        """Updates plot with axes objects corresponding to the specified
        time frame (can be used with FuncAnimation).
        """
        # update background field
        field = getattr(self,var)
        bkgdata = np.ma.masked_invalid(field[itime,:,:])
        self.bkg.set_array(bkgdata.ravel())
        updated = [self.bkg] # list of updated artists to return for blitting
        # update detected wakes
        if wakes is None:
            # plot all wakes
            self.selected_wakes = self.wakes.keys()
        else:
            # check selected wakes exist
            for name in wakes:
                if not name in self.wakes.keys():
                    print('Specified wake',name,
                          'not in',list(self.wakes.keys()))
            self.selected_wakes = wakes
        # update all wakes (if tracking was performed)
        for name,wake in self.wakes.items():
            if not wake.wake_tracked:
                continue
            if verbose:
                print(name, wake.xh_wake[itime], wake.xv_wake[itime])
            hidden = (not (name in self.selected_wakes))
            if self.centers[name] is not None:
                # update wake centers
                if hidden:
                    self.centers[name].set_data([],[]) 
                else:
                    self.centers[name].set_data(wake.xh_wake[itime],
                                                wake.xv_wake[itime]) 
                updated.append(self.centers[name])
            if self.outlines[name] is not None:
                # update wake outlines
                if hidden or (wake.paths[itime] is None):
                    self.outlines[name].set_data([],[]) 
                elif self.MFoR:
                    self.outlines[name].set_data(wake.paths_mfor[itime][:,0],
                                                 wake.paths_mfor[itime][:,1]) 
                else:
                    self.outlines[name].set_data(wake.paths[itime][:,0],
                                                 wake.paths[itime][:,1]) 
                updated.append(self.outlines[name])
        sys.stderr.write('\rPlot: frame {:d}'.format(itime))
        # formatting
        label = self.varnames.get(var,var)
        self.cbar.set_label(label=label,fontsize='x-large')
        self.ax.set_title('itime = {:d}'.format(itime))
        # save figure (optional)
        if 'fpath' in kwargs.keys():
            fpath = kwargs.pop('fpath').format(itime)
            self.fig.savefig(fpath, **kwargs)
            if verbose:
                print('Wrote',fpath)
        return tuple(updated)

    def legend(self,markers=True,**kwargs):
        """Display legend based on markers or outlines"""
        if markers:
            handles = [
                h for name,h in self.centers.items()
                if name in self.selected_wakes
            ]
        else:
            handles = [
                h for name,h in self.outlines.items()
                if name in self.selected_wakes
            ]
        self.lgd = self.ax.legend(handles, self.wakes.keys(), **kwargs)

    def animation(self,frames=None,fargs=None,**kwargs):
        """Wrapper around FuncAnimation to return a
        `matplotlib.animation.Animation` object
        
        You may need to install additional packages on your system
        (e.g., ffmpeg). If you encounter unexpected animation errors
        after installing the additional packages, you may need to 
        explicitly specify which animation writer to use:
          `plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'`

        Note: to show the animation or save the video, call anim.save()
        or anim.to_html5_video().
        """
        if frames is None:
            frames = len(self.u)
        return FuncAnimation(self.fig, self.update, frames=frames,
                             init_func=self.init_plot, fargs=fargs,
                             blit=True, **kwargs)

    def animate(self,fname,frames=None,bitrate=1000,
                fps=24, writer='ffmpeg', codec='h264',
                extra_args=['-pix_fmt','yuv420p'],
                **kwargs):
        """Convenience function for creating an animation using
        FuncAnimation with some default parameters.
        """
        anim = self.animation(frames=frames)
        writer = writers[writer](fps=fps, bitrate=bitrate, codec=codec,
                                 extra_args=extra_args)
        anim.save(fname, writer=writer, **kwargs)
        return anim

    def savefig(self,*args,**kwargs):
        """Convenience function self.fig.savefig"""
        self.fig.savefig(*args,**kwargs)

    def savefigs(self,frames=None,fpath='snapshot_{:04d}.png',**kwargs):
        """Save all wake snapshots.
        
        This is a convenience function for creating an animation and
        saving each frame using FuncAnimation.
        """
        if frames is None:
            frames = range(len(self.u))
        elif not hasattr(frames,'__iter__'):
            # single snapshot
            frames = [frames]
        # do some sanity checks on the file name
        if len(frames) > 1:
            # multiple files to write out, need a formattable fpath
            if fpath.format(0) == fpath:
                # format did nothing; file name will be the same for
                # every frame
                print("Output fpath '{:s}' is not formattable".format(fpath))
                raise ValueError('Image will be repeatedly overwritten')
        # now loop over all frames:
        for itime in frames:
            self.plot(itime, fpath=fpath, **kwargs)
        
