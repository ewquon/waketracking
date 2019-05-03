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
from matplotlib.animation import FuncAnimation
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

class waketracker(object):
    """A general class for wake tracking operations.

    Tracking algorithms are expected to be implemented as children of
    this class.
    """

    def __init__(self,*args,**kwargs):
        """Process structured data in rotor-aligned frames of reference.
        Arguments may be in the form:

            waketracker(x, y, z, u, ...)

        or

            waketracker((x,y,z,u), ...)

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
        self.plot_initialized = False

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
        self.x = xdata
        self.y = ydata
        self.z = zdata
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
        assert(len(udata.shape) in (3,4))
        assert((self.Nh,self.Nv) == udata.shape[1:3])
        self.Ntimes = udata.shape[0]
        if len(udata.shape)==3: # SCALAR data
            self.datasize = 1
            self.u_tot = udata  # shape=(Ntimes,Nh,Nv)
            self.v_tot = np.zeros(udata.shape)
            self.w_tot = np.zeros(udata.shape)
        else: # VECTOR data assumed
            self.datasize = udata.shape[3]
            assert(self.datasize == 3)
            if all(self.norm == [1,0,0]):
                self.u_tot = udata[:,:,:,0]
#                self.v_tot = udata[:,:,:,1]
#                self.w_tot = udata[:,:,:,2]
            else:
                # calculate velocities in the sampling plane frame of reference
                self.u_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
#                self.v_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
#                self.w_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
                for itime in range(self.Ntimes):
                    for ih in range(self.Nh):
                        for iv in range(self.Nv):
                            # normal velocity
                            self.u_tot[itime,ih,iv] = \
                                    udata[itime,ih,iv,:].dot(self.norm)
#                            # horizontal velocity
#                            self.v_tot[itime,ih,iv] = \
#                                    udata[itime,ih,iv,:].dot(self.horz)
#                            # vertical velocity (expect norm[2]==0, vert=[0,0,1])
#                            self.w_tot[itime,ih,iv] = \
#                                    udata[itime,ih,iv,2]

        self.u = self.u_tot  # in case input u already has shear removed
#        self.v = self.v_tot
#        self.w = self.w_tot

        self.xwake = np.zeros(self.Ntimes)
        self.ywake = np.zeros(self.Ntimes)
        self.zwake = np.zeros(self.Ntimes)
        self.xh_wake = np.zeros(self.Ntimes)
        self.xv_wake = np.zeros(self.Ntimes)

        self.paths = self.Ntimes*[None]

        if self.verbose:
            print('Number of time frames to process: {}'.format(self.Ntimes))
            print('\n...finished initializing waketracker')

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

    def remove_shear(self,method='fringe',Navg=None,wind_profile=None,
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
            if isinstance(wind_profile, (list,tuple,np.ndarray)):
                assert(wind_profile.shape[-1] == self.Nv)
                self.Uprofile = wind_profile
            elif isinstance(wind_profile, str):
                #zref,Uref = readRef(wind_profile)
                #self.Uprofile = np.interp(self.z,zref,Uref)
                self.Uprofile = np.loadtxt(wind_profile)
                print('Wind profile read from {}'.format(wind_profile))

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
        self.plot_initialized = False
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

        np.savetxt(fname, data, fmt=fmtlist)

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
            data = np.loadtxt(fname)
        except IOError:
            print('Failed to read',fname)
            return None

        if not len(data) == self.Ntimes:
            print('Incorrect number of time steps in',fname)
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
            self.plot_initialized = False
        cid = self.fig.canvas.mpl_connect('close_event', handle_close)

        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])

        self.ax.set_xlim(self.xh_range[0], self.xh_range[-1])
        self.ax.set_ylim(self.xv_range[0], self.xv_range[-1])
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect('equal',adjustable='box',anchor='C')

        self.ax.tick_params(axis='both', labelsize=12, size=10)
        self.ax.set_xlabel(r'$y$ [m]', fontsize=14)
        self.ax.set_ylabel(r'$z$ [m]', fontsize=14)

    def clear_plot(self):
        """Resets all saved plot handles and requires reinitialization
        the next time plot_contour is called.
        """
        #if hasattr(self,'fig') and self.fig is not None:
        #    plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.plot_initialized = False

    def plot_contour(self,
                    itime=0,
                    vmin=None,vmax=None,
                    cmap='viridis',
                    markercolor='w',
                    outline=False,
                    figsize=(8,6),
                    writepng=False,outdir='.',seriesname='U',
                    dpi=100):
        """Plot/update contour and center marker in the rotor-aligned
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

        if not self.plot_initialized:
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

            self.plot_initialized = True

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
        """Helper function for plotting a representation of the wake
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
        """Write out all snapshots to ${outdir}.

        See plot_contour for keyword arguments.
        """
        if not self.wake_tracked:
            print('Note: wake tracking has not been performed; wake centers will not be plotted.')
        for itime in range(self.Ntimes):
            self.plot_contour(itime,writepng='True',**kwargs)


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


class contourwaketracker(waketracker):
    """Class for wake tracking based on (velocity) contours

    Inherits superclass waketracker
    """

    def __init__(self,*args,**kwargs):
        if python_version < 3:
            super(contourwaketracker,self).__init__(*args,**kwargs)
        else:
            super().__init__(*args,**kwargs)

        self.Clevels = np.zeros(self.Ntimes)
        self.Cfvals = np.zeros(self.Ntimes)

        if self.verbose:
            print('\n...finished initializing contourwaketracker')

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
        be called from contourwaketracker.

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
            data = super(contourwaketracker,self)._read_trajectory(fname)
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
    """Class for plotting wakes and their identified centers and
    outlines.
    """
    def __init__(self,y,z,u,
                 figsize=(8,6),dpi=100,
                 vmin=None,vmax=None,
                 cmap='gray',
                ):
        """
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
        self.wakes = OrderedDict()
        self.centers = OrderedDict()
        self.outlines = OrderedDict()
        self.y = y
        self.z = z
        if len(u.shape) == 4:
            self.u = u[:,:,:,0] # (Nt,Ny,Nz,3)
        else:
            self.u = u
        # create basic plot elements
        self.fig, self.ax = plt.subplots(figsize=figsize,dpi=dpi)
        blank = np.empty(y.shape)
        blank.fill(np.nan)
        if vmin is None:
            vmin = np.nanmin(self.u)
        if vmax is None:
            vmax = np.nanmax(self.u)
        self.bkg = self.ax.pcolormesh(self.y, self.z, blank,
                                      cmap=cmap,vmin=vmin,vmax=vmax,)
        self.cbar = self.fig.colorbar(self.bkg)
        self.cbar.set_label(label=r'$U$ [m/s]',fontsize='x-large')
        self.cbar.ax.tick_params(labelsize='x-large')
        self.ax.axis('equal')
        self.ax.set_xlabel('y [m]')
        self.ax.set_ylabel('z [m]')

    def add(self,name,wake,color=None,center=True,outline=True,
            marker='+',markersize=14,markerwidth=2,markeralpha=1.0,
            outlinewidth=3,outlinealpha=0.5,
           ):
        """Add wake object to visualize"""
        # set up styles
        self.wakes[name] = wake
        if color is None:
            color = colors[len(self.wakes.keys())-1]
        # add plot objects
        if center:
            self.centers[name], = self.ax.plot([],[],marker,color=color,
                                               markersize=markersize,
                                               markeredgewidth=markerwidth,
                                               alpha=markeralpha,
                                               label=name)
        else:
            self.centers[name] = None
        if outline:
            self.outlines[name], = self.ax.plot([],[],color=color,
                                                lw=outlinewidth,
                                                alpha=outlinealpha,
                                                label=name)
        else:
            self.outlines[name] = None

    def set_range(self,minmax):
        """Set the contour level limits"""
        self.bkg.set_clim(minmax)

    def legend(self,markers=True,**kwargs):
        """Display legend based on markers or outlines"""
        if markers:
            handles = [h for _,h in self.centers.items()]
        else:
            handles = [h for _,h in self.outlines.items()]
        self.lgd = self.ax.legend(handles, self.wakes.keys(), **kwargs)

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

    def plot(self,itime):
        """Updates plot with axes objects corresponding to the specified
        time frame (can be used with FuncAnimation).
        """
        bkgdata = np.ma.masked_invalid(self.u[itime,:-1,:-1])
        self.bkg.set_array(bkgdata.ravel())
        updated = [self.bkg]
        for name,wake in self.wakes.items():
            if self.centers[name] is not None:
                self.centers[name].set_data(wake.xh_wake[itime],
                                            wake.xv_wake[itime]) 
                updated.append(self.centers[name])
            if self.outlines[name] is not None:
                self.outlines[name].set_data(wake.paths[itime][:,0],
                                             wake.paths[itime][:,1]) 
                updated.append(self.outlines[name])
        self.ax.set_title('itime = {:d}'.format(itime))
        return tuple(updated)

    def animate(self,frames=None):
        """Wrapper around FuncAnimation"""
        if frames is None:
            frames = len(self.u)
        return FuncAnimation(self.fig, self.plot, frames=frames,
                             init_func=self.init_plot, blit=True)
