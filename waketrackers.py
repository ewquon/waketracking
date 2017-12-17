from __future__ import print_function
import os
import sys
import importlib
import inspect

import numpy as np
from scipy.ndimage import uniform_filter1d  # to perform moving average
from matplotlib._cntr import Cntr  # to process contour data
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatch
import pickle  # to archive path objects containing wake outlines

import contour_functions as contour

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
    trackerList = {}
    modulePath = os.path.dirname(__file__)
    moduleName = os.path.split(modulePath)[-1]
    assert(moduleName == 'waketracking')

    for rootdir,subdirs,filelist in os.walk(modulePath):
        trimdir = rootdir.replace(modulePath,moduleName) # trim path
        pyroot = trimdir.replace(os.sep,'.') # change to python format
        for filename in filelist:
            if filename.endswith('Tracker.py'):
                submodule = pyroot + '.' + filename[:-3]
                mod = importlib.import_module(submodule)
                for name,cls in inspect.getmembers(mod,inspect.isclass):
                    # can have more than one tracker class per module file
                    if cls.__module__ == mod.__name__:
                        trackerList[name] = cls
        
    method = kwargs.get('method',None)
    if method not in trackerList.keys():
        print("Need to specify 'method' as one of:")
        for name,tracker in trackerList.iteritems():
            print('  {:s} ({:s})'.format(name,tracker.__module__))
        return None
    else:
        tracker = trackerList[method]
        if kwargs.get('verbose',True):
            print('Selected Tracker: {}\n'.format(tracker.__name__))
        return tracker(*args,**kwargs)

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
        horzRange,vertRange : tuple, optional
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
        self.wakeTracked = False
        self.shearRemoval = None
        self.Navg = None  # for removing shear
        self.plotInitialized = False

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
        self.hRange = kwargs.get('horzRange',(-1e9,1e9))
        self.vRange = kwargs.get('vertRange',(-1e9,1e9))
        self.jmin = np.argmin(np.abs(self.hRange[0]-self.xh_range-self.y0))
        self.kmin = np.argmin(np.abs(self.vRange[0]-self.xv_range))
        self.jmax = np.argmin(np.abs(self.hRange[1]-self.xh_range-self.y0))
        self.kmax = np.argmin(np.abs(self.vRange[1]-self.xv_range))
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

            # calculate horizontal velocity
#            assert(self.datasize <= 3) 
#            self.u_tot = np.sqrt(u[:,:,:,0]**2 + u[:,:,:,1]**2)

            # calculate velocities in the sampling plane frame of reference
            self.u_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
            self.v_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
            self.w_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
            for itime in range(self.Ntimes):
                for ih in range(self.Nh):
                    for iv in range(self.Nv):
                        # normal velocity
                        self.u_tot[itime,ih,iv] = \
                                udata[itime,ih,iv,:].dot(self.norm)
                        # horizontal velocity
                        self.v_tot[itime,ih,iv] = \
                                udata[itime,ih,iv,:].dot(self.horz)
                        # vertical velocity (expect norm[2]==0, vert=[0,0,1])
                        self.w_tot[itime,ih,iv] = \
                                udata[itime,ih,iv,2]

        self.u = self.u_tot  # in case input u already has shear removed
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
        s += '  Shear removal          : {}\n'.format(self.shearRemoval)
        s += '  Wake tracking complete : {}\n'.format(self.wakeTracked)
        return s

    def averageVelocity(self,Navg=-300):
        """Calculates moving average using
        scipy.ndimage.uniform_filter1d

        Called by removeShear()

        Parameters
        ----------
        Navg : integer, optional
            Number of snapshots over which to average. If Navg < 0,
            average from end of series only.

        Returns
        -------
        uavg : ndarray
            If Navg < 0, uavg.shape==(Nh,Nv); otherwise a moving average
            is return, with uavg.shape==(Ntimes,Nh,Nv).
        """
        if Navg < 0:
            self.uavg = np.mean(self.u_tot[-Navg:,:,:], axis=0)  # shape=(Nh,Nv)
        elif Navg > 0:
            self.uavg = uniform_filter1d(self.u_tot, size=Navg, axis=0, mode='mirror')  # see http://stackoverflow.com/questions/22669252/how-exactly-does-the-reflect-mode-for-scipys-ndimage-filters-work
        else:
            # no averaging performed
            Navg = 1
            self.uavg = self.u_tot
        self.Navg = Navg
        return self.uavg

    def removeShear(self,method='default',Navg=-300,windProfile=None):
        """Removes wind shear from data.

        Calculates self.u from self.u_tot.

        Current supported methods:

        * "default": Estimate from fringes 
        * "specified": Wind profile specified as either a function or an
            array of heights vs horizontal velocity

        Parameters
        ----------
        method : string, optional
            Specify method to remove wind shear, or None to use data as
            is; some methods may require additional keyword arguments
        Navg : integer, optional
            Number of snapshots to average over to obtain an
            instaneous average (when shearRemoval=='default').
            If Navg < 0, average from end of series only.
        """
        Navg = int(Navg)
        self.Navg = Navg
        self.shearRemoval = method

        # determine the wind profile
        if method == 'default':
            if self.verbose:
                print('Estimating velocity profile from fringes of sampling plane with Navg={}'.format(Navg))
            uavg = self.averageVelocity(Navg) # updates self.uavg
            if Navg < 0:
                self.Uprofile = (uavg[0,:] + uavg[-1,:]) / 2  # shape=(Nv)
            else:
                self.Uprofile = (uavg[:,0,:] + uavg[:,-1,:]) / 2 # shape=(Ntimes,Nv)

        elif method == 'specified':
            if windProfile is None:
                print('Need to specify windProfile, shear not removed.')
                return
            if isinstance(windProfile, (list,tuple,np.ndarray)):
                assert(len(windProfile) == self.Nv)
                self.Uprofile = windProfile
            elif isinstance(windProfile, str):
                #zref,Uref = readRef(windProfile)
                #self.Uprofile = np.interp(self.z,zref,Uref) 
                self.Uprofile = np.loadtxt(windProfile)
                print('Wind profile read from {}'.format(windProfile))

        elif method is not None:
            print('Shear removal method ({}) not supported.'.format(method))
            return

        # actually remove shear now
        self.u = np.copy(self.u_tot)  # u_tot.shape==(Ntimes,Nh,Nv)

        if len(self.Uprofile.shape)==1:
            if self.verbose:
                print('  subtracting out constant profile')
            # Uprofile.shape==(Nv)
            for k,umean in enumerate(self.Uprofile):
                self.u[:,:,k] -= umean
        else:
            if self.verbose:
                print('  subtracting out time-varying profile')
            # Uprofile.shape==(Ntimes,Nv)
            for itime in range(self.Ntimes):
                for k,umean in enumerate(self.Uprofile[itime,:]):
                    self.u[itime,:,k] -= umean

    def findCenters(self,
                    trajectoryFile=None,
                    frame='rotor-aligned'):
        self.plotInitialized = False
        print('{} needs to override this function!'.format(self.__class__.__name))
        #self.wakeTracked = True

    def trajectoryIn(self,frame):
        """Returns a tuple with the wake trajectory in the specified frame"""
        if frame == 'inertial':
            return self.xwake, self.ywake, self.zwake
        elif frame == 'rotor-aligned':
            return self.xh_wake, self.xv_wake
        else:
            print('output frame not recognized')

    def fixTrajectoryErrors(self,update=False,istart=0,iend=None):
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

    def _writeData(self,fname,data):
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

    def _readTrajectory(self,fname):
        """Helper function to read trajectory history typically called
        at the beginning of findCenters
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
        self._updateInertial()
        self.wakeTracked = True
        if self.verbose:
            print('Trajectory loaded from',fname)

        return data

    def _writeTrajectory(self,fname,*args):
        """Helper function to write trajectory history"""
        if fname is None: return
        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        data = [self.xh_wake, self.xv_wake]
        for arg in args:
            data.append(arg)
        self._writeData(fname,data)
        if self.verbose:
            print('Wrote out trajectory to',fname)

    def _updateInertial(self):
        """Called after loading/calculating a wake trajectory in the
        rotor-aligned frame to calculate the trajectory in the inertial
        frame.
        """
        ang = self.yaw
        xd = np.mean(self.xd)  # assuming no sampling rotation error, self.xd should be constant
        self.xwake = self.x0 + np.cos(ang)*xd - np.sin(ang)*self.xh_wake
        self.ywake = self.y0 + np.sin(ang)*xd + np.cos(ang)*self.xh_wake
        self.zwake = self.xv_wake

    def _initPlot(self):
        """Set up figure properties here""" 
        if self.verbose: print('Initializing plot')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.fig = plt.figure(figsize=(8,6))

        def handle_close(event):
            self.plotInitialized = False
        cid = self.fig.canvas.mpl_connect('close_event', handle_close)

        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])

        self.ax.set_xlim(self.xh_range[0], self.xh_range[-1])
        self.ax.set_ylim(self.xv_range[0], self.xv_range[-1])
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect('equal',adjustable='box',anchor='C')

        self.ax.xaxis.set_tick_params(size=10)
        self.ax.yaxis.set_tick_params(size=10)

        self.ax.set_xlabel(r'$y (m)$', fontsize=14)
        self.ax.set_ylabel(r'$z (m)$', fontsize=14)

    def clearPlot(self):
        """Resets all saved plot handles and requires reinitialization
        the next time plotContour is called.
        """
        #if hasattr(self,'fig') and self.fig is not None:
        #    plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.plotInitialized = False

    def plotContour(self,
                    itime=0,
                    cmin=None,cmax=None,
                    cmap='jet',
                    markercolor='w',
                    outline=False,
                    writepng=False,outdir='.',seriesname='U',
                    dpi=100):
        """Plot/update contour and center marker in the rotor-aligned
        frame at time ${itime}.
        
        Parameters
        ----------
        itime : integer
            Index of the wake snapshot to plot.
        cmin,cmax : float, optional
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

        outline = outline and self.wakeTracked

        if not self.plotInitialized:
            self._initPlot()  # first time

            if cmin is None:
                cmin = np.min(self.u[itime,:,:])
            if cmax is None:
                cmax = np.max(self.u[itime,:,:])
            self.plot_clevels = np.linspace(cmin, cmax, 100)

            self.plotobj_filledContours = self.ax.contourf(self.xh, self.xv, self.u[itime,:,:],
                                                           self.plot_clevels, cmap=cmap,
                                                           extend='both')

            # add marker for detected wake center
            if self.wakeTracked and markercolor is not None \
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
            cb_ticks = np.linspace(cmin, cmax, 11)
            cb = self.fig.colorbar(self.plotobj_filledContours,
                                   ticks=cb_ticks, label=r'$U \ (m/s)$')

            # add time annotation
            #self.plotobj_txt = self.ax.text(0.98, 0.97,'t={:.1f}'.format(self.t[itime]),
            #        horizontalalignment='right', verticalalignment='center',
            #        transform=self.ax.transAxes)

            self.plotInitialized = True

        else:
            if outline and hasattr(self,'plotobj_wakeOutline'):
                self.plotobj_wakeOutline.remove()

            # update plot
            for i in range( len(self.plotobj_filledContours.collections) ):
                self.plotobj_filledContours.collections[i].remove()
            self.plotobj_filledContours = self.ax.contourf(
                    self.xh, self.xv, self.u[itime,:,:],
                    self.plot_clevels, cmap=cmap, extend='both')

            if self.wakeTracked and markercolor is not None:
                #print('  marker at {} {}'.format(self.xh_wake[itime],self.xv_wake[itime]))
                self.plotobj_ctr.set_data(self.xh_wake[itime], self.xv_wake[itime])
                self.plotobj_crc.set_data(self.xh_wake[itime], self.xv_wake[itime])

        if outline:
            self.plotOutline(itime)

        if writepng:
            fname = os.path.join(
                    outdir,
                    '{:s}_{:05d}.png'.format(seriesname,itime)
                    )
            self.fig.savefig(fname, dpi=dpi)
            print('Saved',fname)


    def plotOutline(self,itime=0,
            lw=2,ls='-',facecolor='none',edgecolor='w',
            **kwargs):
        """Helper function for plotting a representation of the wake
        edge

        Additional plotting style keywords may be specified, e.g.:
            linewidth, linestyle, facecolor, edgecolor,...
        """
        if not self.wakeTracked:
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
        self.plotobj_wakeOutline = mpatch.PathPatch(path,
                                                    lw=lw,ls=ls,
                                                    facecolor=facecolor,
                                                    edgecolor=edgecolor,
                                                    **kwargs)
        plt.gca().add_patch(self.plotobj_wakeOutline)


    def saveSnapshots(self,**kwargs):
        """Write out all snapshots to ${outdir}.

        See plotContour for keyword arguments.
        """ 
        if not self.wakeTracked:
            print('Note: wake tracking has not been performed; wake centers will not be plotted.')
        for itime in range(self.Ntimes):
            self.plotContour(itime,writepng='True',**kwargs)


    def _readOutlines(self,fname):
        """Helper function to read compressed (pickled) outlines"""
        if (fname is None) or (not self.wakeTracked):
            return None

        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        if not fname.endswith('.pkl'):
            fname += '.pkl'

        try:
            self.paths = pickle.load(open(fname,'r'))
        except IOError:
            print('Failed to read',fname)
            return None

        if self.verbose:
            print('Read pickled outlines from',fname)

        return self.paths

    def _writeOutlines(self,fname):
        """Helper function to write compressed (pickled) outlines"""
        if fname is None: return
        if not fname.startswith(self.prefix):
            fname = os.path.join(self.prefix,fname)
        if not fname.endswith('.pkl'):
            fname += '.pkl'
        pickle.dump(self.paths,open(fname,'w'))
        if self.verbose:
            print('Wrote out pickled outlines to',fname)


class contourwaketracker(waketracker):
    """Class for wake tracking based on (velocity) contours
    
    Inherits superclass waketracker
    """

    def __init__(self,*args,**kwargs):
        super(contourwaketracker,self).__init__(*args,**kwargs)

        self.Clevels = np.zeros(self.Ntimes)
        self.Cfvals = np.zeros(self.Ntimes)

        if self.verbose:
            print('\n...finished initializing contourwaketracker')

    def _findContourCenter(self,
                           itime,
                           targetValue,
                           weightedCenter=True,
                           contourClosure=False,
                           Ntest=11,
                           tol=0.01,
                           func=None,
                           field='u_tot',
                           vdcheck=True,
                           debug=True):
        """Helper function that returns the coordinates of the detected
        wake center. Iteration continues in a binary search fashion
        until the difference in contour values is < 'tol'. This *should*
        be called from contourwaketracker.

        If allowOpenContours is True, then open contours are closed
        with segments along the boundaries.

        Sets the following attributes:
        * self.xh_wake[itime]
        * self.xv_wake[itime]
        * self.paths[itime]
        * self.Clevels[itime]
        * self.Cfvals[itime]
        """
        j0,j1 = self.jmin,self.jmax+1
        k0,k1 = self.kmin,self.kmax+1
        usearch = self.u[itime,j0:j1,k0:k1] # velocity deficit contours
        Cdata = Cntr(self.xh[j0:j1,k0:k1],
                     self.xv[j0:j1,k0:k1],
                     usearch)  # contour data object
        Crange = np.linspace(np.min(usearch), 0, Ntest+1)[1:]
        interval = Crange[1] - Crange[0]

        # search statistics:
        #NtraceCalls = 0
        #NfnEvals = 0
        Nrefine = 0

        if func is None:
            testfield = None
        else:
            testfield = getattr(self,field)[itime,:,:]

        Flist = []  # list of evaluated function values
        level = []  # list of candidate contour values
        paths = []  # list of candidate contour paths
        success = True
        converged = False
        while Nrefine == 0 or interval > tol:  # go through search at least once
            Nrefine += 1
            if debug: print('refinement cycle {}'.format(Nrefine))

            # BEGIN search loop
            #vvvvvvvvvvvvvvvvvvvvvvvvvvvv
            for Clevel in Crange:
                if debug: print('  testing contour level {}'.format(Clevel))

                curPathList = contour.getPaths(Cdata,Clevel,closePaths=contourClosure)
                if debug: print('  contour paths found: {}'.format(len(curPathList)))

                if func is None and not vdcheck:
                    # area contours without velocity deficit check
                    # Note: This is MUCH faster, since we don't have to search for interior pts!
                    paths += curPathList
                    level += len(curPathList)*[Clevel]
                    Flist += [ contour.calcArea(path) for path in curPathList ]
                elif func is None:
                    assert(vdcheck)
                    # area contours with velocity deficit check
                    for path in curPathList:
                        fval, corr, avgDeficit = \
                                contour.integrateFunction(path, None,
                                                          self.xh, self.xv, None,
                                                          vd=self.u[itime,:,:])
                        if fval is not None and avgDeficit < 0:
                            paths.append(path)
                            level.append(Clevel)
                            Flist.append(fval)
                else:
                    # flux contours
                    if vdcheck:
                        vd = self.u[itime,:,:]
                    else:
                        vd = None
                    for path in curPathList:
                        fval, corr, avgDeficit = \
                                contour.integrateFunction(path,
                                                          func,
                                                          self.xh, self.xv, testfield,
                                                          vd=vd)
                        #NfnEvals += 1
                        if fval is not None and avgDeficit < 0:
                            paths.append(path)
                            level.append(Clevel)
                            Flist.append(fval)

            # after testing all the candidate contour values...
            if len(Flist) > 0:
                # found at least one candidate contour
                Ferr = np.abs( np.array(Flist) - targetValue )
                idx = np.argmin(Ferr)
                curOptLevel = level[idx]
                if debug:
                    print('target values: {}'.format(Flist))
                    print('current optimum level: {}'.format(level[idx]))
            else:
                # no closed contours within our range?
                yc = self.xh_fail
                zc = self.xv_fail
                success = False
                break
            #^^^^^^^^^^^^^^^^^^^^^^^^^^
            # END search loop

            # update the contour search range
            interval /= 2.
            Crange = np.linspace(curOptLevel-interval,curOptLevel+interval,Ntest)

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
            self.paths[itime] = paths[idx]  # save paths for plotting
            self.Clevels[itime] = level[idx]  # save time-varying contour levels as reference data
            self.Cfvals[itime] = Flist[idx]

            if not weightedCenter == False:
                if weightedCenter == True:
                    func = np.abs
                else:
                    func = weightedCenter # function type
                yc,zc = contour.calcWeightedCenter(paths[idx],
                                                   self.xh,
                                                   self.xv,
                                                   self.u[itime,:,:],
                                                   weightingFunc=func)
            else:
                # geometric center
                yc = np.mean(paths[idx][:,0])
                zc = np.mean(paths[idx][:,1])

        else:
            # tracking failed!
            self.paths[itime] = []
            self.Clevels[itime] = 0
            self.Cfvals[itime] = 0
            yc = self.xh_fail
            zc = self.xv_fail

        self.xh_wake[itime] = yc
        self.xv_wake[itime] = zc

        return yc,zc,info

    def _readTrajectory(self,fname):
        """Helper function to read trajectory history typically called
        at the beginning of findCenters
        """
        data = super(contourwaketracker,self)._readTrajectory(fname)
        if data is not None:
            # assume load was successful
            self.Clevels = data[:,3]
            self.Cfvals = data[:,4]
        return data

