import os
import sys
import importlib
import inspect

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatch

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
    for f in os.listdir(os.path.dirname(__file__)):
        if f.endswith('WakeTracker.py'):
            mod = importlib.import_module('waketracking.'+f[:-3])
            for name,cls in inspect.getmembers(mod,inspect.isclass):
                if cls.__module__ == mod.__name__:
                    trackerList[name] = cls
        
    try:
        method = kwargs['method']
        assert(method in trackerList.keys())
    except (KeyError,AssertionError):
        print "Need to specify 'method' as one of:"
        for name,tracker in trackerList.iteritems():
            print '  {:s} ({:s})'.format(name,tracker.__module__)
        return None
    else:
        tracker = trackerList[method]
        print 'Selected Tracker:',tracker,'\n'
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
        
            waketracker(xh, xv, u, ...)
        
        or

            waketracker((xh,xv,u), ...)

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
            in the vector case, the horizontal velocity is calculated
            from the first two components.
        prefix : string, optional
            Root output directory to save processed data and images.
        verbose : boolean, optional
            Screen output verbosity.
        """
        self.verbose = kwargs.get('verbose',True)
        if len(args) == 0:
            print 'Need to specify x,y,z,u!'
            return

        # set initial/default values
        self.wakeTracked = False
        self.shearRemoval = None
        self.Navg = None  # for removing shear
        self.plotInitialized = False

        self.prefix = kwargs.get('prefix','.')
        if not os.path.isdir(self.prefix):
            if self.verbose: print 'Creating output dir:', self.prefix
            os.makedirs(self.prefix)


        # check and store sampling mesh
        if type(args[0]) in (list,tuple):
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

        yvec = (self.x[-1, 0] - self.x[ 0,0],
                self.y[-1, 0] - self.y[ 0,0],
                self.z[-1, 0] - self.z[ 0,0])
        zvec = (self.x[-1,-1] - self.x[-1,0],
                self.y[-1,-1] - self.y[-1,0],
                self.z[-1,-1] - self.z[-1,0])
        norm = np.cross(yvec,zvec)
        self.norm = norm / np.sqrt(norm.dot(norm))
        if self.verbose:
            print 'Sampling plane normal vector:',self.norm
            if not self.norm[2] == 0:
                print 'WARNING: sampling plane is tilted?'

        # setup planar coordinates
        # d: downwind (x)
        # h: horizontal (y)
        # v: vertical (z)
        #self.xh = y  # rotor axis aligned with Cartesian axes
        self.xv = self.z
        ang = np.arctan2(self.norm[1],self.norm[0])  # ang>0: rotating from x-dir to y-dir
        if self.verbose:
            print '  rotated to rotor-aligned axes (about z):',ang*180./np.pi,'deg'
        self.yaw = ang
        # clockwise rotation (seen from above)
        self.xd =  np.cos(ang)*self.x + np.sin(ang)*self.y  # downstream coordinate--not used for tracking
        self.xh = -np.sin(ang)*self.x + np.cos(ang)*self.y

        self.xh_range = self.xh[:,0]
        self.xv_range = self.xv[0,:]

        # set dummy values in case wake tracking algorithm breaks down
        self.xh_fail = self.xh_range[0]
        self.xv_fail = self.xv_range[0]

        # check plane yaw
        # note: rotated plane should be in y-z with min(x)==max(x)
        #       and tracking should be performed using the xh-xv coordinates
        xd_diff = np.max(self.xd) - np.min(self.xd)
        if self.verbose:
            print '  rotation error:',xd_diff
        if np.abs(xd_diff) > 1e-6:
            print 'WARNING: problem with rotation to rotor-aligned frame?'
        
        # check and calculate instantaneous velocities including shear,
        # u_tot
        assert(len(udata.shape) in (3,4)) 
        assert((self.Nh,self.Nv) == udata.shape[1:3])
        self.Ntimes = udata.shape[0]
        if len(udata.shape)==3:
            self.datasize = 1
            self.u_tot = udata  # shape=(Ntimes,Nh,Nv)
        else:
            self.datasize = udata.shape[3]
            assert(self.datasize == 3)

            # calculate horizontal velocity
#            assert(self.datasize <= 3) 
#            self.u_tot = np.sqrt(u[:,:,:,0]**2 + u[:,:,:,1]**2)

            # calculate velocity normal to sampling plane
            self.u_tot = np.zeros((self.Ntimes,self.Nh,self.Nv))
            for itime in range(self.Ntimes):
                for ih in range(self.Nh):
                    for iv in range(self.Nv):
                        self.u_tot[itime,ih,iv] = udata[itime,ih,iv,:].dot(self.norm)

        self.u = self.u_tot  # in case input u already has shear removed

        self.xwake = np.zeros(self.Ntimes)
        self.ywake = np.zeros(self.Ntimes)
        self.zwake = np.zeros(self.Ntimes)
        self.xh_wake = np.zeros(self.Ntimes)
        self.xv_wake = np.zeros(self.Ntimes)

        if self.verbose:
            print 'Number of time frames to process:',self.Ntimes
            print '\n...finished initializing waketracker'


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

    def removeShear(self,method='default',Navg=-300,windProfile=None,**kwargs):
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
                print 'Estimating velocity profile from fringes of sampling plane', \
                      'with Navg=',Navg
            if Navg < 0:
                self.uavg = np.mean(self.u_tot[-Navg:,:,:], axis=0)  # shape=(Nh,Nv)
                self.Uprofile = (self.uavg[0,:] + self.uavg[-1,:]) / 2  # shape=(Nv)
            else:
                if Navg == 0:
                    Navg = 1  # no averaging performed
                self.uavg = uniform_filter1d(self.u_tot, size=self.Navg, axis=0, mode='mirror')  # see http://stackoverflow.com/questions/22669252/how-exactly-does-the-reflect-mode-for-scipys-ndimage-filters-work
                self.Uprofile = (self.uavg[:,0,:] + self.uavg[:,-1,:]) / 2 # shape=(Ntimes,Nv)

        elif method == 'specified':
            if windProfile is None:
                print 'Need to specify windProfile, shear not removed.'
                return
            if isinstance(windProfile, (list,tuple,np.ndarray)):
                assert(len(windProfile) == self.Nv)
                self.Uprofile = windProfile
            elif isinstance(windProfile, str):
                #zref,Uref = readRef(windProfile)
                #self.Uprofile = np.interp(self.z,zref,Uref) 
                self.Uprofile = np.loadtxt(windProfile)
                print 'Wind profile read from',windProfile

        elif method is not None:
            print 'Shear removal method (',method,') not supported.'
            return

        # actually remove shear now
        self.u = np.copy(self.u_tot)  # u_tot.shape==(Ntimes,Nh,Nv)

        if len(self.Uprofile.shape)==1:
            if self.verbose:
                print '  subtracting out constant profile'
            # Uprofile.shape==(Nv)
            for k,umean in enumerate(self.Uprofile):
                self.u[:,:,k] -= umean
        else:
            if self.verbose:
                print '  subtracting out time-varying profile'
            # Uprofile.shape==(Ntimes,Nv)
            for itime in range(self.Ntimes):
                for k,umean in enumerate(self.Uprofile[itime,:]):
                    self.u[itime,:,k] -= umean

    def findCenters(self,
                    writeTrajectories=None,
                    frame='rotor-aligned'):
        print self.__class__.__name,'needs to override this function!'
        #self.wakeTracked = True

    def fixTrajectoryErrors(self,update=False,istart=0,iend=None):
        """Some wake detection algorithms are not guaranteed to provide
        a valid trajectory. By default, the coordinates of failed
        detection points is set to be (min(y),min(z)). This routine
        locates points believed to be problem points and interpolates
        between surrounding points. Piecewise linear interpolation
        (np.interp) is used.
        
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

        print 'Interpolated wake centers', \
                len(tfix),'times (method: {:s})'.format(self.wakeTracking)
        if update:
            self.ywake = yw_fix
            self.zwake = zw_fix
        
        return yw_fix, zw_fix


    def _writeData(self,fname,data):
        """Helper function to write out specified data (e.g., trajectory
        and optimization parameters)
        """
        Ndata = len(data)
        fmtlist = ['%d'] + Ndata*['%.18e']
        data = np.vstack((np.arange(self.Ntimes),data)).T
        np.savetxt(fname, data, fmt=fmtlist)

    def _readTrajectory(self,fname):
        """Helper function to read trajectory history"""
        data = np.loadtxt(fname)
        assert(len(data) == self.Ntimes)
        # data[:,0] is just an index
        self.xh_wake = data[:,1]
        self.xv_wake = data[:,2]
        return data

    def _updateInertial(self):
        """Called after loading/calculating a wake trajectory in the
        rotor-aligned frame
        """
        ang = self.yaw
        xd = np.mean(self.xd)  # assuming no sampling rotation error, self.xd should be constant
        self.xwake = np.cos(ang)*xd - np.sin(ang)*self.xh_wake
        self.ywake = np.sin(ang)*xd + np.cos(ang)*self.xh_wake
        self.zwake = self.xv_wake

    def _trajectoryIn(self,frame):
        if frame == 'inertial':
            return self.xwake, self.ywake, self.zwake
        elif frame == 'rotor-aligned':
            return self.xh_wake, self.xv_wake
        else:
            print 'output frame not recognized'

    def _initPlot(self):
        """Set up figure properties here""" 
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


    def plotContour(self,
                    itime,
                    cmin=-5.0,cmax=5.0,
                    cmap='jet',
                    markercolor='w',
                    writepng=False,outdir='.',seriesname='U',
                    dpi=100):
        """Plot/update contour and center marker at time ${itime}.
        
        Parameters
        ----------
        itime : integer
            Index of the wake snapshot to plot.
        cmin,cmax : float, optional
            Range of contour values to plot.
        cmap : string, optional
            Colormap for the contour plot.
        markercolor : any matplotlib color, optional
            To plot the detected wake center, otherwise set to None.
        writepng : boolean, optional
            If True, save image to
            ${outdir}/${seriesname}_<timeName>.png
        outdir : string, optional
            Output subdirectory.
        seriesname : string, optional
            Prefix for image series (if writepng==True).
        """
        if writepng and not os.path.isdir(outdir):
            outdir = os.path.join(self.prefix,outdir)
            if self.verbose: print 'Creating output subdirectory:', outdir
            os.makedirs(outdir)

        if not self.plotInitialized:
            self._initPlot()  # first time

            self.plot_clevels = np.linspace(cmin, cmax, 100)
            self.plotobj_filledContours = self.ax.contourf(self.xh, self.xv, self.u[itime,:,:],
                                                           self.plot_clevels, cmap=cmap,
                                                           extend='both')

            # add marker for detected wake center
            if self.wakeTracked and markercolor is not None:
                self.plotobj_ctr, = self.ax.plot(self.ywake[itime], self.zwake[itime], '+',
                                                 color=markercolor, alpha=0.5,
                                                 markersize=10,
                                                 markeredgewidth=1.)
                self.plotobj_crc, = self.ax.plot(self.ywake[itime], self.zwake[itime], 'o',
                                                 color=markercolor, alpha=0.5,
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
            # update plot
            for i in range( len(self.plotobj_filledContours.collections) ):
                self.plotobj_filledContours.collections[i].remove()
            self.plotobj_filledContours = self.ax.contourf(
                    self.xh, self.xv, self.u[itime,:,:],
                    self.plot_clevels, cmap=cmap, extend='both')

            if self.wakeTracked and markercolor is not None:
                self.plotobj_ctr.set_data(self.ywake[itime], self.zwake[itime])
                self.plotobj_crc.set_data(self.ywake[itime], self.zwake[itime])

        if writepng:
            fname = os.path.join(
                    outdir,
                    '{:s}_{:05d}.png'.format(seriesname,itime)
                    )
            self.fig.savefig(fname, dpi=dpi)
            print 'Saved',fname


    def saveSnapshots(self,**kwargs):
        """Write out all snapshots to ${outdir}.

        See plotContour for keyword arguments.
        """ 
        if not self.wakeTracked:
            print 'Note: wake tracking has not been performed; wake centers will not be plotted.'
        for itime in range(self.Ntimes):
            self.plotContour(itime,writepng='True',**kwargs)


class contourwaketracker(waketracker):
    """Class for wake tracking based on (velocity) contours
    
    Inherits superclass WakeTracker
    """

    def __init__(self,*args,**kwargs):
        super(contourwaketracker,self).__init__(*args,**kwargs)

        self.Clevels = np.zeros(self.Ntimes)
        self.Cfvals = np.zeros(self.Ntimes)
        self.paths = self.Ntimes*[None]

        if self.verbose:
            print '\n...finished initializing contourwaketracker'

    def _findContourCenter(self,
                           itime,
                           C,
                           Crange,
                           targetValue,
                           weightedCenter,
                           tol=0.01,
                           fn=None):
        """Helper function that returns the coordinates of the detected
        wake center. Iteration continues in a binary search fashion
        until the difference in contour values is < 'tol'
        """
        interval = Crange[1]-Crange[0]

        if fn is None:
            # Note: This is MUCH faster, since we don't have to search
            #       for interior pts!
            def Cfn(path): return contour.calcArea(path)
        else:
            def Cfn(path):
                return contour.integrateFunction(
                        path, self.xh, self.xv,
                        self.u_tot[itime,:,:], self.u[itime,:,:],
                        fn)

        converged = False
        NtraceCalls = 0
        NfnEvals = 0
        Nref = 0

        Clist,paths,level = [],[],[]
        success = True
        while Nref==0 or interval > tol:  # go through search at least once
            Nref += 1

            # BEGIN search loop
            #vvvvvvvvvvvvvvvvvvvvvvvvvvvv
            for it,thresh in enumerate(Crange):
                for path in C.trace(thresh):
                    NtraceCalls += 1
                    # Returns a list of arrays (floats) and lists (uint8), of the contour
                    # coordinates and segment descriptors, respectively
                    if path.dtype=='uint8': break  # don't need the segment info
                    if np.all(path[-1] == path[0]):  # found a closed path
                        if fn is None:
                            # area contours
                            Clist.append(Cfn(path))
                            level.append(thresh)
                            paths.append(path)
                        else:
                            # flux contours
                            fval, avgDeficit, corr = Cfn(path)
                            if fval is not None and avgDeficit < 0:
                                Clist.append(fval)
                                level.append(thresh)
                                paths.append(path)
                        NfnEvals += 1

            if len(Clist) > 0:
                Cerr = np.abs( np.array(Clist) - targetValue )
                idx = np.argmin(Cerr)
                curOptLevel = level[idx]
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
            Crange = [curOptLevel-interval,curOptLevel+interval]

        # end of refinement loop
        info = {
                'tolerance': tol,
                'Nrefine': Nref,
                'NtraceCalls': NtraceCalls,
                'NfnEvals': NfnEvals,
                'success': success
        }

        if success:
            self.paths[itime] = paths[idx]  # save paths for plotting
            self.Clevels[itime] = level[idx]  # save time-varying contour levels as reference data
            self.Cfvals[itime] = Clist[idx]

            if weightedCenter:
                yc,zc = contour.calcWeightedCenter(paths[idx],
                                                   self.xh,
                                                   self.xv,
                                                   self.u[itime,:,:])
            else:
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

    def plotContour(self,itime,outline=True,**kwargs):
        """Plot/update contour and center marker at time ${itime}.
        
        Overridden waketracker.plotContour function to include the calculated 
        wake contour outline.

        Parameters
        ----------
        itime : integer
            Index of the wake snapshot to plot.
        cmin,cmax : float, optional
            Range of contour values to plot.
        cmap : string, optional
            Colormap for the contour plot.
        markercolor : any matplotlib color, optional
            To plot the detected wake center, otherwise set to None.
        writepng : boolean, optional
            If True, save image to
            ${outdir}/${seriesname}_<timeName>.png
        outdir : string, optional
            Output subdirectory.
        seriesname : string, optional
            Prefix for image series (if writepng==True).

        Additional Parameters
        ---------------------
        outline : boolean, optional
            Plot the wake contour outline (a path object). If False,
            operates the same as waketracker.plotContour.
        """
        writepng = kwargs.get('writepng',False)
        outdir = os.path.join(self.prefix,kwargs.get('outdir','.'))
        seriesname = kwargs.get('seriesname','U')
        dpi = kwargs.get('dpi',100)

        if writepng and not os.path.isdir(outdir):
            if self.verbose: print 'Creating output subdirectory:', outdir
            os.makedirs(outdir)

        plotOutline = outline and self.wakeTracked

        if self.plotInitialized and plotOutline:
            self.plotobj_wakeOutline.remove()

        kwargs['writepng'] = False
        super(contourwaketracker,self).plotContour(itime,**kwargs)

        if plotOutline:
            path = mpath.Path(self.paths[itime])
            self.plotobj_wakeOutline = mpatch.PathPatch(path,facecolor='none',
                                                        edgecolor='w',ls='-')
            self.ax.add_patch(self.plotobj_wakeOutline)
#            if hasattr(self,'plotobj_linecontours'):
#                for i in range(len(self.plotobj_linecontours.collections)):
#                    self.plotobj_linecontours.collections[i].remove()
#            self.plotobj_linecontours = self.ax.contour(
#                    self.xh, self.xv, self.u[itime,:,:], [self.Clevels[itime]],
#                    colors='w', linestyles='-', linewidths=2)

        if writepng:
            fname = os.path.join(
                    outdir,
                    '{:s}_{:05d}.png'.format(seriesname,itime)
                    )
            self.fig.savefig(fname, dpi=dpi)
            print 'Saved',fname

