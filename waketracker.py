import os

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatch

import contour

class WakeTracker(object):
    """A general class for wake tracking operations.

    Tracking algorithms are expected to be implemented as children of
    this class.
    """

    def __init__(self,dataLoaderObject):
        print 'Processing data from',dataLoaderObject

    def fixTrajectoryErrors(self,update=False,istart=0,iend=None):
        """Some wake detection algorithms are not guaranteed to provide
        a valid trajectory. By default, the coordinates of failed
        detection points is set to be (${y_start},${z_start}). This
        routine locates points believed to be problem points and
        interpolates between surrounding points. Piecewise linear
        interpolation (np.interp) is used.
        
        Parameters
        ----------
        update : boolean, optional
            Set to True to udpate the ywake and zwake in place.
        istart,iend : integer, optional
            The range of values to correct.

        Returns
        -------
        yw_fix,zw_fix : ndarray
            Cleaned-up coordinates
        """
        idx = np.nonzero(
                (self.ywake > self.y_start) & (self.zwake > self.z_start)
                )[0]  # indices of reliable values
        ifix0 = np.nonzero(
                (self.ywake==self.y_start) & (self.zwake==self.z_start)
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

    def _writeTrajectory(self,fname,data):
        """Helper function to write out specified data (e.g., trajectory
        and optimization parameters)
        """
        Ndata = len(data)
        fmtlist = ['%d'] + Ndata*['%.18e']
        data = np.vstack((np.arange(self.Ntimes),data)).T
        np.savetxt( fname, data, fmt=fmtlist )

    def _initPlot(self):
        """Set up figure properties here""" 
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #self.fig = plt.figure( figsize=(10,6), dpi=200 )
        self.fig = plt.figure( figsize=(8,6) )

        def handle_close(event):
            self.plotInitialized = False
        cid = self.fig.canvas.mpl_connect('close_event', handle_close)

        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])

        self.ax.set_xlim(self.y_start, self.y_end)
        self.ax.set_ylim(self.z_start, self.z_end)
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect('equal',adjustable='box',anchor='C')

        self.ax.xaxis.set_tick_params( size=10 )
        self.ax.yaxis.set_tick_params( size=10 )

        self.ax.set_xlabel(r'$y (m)$', fontsize=14)
        self.ax.set_ylabel(r'$z (m)$', fontsize=14)
        #self.ax.set_title('Wake tracking method: {:}'.format(self.wakeTracking))


class ContourWakeTracker(WakeTracker):
    """Class for wake tracking based on (velocity) contours
    
    Inherits superclass WakeTracker
    """

    def __init__(self,dataLoaderObject):
        super(self.__class__,self).__init__(*args,**kwargs)
        print 'Processing data from',dataLoaderObject

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
            def Cfn(path): return calcContourArea(path)
        else:
            def Cfn(path):
                return calcContourFunction(
                        path, self.yg, self.zg,
                        self.utot[itime,:,:], self.u[itime,:,:],
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
                if DEBUG: print dbgprefix,'threshold=',thresh
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
                yc = self.y_fail
                zc = self.z_fail
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
                yc,zc = calcContourWeightedCenter(paths[idx],self.yg,self.zg,self.u[itime,:,:])
            else:
                yc = np.mean( paths[idx][:,0] )
                zc = np.mean( paths[idx][:,1] )

        else:
            # tracking failed!
            self.paths[itime] = []
            self.Clevels[itime] = 0
            self.Cfvals[itime] = 0

            yc = self.y_fail
            zc = self.z_fail

        self.ywake[itime] = yc
        self.zwake[itime] = zc

        return yc,zc,info

    def _readOutlines(self,pklname=None,prefix='trajectory_contours'):
        """Read saved paths from a contour collection for the contour
        level identified as embodying the wake
        """
        if not prefix[-1]=='_': prefix += '_'
        if pklname is None:
            pklname = os.path.join(self.outdir,prefix+str(self.downD)+'D_paths.pkl')
        try:
            import pickle
            self.paths = pickle.load( open(pklname,'r') )
            print 'Read wake outlines from',pklname
        except ImportError: pass
        except IOError:
            print 'Wake outline file',pklname,'was not found'
        except:
            print pklname,'was not read'

    def plotContour(self,
                    itime,
                    cmin=-5.0,cmax=5.0,
                    cmap='jet',
                    markercolor='w',
                    plotpath=True,
                    writepng=False,name='',
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
        plotpath : boolean, optional
            Whether to plot the identified contour path (as an outline).
        writepng : boolean, optional
            If True, save image to
            ${outdir}/<timeName>_${name}${downD}D_U.png
        """ 
        if not self.plotInitialized:
            self._initPlot()  # first time

            self.plot_clevels = np.linspace(cmin, cmax, 100)
            self.plotobj_cf = self.ax.contourf(self.yg, self.zg, self.u[itime,:,:],
                                               self.plot_clevels, cmap=cmap, extend='both')

            # add marker for detected wake center
            if markercolor is not None:
                self.plotobj_ctr, = self.ax.plot(self.ywake[itime], self.zwake[itime], '+',
                                                 color=markercolor, alpha=0.5,
                                                 markersize=10,
                                                 markeredgewidth=1.)
                self.plotobj_crc, = self.ax.plot(self.ywake[itime], self.zwake[itime], 'o',
                                                 color=markercolor, alpha=0.5,
                                                 markersize=10,
                                                 markeredgewidth=1.,
                                                 markeredgecolor=markercolor )

            # add time annotation
            #self.plotobj_txt = self.ax.text(0.98, 0.97,'t={:.1f}'.format(self.t[itime]),
            #        horizontalalignment='right', verticalalignment='center',
            #        transform=self.ax.transAxes)

            # add colorbar
            cb_ticks = np.linspace(cmin, cmax, 11)
            cb = self.fig.colorbar( self.plotobj_cf, ticks=cb_ticks, label=r'$U \ (m/s)$' )

            self.plotInitialized = True

        else:
            # update plot
            for i in range( len(self.plotobj_cf.collections) ):
                self.plotobj_cf.collections[i].remove()
            self.plotobj_cf = self.ax.contourf(
                    self.yg, self.zg, self.u[itime,:,:],
                    self.plot_clevels, cmap=cmap, extend='both')

            if not markercolor=='':
                self.plotobj_ctr.set_data(self.ywake[itime], self.zwake[itime])
                self.plotobj_crc.set_data(self.ywake[itime], self.zwake[itime])
            try:
                for i in range(len(self.plotobj_cnt.collections)):
                    self.plotobj_cnt.collections[i].remove()
                self.plotobj_cnt = self.ax.contour(
                        self.yg, self.zg, self.u[itime,:,:], [self.Clevels[itime]],
                        colors='w', linestyles='-', linewidths=2)
            except: pass

            if plotpath:
                try:
                    self.plotobj_pth.remove()
                except: pass

        if plotpath:
            # add contour if available
            try:
                path = mpath.Path(self.paths[itime])
                self.plotobj_pth = mpatch.PathPatch(path,edgecolor='w',facecolor='none',ls='-')
                self.ax.add_patch(self.plotobj_pth)
            except: pass

        if writepng:
            if name: name += '_'
            fname = os.path.join(
                    self.outdir,
                    '{:g}_{:s}{:d}D_U.png'.format(float(self.dirs[itime]),name,self.downD)
                    )
            self.fig.savefig(fname, dpi=dpi)
            print 'Saved',fname

    def printSnapshots(self,**kwargs):
        """Write out all snapshots to $outdir.

        See plotContour for keyword arguments.
        """ 
        kwargs['writepng'] = True
        if self.wakeTracking:
            print 'Outputting snapshots (wake tracking method: {:})'.format(self.wakeTracking)
            for itime in range(self.Ntimes):
                self.plotContour(itime,**kwargs)
        else:
            print 'Wake centers have not been calculated!'

