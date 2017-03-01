import os
import sys
import importlib

import numpy as np
from matplotlib._cntr import Cntr
import pickle

from waketrackers import contourwaketracker

class ConstantArea(contourwaketracker):
    """ Identifies a wake as a region with velocity deficit contour
    enclosing an area closest to the rotor (or another specified
    reference area).

    The wake center is identified as the velocity-deficit-weighted 
    "center of mass" of all points within the enclosed region. 

    This is the fastest of the contour-based tracking methods.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__

    def findCenters(self,refArea,
                    writeTrajectories=None,writeOutlines=None,
                    weightedCenter=True,frame='rotor-aligned',
                    Ntest=51,tol=0.01):
        """Uses a binary search algorithm (findContourCenter) to
        locate the contour with flux closest to the targetValue.
        
        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        refArea : float
            Area to attempt to match, e.g., the rotor disk area.
        writeTrajectories : string
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        writeOutlines : string
            Name of pickle archive file (\*.pkl) to attempt inputting
            and to write out detected contour outlines; set to None to
            skip I/O.
        weightedCenter : boolean
            If True, calculate the velocity-deficit-weighted "center of
            mass"; if False, calculate the geometric center of the wake.
        frame : string
            Reference frame, either 'inertial' or 'rotor-aligned'.
        Ntest : integer
            The number of initial test contours to calculate.
        tol : float
            Minimum spacing to test during the binary search.

        Returns
        -------
        x_wake,y_wake,z_wake : ndarray
            Wake trajectory if frame is 'inertial'
        xh_wake,xv_wake : ndarray
            Wake trajectory if frame is 'rotor-aligned'
        """
        # try to read trajectories
        if writeTrajectories is not None:
            trajectoryfile = os.path.join(self.prefix,writeTrajectories)
            try:
                data = self._readTrajectory(trajectoryfile)
                self.Clevels = data[:,3]
                self.Cfvals = data[:,4]
                if self.verbose:
                    print 'Trajectory loaded from',trajectoryfile
                self._updateInertial()
                self.wakeTracked = True

            except AssertionError:
                print 'Incorrect number of time steps in',trajectoryfile

            except IOError:
                print 'Failed to read',trajectoryfile

        # try to read wake outlines (optional)
        if writeOutlines is not None:
            pklname = os.path.join(self.prefix,writeOutlines)
            if not pklname.endswith('.pkl'):
                pklname += '.pkl'
            self.paths = pickle.load(open(pklname,'r'))
            if self.verbose:
                print 'Read pickled paths from',pklname
            
        if self.wakeTracked:
            return self._trajectoryIn(frame)

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            contourData = Cntr(self.xh, self.xv, self.u[itime,:,:])
            Crange = np.linspace(np.min(self.u[itime,:,:]), 0, Ntest+1)[1:]

            yc,zc,info = self._findContourCenter(itime,
                                                 contourData,
                                                 Crange,
                                                 refArea,
                                                 weightedCenter=weightedCenter,
                                                 tol=tol,
                                                 fn=None)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                sys.stderr.flush()
        if self.verbose: print ''

        self._updateInertial()

        # write out trajectories
        if writeTrajectories is not None:
            self._writeData(trajectoryfile,
                    (self.xh_wake, self.xv_wake, self.Clevels, self.Cfvals))
            if self.verbose:
                print 'Wrote out trajectory to',trajectoryfile

        # write out wake outlines
        if writeOutlines is not None:
            pklname = os.path.join(self.prefix,writeOutlines)
            if not pklname.endswith('.pkl'):
                pklname += '.pkl'
            pickle.dump(self.paths,open(pklname,'w'))
            if self.verbose:
                print 'Wrote out pickled paths to',pklname

        self.wakeTracked = True
    
        return self._trajectoryIn(frame)

