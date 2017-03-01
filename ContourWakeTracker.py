import os
import sys
import importlib

import numpy as np
from matplotlib._cntr import Cntr
import pickle

from waketrackers import contourwaketracker
import contour

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
                    writeTrajectories=None,writePaths=None,
                    weightedCenter=True,frame='rotor-aligned',
                    Ntest=51,tol=0.01):
        """Uses a binary search algorithm (in findContourCenter) to
        locate the contour with flux closest to the targetValue.
        
        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        refArea : float
            Area to attempt to match, e.g., the rotor disk area.
        writeTrajectories : string
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O.
        writePaths : string
            Name of pickle archive file (\*.pkl) to write out for future
            reloading or None to skip output.
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
        if writeTrajectories is not None:
            trajectoryfile = os.path.join(self.prefix,writeTrajectories)
            # TODO: attempt read of previously calculated contours
            print 'Attempting to read',trajectoryfile
            print 'Loading trajectories not implemented yet!'

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

        ang = self.yaw
        xd = np.mean(self.xd)  # assuming no sampling rotation error, self.xd should be constant
        self.xwake = np.cos(ang)*xd - np.sin(ang)*self.xh_wake
        self.ywake = np.sin(ang)*xd + np.cos(ang)*self.xh_wake
        self.zwake = self.xv_wake

        if writeTrajectories is not None:
            self._writeTrajectory(trajectoryfile,
                    (self.xh_wake, self.xv_wake, self.Clevels, self.Cfvals))
            if self.verbose:
                print 'Wrote out trajectory to',trajectoryfile

        if writePaths is not None:
            pklname = os.path.join(self.prefix,writePaths)
            if not pklname.endswith('.pkl'):
                pklname += '.pkl'
            pickle.dump(self.paths,open(pklname,'w'))
            if self.verbose:
                print 'Wrote out pickled paths to',pklname

        self.wakeTracked = True
    
        if frame == 'inertial':
            return self.xwake, self.ywake, self.zwake
        elif frame == 'rotor-aligned':
            return self.xh_wake, self.xv_wake
        else:
            print 'output frame not recognized'
