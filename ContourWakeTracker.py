import os
import sys
import importlib

import numpy as np

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
                    trajectoryFile=None,outlinesFile=None,
                    weightedCenter=True,frame='rotor-aligned',
                    Ntest=51,tol=0.01):
        """Uses a binary search algorithm (findContourCenter) to
        locate the contour with flux closest to the targetValue.
        
        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        refArea : float
            Area to attempt to match, e.g., the rotor disk area.
        trajectoryFile : string, optional
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlinesFile : string, optional
            Name of pickle archive file (\*.pkl) to attempt input and to
            write out detected contour outlines; set to None to skip
            I/O.
        weightedCenter : boolean, optional
            If True, calculate the velocity-deficit-weighted "center of
            mass"; if False, calculate the geometric center of the wake.
        frame : string, optional
            Reference frame, either 'inertial' or 'rotor-aligned'.
        Ntest : integer, optional
            The number of initial test contours to calculate.
        tol : float, optional
            Minimum spacing to test during the binary search.

        Returns
        -------
        x_wake,y_wake,z_wake : ndarray
            Wake trajectory if frame is 'inertial'
        xh_wake,xv_wake : ndarray
            Wake trajectory if frame is 'rotor-aligned'
        """
        # try to read trajectories
        if trajectoryFile is not None:
            try:
                data = self._readTrajectory(trajectoryFile)
                if not data is None:
                    if self.verbose:
                        print 'Trajectory loaded from',trajectoryFile
                    self._updateInertial()
                    self.wakeTracked = True
            except AssertionError:
                print 'Incorrect number of time steps in',trajectoryFile
            except IOError:
                print 'Failed to read',trajectoryFile

        # try to read wake outlines (optional)
        if self.wakeTracked and outlinesFile is not None:
            try:
                self._readOutlines(outlinesFile)
                if self.verbose:
                    print 'Read pickled outlines from',outlinesFile
            except IOError:
                print 'Failed to read',outlinesFile

        # done if read was successful
        if self.wakeTracked:
            return self.trajectoryIn(frame)

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            yc,zc,info = self._findContourCenter(itime,
                                                 refArea,
                                                 weightedCenter=weightedCenter,
                                                 Ntest=Ntest,
                                                 tol=tol,
                                                 fn=None)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')
        self._updateInertial()

        # write out trajectories
        if trajectoryFile is not None:
            self._writeTrajectory(trajectoryFile)
            if self.verbose:
                print 'Wrote out trajectory to',trajectoryFile

        # write out wake outlines
        if outlinesFile is not None:
            self._writeOutlines(outlinesFile)
            if self.verbose:
                print 'Wrote out pickled outlines to',outlinesFile
    
        return self.trajectoryIn(frame)

