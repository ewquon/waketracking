import os
import sys
import importlib

import numpy as np

from waketrackers import contourwaketracker

class ConstantArea(contourwaketracker):
    """Identifies a wake as a region with velocity deficit contour
    enclosing an area closest to the rotor (or another specified
    reference area).

    This is the fastest of the contour-based tracking methods.

    Inherits class contourwaketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__,'\n'

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
        self.plotInitialized = False

        # try to read trajectories (required) and outlines (optional)
        self._readTrajectory(trajectoryFile)
        self._readOutlines(outlinesFile)

        # done if read was successful
        if self.wakeTracked:
            return self.trajectoryIn(frame)

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            _,_,info = self._findContourCenter(itime,
                                               refArea,
                                               weightedCenter=weightedCenter,
                                               Ntest=Ntest,
                                               tol=tol,
                                               func=None)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._updateInertial()

        # write out everything
        self._writeTrajectory(trajectoryFile)
        self._writeOutlines(outlinesFile)
    
        return self.trajectoryIn(frame)


class ConstantFlux(contourwaketracker):
    """Identifies a wake as a region outlined by a velocity deficit
    contour over which the integrated flux matches a specified target
    value, given a flux function.

    Inherits class contourwaketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__,'\n'

    def findCenters(self,refFlux,
                    fluxFunction,fluxField='u_tot',
                    trajectoryFile=None,outlinesFile=None,
                    weightedCenter=True,frame='rotor-aligned',
                    Ntest=51,tol=0.01):
        """Uses a binary search algorithm (findContourCenter) to
        locate the contour with flux closest to the targetValue.
        
        Some candidate functions:

        * mass flow, func = lambda u: u
        * momentum flux, func = lambda u: u**2

        The contour area can be referenced as func = lambda u,A: ...

        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        refFlux : float
            Flux to attempt to match, e.g., the massflow rate.
        fluxFunction : function
            A specified function of two variables, the velocity deficit
            and the instantaneous velocity (with shear removed).
        fluxField : string, optional
            Name of the field to use as input to the fluxFunction; use
            the instantaneous velocity, 'u_tot', by default.
        trajectoryFile : string
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlinesFile : string
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
        self.plotInitialized = False

        # try to read trajectories (required) and outlines (optional)
        self._readTrajectory(trajectoryFile)
        self._readOutlines(outlinesFile)

        # done if read was successful
        if self.wakeTracked:
            return self.trajectoryIn(frame)

        try:
            testField = getattr(self,fluxField)
        except AttributeError:
            print 'Warning: flux field',fluxField,'not available,', \
                    'using \'u_tot\' by default'
            fluxField = 'u_tot'

        # some sanity checks if needed
        if self.verbose:
            Umean = np.mean(testField[-1,self.jmin:self.jmax,self.kmin:self.kmax])
            print 'Sample function evaluation:',fluxFunction(Umean)
            print ' ~= targetValue / area =',refFlux,'/ A'

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            _,_,info = self._findContourCenter(itime,
                                               refFlux,
                                               weightedCenter=weightedCenter,
                                               Ntest=Ntest,
                                               tol=tol,
                                               func=fluxFunction,
                                               field=fluxField)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._updateInertial()

        # write out everything
        self._writeTrajectory(trajectoryFile)
        self._writeOutlines(outlinesFile)
    
        return self.trajectoryIn(frame)

