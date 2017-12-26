import sys

import numpy as np

from samwich.waketrackers import contourwaketracker

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
                    weightedCenter=True,
                    contourClosure=None,
                    frame='rotor-aligned',
                    Ntest=21,tol=0.01,
                    checkdeficit=True,
                    debug=False):
        """Uses a binary search algorithm (findContourCenter) to
        locate the contour with flux closest to the targetValue.
        
        Some candidate functions to be integrated over the contour
        surface:

        * mass flow, func = lambda u: u
        * momentum flux, func = lambda u: u**2

        The contour area can be referenced as func = lambda u,A: ...

        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        refFlux : float
            Flux to attempt to match, e.g., the massflow rate.
        fluxFunction : function
            A specified function of one (or two) variables, the velocity
            deficit (and the contour area).
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
        weightedCenter : boolean or function, optional
            If True, calculate the velocity-deficit-weighted "center of
            mass"; if False, calculate the geometric center of the wake.
            This can also be a weighting function.
        contourClosure : string, optional
            If 'simple', then open paths with endpoints on the same
            edge will be closed by connecting the endpoints; if
            'compound', then open paths will be closed by adding
            segments along the domain boundaries to form closed
            contours.
        frame : string, optional
            Reference frame, either 'inertial' or 'rotor-aligned'.
        Ntest : integer, optional
            The number of initial test contours to calculate.
        tol : float, optional
            Minimum spacing to test during the binary search.
        checkdeficit : boolean, optional
            If True, only consider wake candidates in which the average
            velocity deficit is less than 0.
        debug : boolean, optional
            Print out debugging information about the contour search
            routine.

        Returns
        -------
        x_wake,y_wake,z_wake : ndarray
            Wake trajectory if frame is 'inertial'
        xh_wake,xv_wake : ndarray
            Wake trajectory if frame is 'rotor-aligned'
        """
        self.clearPlot()

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
            testField = getattr(self,fluxField)

        # some sanity checks if needed
        if self.verbose:
            Utest = np.min(testField[-1,self.jmin:self.jmax,self.kmin:self.kmax])
            if fluxFunction.func_code.co_argcount==1: # fn(u)
                print 'Sample function evaluation: f(u={:g}) = {:g}'.format(
                        Utest,fluxFunction(Utest))
            else: # fn(u,A)
                print 'Sample function evaluation: f(u={:g},1.0) = {:g}'.format(
                        Utest,fluxFunction(Utest,1))
            print ' ~= targetValue / area =',refFlux,'/ A'

        if contourClosure is None or contourClosure=='none':
            closure = False
        elif contourClosure=='simple':
            closure = True
        elif contourClosure=='compound':
            closure = (self.xh_range,self.xv_range)

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            _,_,info = self._findContourCenter(itime,
                                               refFlux,
                                               weightedCenter=weightedCenter,
                                               contourClosure=closure,
                                               Ntest=Ntest,
                                               tol=tol,
                                               func=fluxFunction,
                                               field=fluxField,
                                               vdcheck=checkdeficit,
                                               debug=debug)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._updateInertial()

        self.wakeTracked = True

        # write out everything
        self._writeTrajectory(trajectoryFile, self.Clevels, self.Cfvals)
        self._writeOutlines(outlinesFile)
    
        return self.trajectoryIn(frame)

