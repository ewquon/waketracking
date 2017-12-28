import sys

import numpy as np

from samwich.waketrackers import contourwaketracker

class ConstantArea(contourwaketracker):
    """Identifies a wake as a region with velocity deficit contour
    enclosing an area closest to the rotor (or another specified
    reference area).

    This is the fastest of the contour-based tracking methods since
    it does not necessarily depend on the 'contain_pts' function (unless
    checkDeficit is set to True).

    Inherits class contourwaketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__,'\n'

    def findCenters(self,refArea,
                    trajectoryFile=None,outlinesFile=None,
                    weightedCenter=True,
                    contour_closure=None,
                    frame='rotor-aligned',
                    Ntest=21,tol=0.01,
                    checkdeficit=False,
                    debug=False):
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
        weightedCenter : boolean or function, optional
            If True, calculate the velocity-deficit-weighted "center of
            mass"; if False, calculate the geometric center of the wake.
            This can also be a weighting function.
        contour_closure : string, optional
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

        if contour_closure is None or contour_closure=='none':
            closure = False
        elif contour_closure=='simple':
            closure = True
        elif contour_closure=='compound':
            closure = (self.xh_range,self.xv_range)

        # calculate trajectories for each time step
        if self.verbose:
            print 'Attempting to match area:',refArea,'m^2'
        for itime in range(self.Ntimes):
            _,_,info = self._findContourCenter(itime,
                                               refArea,
                                               weightedCenter=weightedCenter,
                                               contour_closure=closure,
                                               Ntest=Ntest,
                                               tol=tol,
                                               func=None,
                                               vdcheck=checkdeficit,
                                               debug=debug)
            if not info['success']:
                print 'WARNING: findContourCenter was unsuccessful.'
                print info

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

