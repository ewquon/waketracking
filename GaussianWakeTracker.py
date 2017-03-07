import os
import sys
import importlib

import numpy as np
from scipy.optimize import least_squares

from waketrackers import waketracker

class Gaussian(waketracker):
    """Identifies a wake as the best fit to a univariate Gaussian
    distribution described by:
    :math:`\exp(-0.5 (\Delta_y^2/\sigma_y^2 + \Delta_z^2/\sigma_z^2))`

    Inherits class waketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__,'\n'

    def findCenters(self,
                    umin=None,sigma=50.,
                    trajectoryFile=None,
                    frame='rotor-aligned'):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit.
        
        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        umin : float or ndarray
            Max velocity deficit (i.e., should be < 0). If None, then
            this is detected from the field.
        sigma : float
            Wake width parameter, equivalent to the standard deviation
            of the Gaussian function. This may be a constant or a
            function of downstream distance.
        trajectoryFile : string, optional
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        frame : string, optional
            Reference frame, either 'inertial' or 'rotor-aligned'.

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

        # done if read was successful
        if self.wakeTracked:
            return self.trajectoryIn(frame)

        # setup Gaussian parameters
        if umin is None:
            if self.shearRemoval is None:
                print 'Note: removeShear has not been called'
            if hasattr(self,'uavg'):
                if len(self.uavg.shape)==2: # (Nh,Nv)
                    umin = np.min(self.uavg) * np.ones(self.Ntimes)
                else:
                    assert(len(self.uavg.shape)==3) # (Ntimes,Nh,Nv)
                    umin = np.min(self.uavg,axis=(1,2))
            else:
                print 'Note: using instantaneous values'
                umin = np.min(self.u,axis=(1,2))
            if not np.all(umin < 0):
                print 'Warning: unexpected positive velocity deficit'
        elif not isinstance(umin,np.ndarray):
            umin = umin * np.ones(self.Ntimes)
        self.umin = umin
        if self.verbose:
            print 'average Gaussian function amplitude =',np.mean(self.umin),'m/s'

        try:
            self.sigma = float(sigma)
            if self.verbose:
                print 'Gaussian width =',self.sigma,'m'
        except TypeError:
            xd = np.mean(self.xd)
            self.sigma = sigma(xd)
            if self.verbose:
                print 'Calculated sigma =',self.sigma,'m at x=',xd,'m'

        # calculate trajectories for each time step
        y1 = self.xh.ravel()
        z1 = self.xv.ravel()
        guess = [0,0] # since we're in the rotor-aligned frame, centered at the origin, already
        minmax = ([self.xh_min,self.xv_min],
                  [self.xh_max,self.xv_max])
        for itime in range(self.Ntimes):
            u1 = self.u[itime,:,:].ravel()
            def func(x):
                """objective function for x=[yc,zc]"""
                delta_y = x[0] - y1
                delta_z = x[1] - z1
                return self.umin[itime] * \
                        np.exp( -0.5 * (delta_y**2 + delta_z**2)/sigma**2 ) - u1

            res = least_squares(func, guess, bounds=minmax)

            if res.success:
                self.xh_wake[itime], self.xv_wake[itime] = res.x[:2]
            else:
                self.xh_wake[itime], self.xv_wake[itime] = self.xh_fail, self.xv_fail

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._updateInertial()

        self.wakeTracked = True

        # write out everything
        self._writeTrajectory(trajectoryFile)
    
        return self.trajectoryIn(frame)


