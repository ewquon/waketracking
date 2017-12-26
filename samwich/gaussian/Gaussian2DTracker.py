import sys
import importlib

import numpy as np
from scipy.optimize import least_squares

from samwich.waketrackers import waketracker

class Gaussian2D(waketracker):
    """Identifies a wake as the best fit to a bivariate (elliptical)
    Gaussian distribution described by:

    .. math::
        f(y,z) = A \exp \left[
            -\\frac{1}{2(1-\\rho^2)} \left(
                \\frac{(\\bar{y}-y_c)^2}{\sigma_y^2}
                - \\frac{2\\rho (\\bar{y}-y_c) (\\bar{z}-z_c)}{\sigma_y\sigma_z}
                + \\frac{(\\bar{z}-z_c)^2}{\sigma_z^2}
            \\right)
        \\right]

    where

    .. math::
        \\begin{align}
        \\bar{y} &= y\cos\\theta - z\sin\\theta \\\\
        \\bar{z} &= y\sin\\theta + z\cos\\theta
        \end{align}

    provides a rotation about +x.

    Least-squares optimization is performed to simultaneously identify
    the wake center along with the additional Gaussian parameters.

    Inherits class waketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print '\n...finished initializing',self.__class__.__name__,'\n'

    def findCenters(self,
                    umin=None,
                    A_min=100.,
                    A_max=np.inf,
                    AR_max=10.,
                    rho=None,
                    res=100,
                    trajectoryFile=None,outlinesFile=None,
                    frame='rotor-aligned'):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit.

        Overrides the parent findCenters routine.
        
        Parameters
        ----------
        umin : float or ndarray
            Max velocity deficit (i.e., should be < 0). If None, then
            this is detected from the field.
        A_min : float, optional
            The minimum approximate wake area, estimated to be the
            pi times the product of the wake widths.
        A_max : float, optional
            The maximum approximate wake area, estimated to be the
            pi times the product of the wake widths. If None, then there
            is no limit on the "optimal" wake size.
        AR_max : float, optional
            The maximum aspect ratio between the wake widths in two
            directions (analogous to the semi-major and -minor axes in
            an ellipse); this dictates the maximum amount of stretching.
        rho : float, optional
            The cross-correlation parameter--CURRENTLY UNTESTED.
        res : integer, optional
            Number of points to represent the wake outline as a circle
        trajectoryFile : string, optional
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlinesFile : string, optional
            Name of pickle archive file (\*.pkl) to attempt input and to
            write out approximate wake outlines; set to None to skip I/O.
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
        self._readOutlines(outlinesFile)

        # done if read was successful
        if self.wakeTracked:
            return self.trajectoryIn(frame)

        # setup Gaussian parameters
        if self.shearRemoval is None:
            print 'Note: removeShear has not been called'
        if umin is None:
            # calculate umin available data
            self.umin = np.min(self.u,axis=(1,2))
        elif isinstance(umin,np.ndarray):
            # specified umin as array with length Ntimes
            self.umin = umin
        else:
            # specified constant umin
            self.umin = umin * np.ones(self.Ntimes)
        self.rotation = np.zeros(self.Ntimes)
        self.sigma_y = np.zeros(self.Ntimes)
        self.sigma_z = np.zeros(self.Ntimes)

        if not np.all(umin < 0):
            print 'Warning: Unexpected positive velocity deficit at', \
                    len(np.nonzero(self.umin > 0)[0]),'of',self.Ntimes,'times'
        if self.verbose:
            print 'average Gaussian function amplitude =',np.mean(self.umin),'m/s'

        if rho is not None:
            print 'Note: cross-correlation parameter is not yet implemented'

        # approximate wake outline as best-fit ellipse at 36.8% of the max wake
        #   deficit (corresponding to f(y,z) = A*exp(-1))
        azi = np.linspace(0,2*np.pi,res)

        # set up optimization parameters
        # note: origin of the rotor-aligned frame is at the center of the sampling
        #   plane already
        # note: sigmay = AR*sigmaz
        #       sigmaz = sqrt(A/(pi*AR))
        guess = [
                (self.xh_min+self.xh_max)/2,    # 0: yc
                (self.xv_min+self.xv_max)/2,    # 1: zc
                0.,                             # 2: theta
                7500.,                          # 3: A == pi * sigma_y * sigma_z
                1.,                             # 4: AR == sigma_y/sigma_z
        ]
        minmax = (
                # y range,    z range,     rotation range, wake size, wake stretching
                [self.xh_min, self.xv_min, -np.pi/2,       A_min,        1.0],
                [self.xh_max, self.xv_max,  np.pi/2,       A_max,     AR_max],
        )

        # calculate trajectories for each time step
        y1 = self.xh.ravel()
        z1 = self.xv.ravel()
        for itime in range(self.Ntimes):
            u1 = self.u[itime,:,:].ravel()
            def func(x):
                """objective function for x=[yc,zc,theta,Awake,AR]"""
                ang = x[2]
                sigma_z = np.sqrt(x[3] / (np.pi*x[4]))  # sqrt( A / (pi * AR) )
                sigma_y = x[4] * sigma_z
                delta_y =  x[0]*np.cos(ang) + x[1]*np.sin(ang) - y1
                delta_z = -x[0]*np.sin(ang) + x[1]*np.cos(ang) - z1
                return self.umin[itime] * \
                        np.exp( -0.5 * ((delta_y/sigma_y)**2 + (delta_z/sigma_z)**2) ) - u1

            res = least_squares(func, guess, bounds=minmax)

            if res.success:
                self.xh_wake[itime], self.xv_wake[itime] = res.x[:2]
                # calculate elliptical wake outline
                ang = res.x[2]
                sigma_z = np.sqrt(res.x[3] / (np.pi*res.x[4]))
                sigma_y = res.x[4] * sigma_z
                r = np.sqrt(2)*sigma_y*sigma_z / np.sqrt(sigma_z**2*np.cos(azi)**2 + sigma_y**2*np.sin(azi)**2)
                tmpy = r*np.cos(azi)
                tmpz = r*np.sin(azi)
                yellip =  tmpy*np.cos(ang) + tmpz*np.sin(ang) + self.xh_wake[itime]
                zellip = -tmpy*np.sin(ang) + tmpz*np.cos(ang) + self.xv_wake[itime]
                self.paths[itime] = np.vstack((yellip,zellip)).T
                self.sigma_y[itime] = sigma_y
                self.sigma_z[itime] = sigma_z
                self.rotation[itime] = ang
            else:
                self.xh_wake[itime], self.xv_wake[itime] = \
                        self.xh_fail, self.xv_fail
                self.sigma_y[itime] = np.nan
                self.sigma_z[itime] = np.nan
                self.rotation[itime] = np.nan

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._updateInertial()

        self.wakeTracked = True

        # write out everything
        self._writeTrajectory(trajectoryFile,
                self.sigma_y,
                self.sigma_z,
                self.rotation)
        self._writeOutlines(outlinesFile)
    
        return self.trajectoryIn(frame)

