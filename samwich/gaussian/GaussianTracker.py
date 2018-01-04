from __future__ import print_function
import sys
import importlib

import numpy as np
from scipy.optimize import least_squares

from samwich.waketrackers import waketracker

class Gaussian(waketracker):
    """Identifies a wake as the best fit to a univariate Gaussian
    distribution described by:

    .. math::

        f(y,z) = A \exp \left(
            -\\frac{1}{2} \\frac{(y-y_c)^2 + (z-z_c)^2}{\sigma^2}
        \\right)

    Inherits class waketracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,
                     umin=None,
                     sigma=50.,
                     res=100,
                     trajectory_file=None,outlines_file=None,
                     frame='rotor-aligned'):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit to the wake center location, given the wake width.
        
        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        umin : float or ndarray
            Max velocity deficit (i.e., should be < 0). If None, then
            this is detected from the field.
        sigma : float
            Wake width parameter, equivalent to the standard deviation
            of the Gaussian function. This may be a constant or a
            function of downstream distance.
        res : integer, optional
            Number of points to represent the wake outline as a circle
        trajectory_file : string, optional
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlines_file : string, optional
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
        self.clear_plot()

        # try to read trajectories (required) and outlines (optional)
        self._read_trajectory(trajectory_file)
        self._read_outlines(outlines_file)

        # done if read was successful
        if self.wake_tracked:
            return self.trajectory_in(frame)

        # setup Gaussian parameters
        if self.shear_removal is None:
            print('Note: remove_shear has not been called')
        if umin is None:
            # calculate umin available data
            self.umin = np.min(self.u,axis=(1,2))
        elif isinstance(umin,np.ndarray):
            # specified umin as array with length Ntimes
            self.umin = umin
        else:
            # specified constant umin
            self.umin = umin * np.ones(self.Ntimes)

        if not np.all(umin < 0):
            print('Warning: Unexpected positive velocity deficit at',
                    len(np.nonzero(self.umin > 0)[0]),'of',self.Ntimes,'times')
        if self.verbose:
            print('average Gaussian function amplitude =',
                    np.mean(self.umin),'m/s')

        try:
            # sigma is a specified constnat
            self.sigma = float(sigma)
            if self.verbose:
                print('Specified Gaussian width =',self.sigma,'m')
        except TypeError:
            # sigma is specified as a function of downstream distance
            xd = np.mean(self.xd)
            self.sigma = sigma(xd)
            if self.verbose:
                print('Calculated sigma =',self.sigma,'m at x=',xd,'m')

        # approximate wake outline with specified wake width, sigma
        azi = np.linspace(0,2*np.pi,res)
        ycirc = self.sigma*np.cos(azi)
        zcirc = self.sigma*np.sin(azi)

        # set up optimization parameters
        guess = [
                (self.xh_min+self.xh_max)/2,
                (self.xv_min+self.xv_max)/2,
        ]
        minmax = (
                [self.xh_min,self.xv_min],
                [self.xh_max,self.xv_max],
        )

        # calculate trajectories for each time step
        y1 = self.xh.ravel()
        z1 = self.xv.ravel()
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
                self.paths[itime] = np.vstack((ycirc + self.xh_wake[itime],
                                               zcirc + self.xv_wake[itime])).T
            else:
                self.xh_wake[itime], self.xv_wake[itime] = \
                        self.xh_fail, self.xv_fail

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._update_inertial()

        self.wake_tracked = True

        # write out everything
        self._write_trajectory(trajectory_file)
        self._write_outlines(outlines_file)
    
        return self.trajectory_in(frame)

