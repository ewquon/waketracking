from __future__ import print_function
import sys
import importlib

import numpy as np
from scipy.optimize import least_squares

from samwich.waketrackers import WakeTracker

class Gaussian(WakeTracker):
    """Identifies a wake as the best fit to a univariate Gaussian
    distribution described by:

    .. math::

        f(y,z) = A \exp \left(
            -\\frac{1}{2} \\frac{(y-y_c)^2 + (z-z_c)^2}{\sigma^2}
        \\right)

    Inherits class WakeTracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,
                     umin=None,sigma=None,
                     res=100,plotscale=2.0,
                     trajectory_file=None,outlines_file=None,
                     frame='rotor-aligned',
                     verbosity=0):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit to the wake center location, given the wake width.
        
        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        umin : float or ndarray
            Max velocity deficit (i.e., should be < 0). If None, then
            this is detected from the field.
        sigma : float or ndarray
            Wake width parameter, equivalent to the standard deviation
            of the Gaussian function. This may be a constant or a
            function of downstream distance.
        res : integer, optional
            Number of points to represent the wake outline as a circle.
        plotscale : float, optional
            Scaling factor in standard deviations for a representative
            wake outline (==1.1774 for FWHM, 1.96 for a 95% CI).
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
            print('Note: wake tracking has already been performed')
            return self.trajectory_in(frame)

        # setup Gaussian parameters
        if self.shear_removal is None:
            print('Note: remove_shear has not been called')
        if umin is None:
            # calculate umin available data
            self.umin = np.nanmin(self.u,axis=(1,2))
            print('Calculated umin=',self.umin)
        elif isinstance(umin,np.ndarray):
            # specified umin as array with length Ntimes
            self.umin = umin
        else:
            # specified constant umin
            self.umin = umin * np.ones(self.Ntimes)

        if not np.all(self.umin < 0):
            print('Warning: Unexpected positive velocity deficit at',
                    len(np.nonzero(self.umin > 0)[0]),'of',self.Ntimes,'times')
        if self.verbose:
            print('Average Gaussian function amplitude =',
                    np.mean(self.umin),'m/s (over',self.Ntimes,'times)')

        # set up wake width
        if hasattr(sigma,'__iter__'):
            # sigma is a specified list-like object
            self.sigma = np.array(sigma)
            constant_sigma = False
            refarea = np.pi*self.sigma**2
            print('Mean/min/max reference Gaussian area:',
                    np.mean(refarea),np.min(refarea),np.max(refarea),'m^2')
        else:
            constant_sigma = True
            try:
                # sigma is a specified constnat
                self.sigma = float(sigma) * np.ones(self.Ntimes)
                if self.verbose:
                    print('Specified Gaussian width =',self.sigma[0],'m')
            except TypeError:
                # sigma is specified as a function of downstream distance
                assert(callable(sigma))
                xd = np.mean(self.xd)
                self.sigma = sigma(xd) * np.ones(self.Ntimes)
                if self.verbose:
                    print('Calculated sigma =',self.sigma[0],'m at x=',xd,'m')
            if self.verbose:
                print('Reference Gaussian area =',np.pi*self.sigma[0]**2,'m^2')

        # approximate wake outline with specified wake width, sigma
        azi = np.linspace(0,2*np.pi,res+1)
        ycirc = plotscale*np.cos(azi)
        zcirc = plotscale*np.sin(azi)

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
            u1 = self.u[itime,:,:].ravel().copy()
            u1[np.isnan(u1)] = 0.
            def func(x):
                """Residuals for x=[yc,zc]"""
                delta_y = y1 - x[0]
                delta_z = z1 - x[1]
                resid = self.umin[itime] \
                        * np.exp(-0.5*(delta_y**2 + delta_z**2)/self.sigma[itime]**2
                                ) - u1
                return resid
            result = least_squares(func, guess, bounds=minmax)
            if result.success:
                self.xh_wake[itime], self.xv_wake[itime] = result.x[:2]
                self.paths[itime] = np.vstack((self.sigma[itime]*ycirc + self.xh_wake[itime],
                                               self.sigma[itime]*zcirc + self.xv_wake[itime])).T
            else:
                self.xh_wake[itime] = self.xh_fail
                self.xv_wake[itime] = self.xv_fail

            if self.verbose:
                if verbosity > 0:
                    plotlevel = self.umin[itime] * np.exp(-0.5*plotscale**2)
                    #print(f'yc,zc : {self.xh_wake[itime]:.1f},',
                    #        f' {self.xv_wake[itime]:.1f}',
                    #        f' (outline level={plotlevel})')
                    print('yc,zc : {:.1f}, {:.1f} (outline level={})'.format(self.xh_wake[itime],
                                                                             self.xv_wake[itime],
                                                                             plotlevel))
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._update_inertial()

        self.wake_tracked = True

        # write out everything
        if constant_sigma:
            self._write_trajectory(trajectory_file)
        else:
            self._write_trajectory(trajectory_file,self.sigma)
        self._write_outlines(outlines_file)
    
        return self.trajectory_in(frame)

