from __future__ import print_function
import sys
import importlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from samwich.waketrackers import WakeTracker

class CircularTestRegion(WakeTracker):
    """Identifies a wake as the circular region that minimizes a given
    function. For instance, this is used to apply the minimum-power 
    method proposed by Vollmer et al in Wind Energ. Sci. 2016. 

    Least-squares optimization is performed to identify the wake center.
    Sampling is assumed to be on a uniformly spaced grid.

    Inherits class WakeTracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,test_radius,
                     test_function=lambda u: u**3,
                     test_field='u_tot',
                     Nradial=25, Nazimuthal=24,
                     res=100,
                     trajectory_file=None,outlines_file=None,
                     frame='rotor-aligned',
                     verbosity=0):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit.

        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        test_radius : float
            Defines the circular test region.
        test_function : callable
            The integrand.
        test_field : string, optional
            Name of the field to use as input to test_function; the
            instantaneous velocity, 'u_tot', is used by default.
        Nradial : float, optional
            Number of radial test points to which to interpolate.
        Nazimuthal : float, optional
            Number of azimuthal test points to which to interpolate.
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
            print('Note: wake tracking has already been performed')
            return self.trajectory_in(frame)

        try:
            utest = getattr(self,test_field)
        except AttributeError:
            print('Warning: test field',test_field,'not available,',
                    "using 'u_tot' by default")
            test_field = 'u_tot'
            utest = getattr(self,test_field)

        # some sanity checks
        if self.shear_removal is None:
            print('Note: remove_shear has not been called')

        if self.verbose:
            print('Searching within a region with radius={}'.format(test_radius))

        # set up optimization parameters
        guess = [
                (self.xh_min+self.xh_max)/2,    # 0: yc
                (self.xv_min+self.xv_max)/2,    # 1: zc
        ]
        minmax = (
                # y range,                z range
                [self.xh_min+test_radius, self.xv_min+test_radius],
                [self.xh_max-test_radius, self.xv_max-test_radius],
        )

        azi = np.linspace(0,2*np.pi,res)
        ycirc = test_radius*np.cos(azi)
        zcirc = test_radius*np.sin(azi)

        # set up test grid
        r1 = np.linspace(0,test_radius,Nradial+1)[1:]
        q1 = np.linspace(0,2*np.pi,Nazimuthal+1)[:-1]
        dr = r1[1] - r1[0]
        dq = q1[1] - q1[0]
        testr, testq = np.meshgrid(r1,q1,indexing='ij')
        testr = testr.ravel()
        testq = testq.ravel()
        testy = testr*np.cos(testq)
        testz = testr*np.sin(testq)

        # calculate trajectories for each time step
        y1 = self.xh.ravel()
        z1 = self.xv.ravel()
        for itime in range(self.Ntimes):
            u1 = utest[itime,:,:].ravel()
# Note: This approach causes scipy.optimize.least_squares to fail because it
#       perturbs the guess value by eps, which isn't enough to return a
#       different set of points and thus all initial function evaluations
#       returns the same value. In other words, minimize_error thinks that
#       there the field is constant.
#
#            def func(x):
#                """objective function for x=[yc,zc]"""
#                r2_from_center = (y1-x[0])**2 + (z1-x[1])**2
#                in_test_region = (r2_from_center < test_radius**2)
#                integrand = test_function(u1[in_test_region])
#                if verbosity > 0:
#                    print(f' {len(np.nonzero(in_test_region)[0])} points in',
#                            f' test region, around ({x[0]},{x[1]})'
#                            f' fval={np.sum(integrand)}')
#                return np.sum(integrand)
            interp_fun = \
                    RegularGridInterpolator((self.xh[:,0],self.xv[0,:]),
                                            utest[itime,:,:])
            def func(x):
                """objective function for x=[yc,zc]"""
                pts = np.array([testy + x[0], testz + x[1]]).T
                uinterp = interp_fun(pts)
                integrand = test_function(uinterp)
                return np.sum(integrand*testr)*dr*dq  # integrate f*r*dr*dtheta
            result = least_squares(func, guess, bounds=minmax,
                                   verbose=verbosity)
            if result.success:
                self.xh_wake[itime], self.xv_wake[itime] = result.x[:2]
                self.paths[itime] = np.vstack((ycirc + self.xh_wake[itime],
                                               zcirc + self.xv_wake[itime])).T
            else:
                self.xh_wake[itime] = self.xh_fail
                self.xv_wake[itime] = self.xv_fail

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

