from __future__ import print_function
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
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,
                     umin=None,
                     A_min=100.0,A_max=np.inf,
                     A_ref=np.pi*63.**2,
                     AR_max=10.0,
                     rho=None,
                     weighting=1.0,
                     res=100,plotscale=2.0,
                     trajectory_file=None,outlines_file=None,
                     frame='rotor-aligned',
                     verbosity=0):
        """Uses optimization algorithms in scipy.optimize to determine
        the best fit.

        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        umin : float or ndarray
            Max velocity deficit (i.e., should be < 0). If None, then
            this is detected from the field.
        A_min : float
            The minimum approximate wake area, where wake area is
            estimated to be the pi times the product of the wake widths.
        A_max : float
            The maximum approximate wake area, where wake area is
            estimated to be the pi times the product of the wake widths.
            If None, then there is no limit on the "optimal" wake size.
        A_ref : float
            Reference area used to provide an initial guess and scale
            the least-squares optimization weighting function.
        AR_max : float, optional
            The maximum aspect ratio between the wake widths in two
            directions (analogous to the semi-major and -minor axes in
            an ellipse); this dictates the maximum amount of stretching.
        rho : float, optional
            The cross-correlation parameter--CURRENTLY UNTESTED.
        weighting : float, optional
            The width of the exponential weighting function, specified
            as the number of reference radii (calculated from the 
            reference area).
        res : integer, optional
            Number of points to represent the wake outline as a circle
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
        u_in_range = self.u[:,self.jmin:self.jmax+1,self.kmin:self.kmax+1]
        if self.shear_removal is None:
            print('Note: remove_shear has not been called')
        if umin is None:
            # calculate umin available data
            #self.umin = np.min(self.u,axis=(1,2))
            self.umin = np.min(u_in_range,axis=(1,2))
        elif isinstance(umin,np.ndarray):
            # specified umin as array with length Ntimes
            self.umin = umin
        else:
            # specified constant umin
            self.umin = umin * np.ones(self.Ntimes)
        self.rotation = np.zeros(self.Ntimes)
        self.sigma_y = np.zeros(self.Ntimes)
        self.sigma_z = np.zeros(self.Ntimes)

        if not np.all(self.umin < 0):
            print('Warning: Unexpected positive velocity deficit at',
                    len(np.nonzero(self.umin > 0)[0]),'of',self.Ntimes,'times')
        if self.verbose:
            print('average Gaussian function amplitude =',
                    np.mean(self.umin),'m/s (over',self.Ntimes,'times)')

        if rho is not None:
            print('Note: cross-correlation parameter is not yet implemented')

        # set up optimization parameters
        # note: origin of the rotor-aligned frame is at the center of the sampling
        #   plane already
        # note: sigma_y = AR*sigma_z
        #       sigma_z = sqrt(A/(pi*AR))
        x0 = [
            None,  # 0: yc
            None,  # 1: zc
            0.0,   # 2: theta
            A_ref, # 3: A == pi * sigma_y * sigma_z
            1.0,   # 4: AR == sigma_y/sigma_z
        ]
        minmax = (
            # y range,    z range,     rotation range, wake size, wake stretching
            [self.xh_min, self.xv_min, -np.pi/2,       A_min,        1.0],
            [self.xh_max, self.xv_max,  np.pi/2,       A_max,     AR_max],
        )
        sigma0_sq = A_ref/np.pi # R = (A/pi)**0.5

        # calculate trajectories for each time step
        y1 = self.xh.ravel()
        z1 = self.xv.ravel()
        azi = np.linspace(0,2*np.pi,res+1)
        for itime in range(self.Ntimes):
            u1 = self.u[itime,:,:].ravel()
            def fun1(x):
                """Residuals for x=[yc,zc]"""
                delta_y = y1 - x[0]
                delta_z = z1 - x[1]
                return self.umin[itime] \
                        * np.exp(-0.5*(delta_y**2 + delta_z**2)/sigma0_sq
                                ) - u1
            def fun2(x):
                """Residuals for x=[yc,zc,theta,A,AR], with m DOFs and n=5
                variables"""
                yc,zc,theta,A,AR = x
                sigz2 = A / (np.pi*AR)  # sigma_z**2
                delta_y =  (y1-yc)*np.cos(theta) + (z1-zc)*np.sin(theta)
                delta_z = -(y1-yc)*np.sin(theta) + (z1-zc)*np.cos(theta)
                #expfun = np.exp(-0.5*((delta_y/AR)**2 + delta_z**2)/sigz2)
                #W = np.sqrt(expfun)
                # weighting function changes in time
                #sigma_sq = max(sigma0_sq,sigz2,sigz2*AR)
                #sigma_sq *= weighting**2
                # weighting function is constant in time
                sigma_sq = sigma0_sq * weighting**2
                W = np.sqrt(
                    np.exp(-0.5*((y1-yc)**2 + (z1-zc)**2)/sigma_sq)
                )
                resid = u1 - self.umin[itime]*np.exp(-0.5*((delta_y/AR)**2 + delta_z**2)/sigz2)
                return W*resid
#            def jac(x):
#                """Exact jacobian (m by n) matrix"""
#                yc,zc,theta,A,AR = x
#                delta_y =  (y1-yc)*np.cos(theta) + (z1-zc)*np.sin(theta)
#                delta_z = -(y1-yc)*np.sin(theta) + (z1-zc)*np.cos(theta)
#                coef = -np.pi/2 * fun2(x)
#                jac = np.empty((len(u1),5))
#                jac[:,0] =  2*coef/A * (-delta_y/AR*np.cos(theta) + AR*delta_z*np.sin(theta))
#                jac[:,1] = -2*coef/A * ( delta_y/AR*np.sin(theta) + AR*delta_z*np.cos(theta))
#                jac[:,2] =  2*coef/A * delta_y * delta_z * (-AR + 1./AR)
#                jac[:,3] = -coef/A**2 * (delta_y**2/AR + AR*delta_z**2)
#                jac[:,4] =  coef/A * (-delta_y**2/AR**2 + delta_z**2)
#                return jac

            # start at center of plane
            guess = [(self.xh_min+self.xh_max)/2, (self.xv_min+self.xv_max)/2]

            # perform 1-D Gaussian to get a better guess for 2-D
            minmax_1d = (
                    [self.xh_min,self.xv_min],
                    [self.xh_max,self.xv_max],
            )
            result1 = least_squares(fun1, guess,bounds=minmax_1d)
            if result1.success:
                guess = result1.x
                if verbosity > 0:
                    print('1D-Gaussian guess:',guess)
            else:
                # if the 1-D Gaussian wake identificaiont failed, fall back on
                # the next best thing--the location of the largest velocity
                # deficit
                i,j = np.unravel_index(np.argmin(u1), (self.Nh,self.Nv))
                guess = [self.xh[i,j], self.xv[i,j]]
                if self.verbose:
                    print('Falling back on u_min guess:',guess)

            # now solve the harder optimization problem
            x0[0:2] = guess
            result = least_squares(fun2, x0, #jac=jac,
                                   ftol=1e-14,
                                   xtol=1e-14,
                                   gtol=1e-14,
                                   bounds=minmax,
                                   verbose=verbosity)
            if verbosity > 0:
                print(result)
            if result.success:
                # calculate elliptical wake outline
                yc,zc,theta,A,AR = result.x
                sigma_z = np.sqrt(A / (np.pi*AR))
                sigma_y = AR * sigma_z
                tmpy = plotscale*sigma_y*np.cos(azi) # unrotated ellipse
                tmpz = plotscale*sigma_z*np.sin(azi)
                yellipse = yc + tmpy*np.cos(theta) - tmpz*np.sin(theta)
                zellipse = zc + tmpy*np.sin(theta) + tmpz*np.cos(theta)
                self.paths[itime] = np.vstack((yellipse,zellipse)).T
                self.xh_wake[itime] = yc
                self.xv_wake[itime] = zc
                self.sigma_y[itime] = sigma_y
                self.sigma_z[itime] = sigma_z
                self.rotation[itime] = theta
            else:
                self.xh_wake[itime] = self.xh_fail
                self.xv_wake[itime] = self.xv_fail
                self.sigma_y[itime] = np.nan
                self.sigma_z[itime] = np.nan
                self.rotation[itime] = np.nan

            if self.verbose:
                if verbosity > 0:
                    A = np.pi * self.sigma_y[itime] * self.sigma_z[itime]
                    plotlevel = self.umin[itime] * np.exp(-0.5*plotscale**2)
                    #print(f'yc,zc : {self.xh_wake[itime]:.1f},',
                    #        f' {self.xv_wake[itime]:.1f};',
                    #        f' rotation={self.rotation[itime]*180/np.pi} deg;',
                    #        f' ref wake area={A} m^2'
                    #        f' (outline level={plotlevel})')
                    print(('yc,zc : {:.1f}, {:.1f};' \
                          +' rotation={} deg; ref wake area={} m^2' \
                          +' (outline level={})').format(self.xh_wake[itime],
                                                         self.xv_wake[itime],
                                                         self.rotation[itime]*180/np.pi,
                                                         A,
                                                         plotlevel))
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._update_inertial()

        self.wake_tracked = True

        # write out everything
        self._write_trajectory(trajectory_file,
                               self.sigma_y,
                               self.sigma_z,
                               self.rotation)
        self._write_outlines(outlines_file)
    
        return self.trajectory_in(frame)

