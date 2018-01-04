from __future__ import print_function
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
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,ref_flux,
                     flux_function,flux_field='u_tot',
                     trajectory_file=None,outlines_file=None,
                     weighted_center=True,
                     contour_closure=None,
                     frame='rotor-aligned',
                     Ntest=21,tol=0.01,
                     check_deficit=True,
                     debug=False):
        """Uses a binary search algorithm (find_contour_center) to
        locate the contour with flux closest to the targetValue.
        
        Some candidate functions to be integrated over the contour
        surface:

        * mass flow, func = lambda u: u
        * momentum flux, func = lambda u: u**2

        The contour area can be referenced as func = lambda u,A: ...

        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        ref_flux : float
            Flux to attempt to match, e.g., the massflow rate.
         : function
            A specified function of one (or two) variables, the velocity
            deficit (and the contour area).
        flux_field : string, optional
            Name of the field to use as input to the flux_function; use
            the instantaneous velocity, 'u_tot', by default.
        trajectory_file : string
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlines_file : string
            Name of pickle archive file (\*.pkl) to attempt input and to
            write out detected contour outlines; set to None to skip
            I/O.
        weighted_center : boolean or function, optional
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
        check_deficit : boolean, optional
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
        self.clear_plot()

        # try to read trajectories (required) and outlines (optional)
        self._read_trajectory(trajectory_file)
        self._read_outlines(outlines_file)

        # done if read was successful
        if self.wake_tracked:
            return self.trajectory_in(frame)

        try:
            test_field = getattr(self,flux_field)
        except AttributeError:
            print('Warning: flux field',flux_field,'not available,',
                    "using 'u_tot' by default")
            flux_field = 'u_tot'
            test_field = getattr(self,flux_field)

        # some sanity checks if needed
        if self.verbose:
            Utest = np.min(test_field[-1,self.jmin:self.jmax,self.kmin:self.kmax])
            if flux_function.func_code.co_argcount==1: # fn(u)
                print('Sample function evaluation: f(u={:g}) = {:g}'.format(
                        Utest,flux_function(Utest)))
            else: # fn(u,A)
                print('Sample function evaluation: f(u={:g},1.0) = {:g}'.format(
                        Utest,flux_function(Utest,1)))
            print(' ~= targetValue / area =',ref_flux,'/ A')

        if contour_closure is None or contour_closure=='none':
            closure = False
        elif contour_closure=='simple':
            closure = True
        elif contour_closure=='compound':
            closure = (self.xh_range,self.xv_range)

        # calculate trajectories for each time step
        for itime in range(self.Ntimes):
            _,_,info = self._find_contour_center(itime,
                                                 ref_flux,
                                                 weighted_center=weighted_center,
                                                 contour_closure=closure,
                                                 Ntest=Ntest,
                                                 tol=tol,
                                                 func=flux_function,
                                                 field=flux_field,
                                                 vdcheck=check_deficit,
                                                 debug=debug)
            if not info['success']:
                print('WARNING: find_contour_center was unsuccessful.')

            if self.verbose:
                sys.stderr.write('\rProcessed frame {:d}'.format(itime))
                #sys.stderr.flush()
        if self.verbose: sys.stderr.write('\n')

        self._update_inertial()

        self.wake_tracked = True

        # write out everything
        self._write_trajectory(trajectory_file, self.Clevels, self.Cfvals)
        self._write_outlines(outlines_file)
    
        return self.trajectory_in(frame)

