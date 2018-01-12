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
                     flux_function,field_names=('u_tot',),
                     trajectory_file=None,outlines_file=None,
                     weighted_center=True,
                     contour_closure=None,
                     min_contour_points=50,
                     frame='rotor-aligned',
                     Ntest=21,tol=0.01,
                     check_deficit=True,
                     verbosity=0):
        """Uses a binary search algorithm (find_contour_center) to
        locate the contour with flux closest to the targetValue.

        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        ref_flux : float
            Numerically integrated value to match.
        flux_function : callable
            The integrand.
        field_names : tuple, optional
            List of field names to use as parameters for flux_function;
            number of arguments to flux_function should match the length
            of this list. The instantaneous velocity, 'u_tot', is used
            by default. Other possibilities include 'u' ('u_tot' minus
            freestream shear), 'v', 'w', etc.
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
            This can also be a univariate weighting function (of
            velocity deficit).
        contour_closure : string, optional
            If 'simple', then open paths with endpoints on the same
            edge will be closed by connecting the endpoints; if
            'compound', then open paths will be closed by adding
            segments along the domain boundaries to form closed
            contours.
        min_contour_points : int, optional
            Minimum number of points a closed loop must contain for it
            to be considered a candidate path. This is indirectly
            related to the smallest allowable contour region.
        frame : string, optional
            Reference frame, either 'inertial' or 'rotor-aligned'.
        Ntest : integer, optional
            The number of initial test contours to calculate.
        tol : float, optional
            Minimum spacing to test during the binary search.
        check_deficit : boolean, optional
            If True, only consider wake candidates in which the average
            velocity deficit is less than 0.
        verbosity : int, optional
            Control verbose output level (e.g., for debug).

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

        # make sure we can get the requested field for calculating flux
        test_fields = [ getattr(self,fieldname)
                        for fieldname in field_names ]

        # some sanity checks if needed
        if self.verbose:
            Utest = [ np.min(testfield[-1,self.jmin:self.jmax,self.kmin:self.kmax])
                        for testfield in test_fields ]
            input_string = '{}={}'.format(field_names[0],Utest[0])
            for fieldname, fieldvalue in zip(field_names[1:],Utest[1:]):
                input_string += ',{:s}={:g}'.format(fieldname,fieldvalue)
            #if flux_function.func_code.co_argcount==1: # fn(u)
            print('Sample function evaluation: f({:s}) = {}'.format(
                    input_string,flux_function(*Utest)))
            #else: # fn(u,A)
            #    print('Sample function evaluation: f(u={:g},1.0) = {:g}'.format(
            #            Utest,flux_function(Utest,1)))

        if contour_closure is None or contour_closure=='none':
            closure = False
        elif contour_closure=='simple':
            closure = True
        elif contour_closure=='compound':
            closure = (self.xh_range,self.xv_range)

        # calculate trajectories for each time step
        if self.verbose:
            print('Attempting to match integral:',ref_flux)
        for itime in range(self.Ntimes):
            _,_,info = self._find_contour_center(itime,
                                             ref_flux,
                                             weighted_center=weighted_center,
                                             contour_closure=closure,
                                             min_contour_points=min_contour_points,
                                             Ntest=Ntest,
                                             tol=tol,
                                             func=flux_function,
                                             fields=field_names,
                                             vdcheck=check_deficit,
                                             debug=(verbosity > 0))
            if not info['success']:
                print('WARNING: find_contour_center was unsuccessful.')
            elif self.verbose and verbosity > 0:
                print('itime={:d} : found contour (u={:.4f}) with integral {:g}'.format(
                        itime,self.Clevels[itime],self.Cfvals[itime]))

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

