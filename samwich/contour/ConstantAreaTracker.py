from __future__ import print_function
import sys

import numpy as np
from scipy.signal import convolve2d

from samwich.waketrackers import ContourWakeTracker

class ConstantArea(ContourWakeTracker):
    """Identifies a wake as a region with velocity deficit contour
    enclosing an area closest to the rotor (or another specified
    reference area).

    This is the fastest of the contour-based tracking methods since
    it does not necessarily depend on the 'contain_pts' function (unless
    check_deficit is set to True).

    Inherits class ContourWakeTracker.
    """

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        if self.verbose:
            print('\n...finished initializing',self.__class__.__name__,'\n')

    def find_centers(self,ref_area,
                     trajectory_file=None,outlines_file=None,
                     weighted_center=True,
                     contour_closure=None,
                     min_contour_points=10,
                     Ntest0=50,Ntest=4,tol=0.01,
                     umax=0,
                     check_deficit=False,uniform_filter=3,
                     vdcutoff=0.7,
                     frame='rotor-aligned',
                     verbosity=0):
        """Uses a binary search algorithm (find_contour_center) to
        locate the contour with flux closest to the targetValue.
        
        Overrides the parent find_centers routine.
        
        Parameters
        ----------
        ref_area : float
            Area to attempt to match, e.g., the rotor disk area.
        trajectory_file : string, optional
            Name of trajectory data file to attempt inputting and to
            write out to; set to None to skip I/O. Data are written out
            in the rotor-aligned frame.
        outlines_file : string, optional
            Name of pickle archive file (\*.pkl) to attempt input and to
            write out detected wake outlines; set to None to skip I/O.
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
        Ntest0 : integer, optional
            The number of initial test contours to calculate; should be
            relatively large to ensure that a global optimum is found.
        Ntest : integer, optional
            The number of test contours to calculate in each refinement
            cycle.
        uniform_filter : bool, int, optional
            Use scipy.signal.convolve2d to smooth each wake snapshot,
            which changes the detected u_min used if 'check_deficit' is
            True. If 'uniform_filter' is True, then the filter size is
            estimated as the 1/5 of the reference diameter (calculated
            from ref_area); otherwise, the filter size may be directly
            specified.
        tol : float, optional
            Minimum spacing to test during the binary search.
        umax : float, optional
            Maximum contour level to consider as a wake edge.
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
            print('Note: wake tracking has already been performed')
            return self.trajectory_in(frame)

        # filter velocity field 
        if check_deficit and (uniform_filter is not None):
            if uniform_filter == True:
                # guesstimate filter size as D/5
                dy = np.diff(self.xh_range)[0]
                dz = np.diff(self.xv_range)[0]
                ds = (dy + dz) / 2
                D = 2*(ref_area/np.pi)**0.5  # R = (A/pi)**0.5
                if verbosity > 0:
                    print('dy,dz,ds ~=',dy,dz,ds)
                size = max(int((D/5)/ds),3)
            else:
                size = int(uniform_filter)
            if uniform_filter % 2 == 0:
                # make sure filter size is odd
                uniform_filter += 1
            if verbosity > 0:
                print('uniform filter size :',size)
            filterarray = np.ones((size,size))
            filterarray /= np.sum(filterarray)
            off = int(size/2)
            if self.verbose:
                print('Applying uniform filter')
            utest = np.stack([
                convolve2d(self.u[itime,:,:],filterarray,mode='full')[off:-off,off:-off]
                for itime in range(self.Ntimes)
            ])
            assert utest.shape==self.u.shape
            check_deficit = utest

        # calculate trajectories for each time step
        if self.verbose:
            print('Attempting to match area:',ref_area,'m^2')
        for itime in range(self.Ntimes):
            _,_,info = self._find_contour_center(itime,
                                             ref_area,
                                             weighted_center=weighted_center,
                                             contour_closure=contour_closure,
                                             min_contour_points=min_contour_points,
                                             Ntest=Ntest,Ntest0=Ntest0,
                                             umax=umax,
                                             vdcheck=check_deficit,
                                             vdcutoff=vdcutoff,
                                             func=None,
                                             tol=tol,
                                             debug=(verbosity > 0))
            if not info['success']:
                print('WARNING: find_contour_center was unsuccessful.')
                print(info)
                print('Try re-running with verbosity > 0;'
                      ' if contours found were ~0, try re-running with'
                      ' fewer min_contour_points')
            elif self.verbose and verbosity > 0:
                print('itime={:.1f} : found contour (u={:.4f}) with area {:g}'.format(
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

