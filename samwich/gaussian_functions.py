from __future__ import print_function
import numpy as np

class PorteAgel(object):
    """Based on model described in Bastankhah and Porte-Agel, "A new
    analytical model for wind-turbine wakes," Renewable Energy 70 (2014)
    pp.116--123. 
    """

    def __init__(self,
                 CT,d0,kstar=None,
                 TI=None,ka=0.3837,kb=0.003678,
                 verbose=True):
        """Initialize the 1-D Gaussian model, and calculate required
        parameters. The wake growth rate should either be directly 
        specified or calculated from a specified turbulence intensity.

        Parameters
        ----------
        CT : float
            Reference thrust coefficient.
        d0 : float
            Wind turbine rotor diameter. [m]
        kstar : float
            Wake growth rate, equivalent to sigma/d0 for x-->0.
        TI : float, optional
            Local turbulence intensity at the rotor hub height, used to
            calculate the k* by the empirical model described in
            Niayifar and Porte-Agel, "A new analytical model for wind 
            farm power prediction." [-]
        ka, kb : float, optional
            Empirical parameters for the k* model.
        """
        if (kstar is None) and (TI is None):
            raise ValueError('Need to specify wake growth rate ("kstar"),' \
                    ' or turbulence intensity ("TI") along with optional' \
                    ' "ka" and "kb" empirical parameters)')
        self.verbose = verbose
        self.CT = CT
        self.d0 = d0
        if kstar is not None:
            self.kstar = kstar
            self.TI = None
            self.ka = None
            self.kb = None
        else:
            self.TI = TI
            self.ka = ka
            self.kb = kb
            self.kstar = ka*TI + kb
            if self.verbose: print('Calculated k* :',self.kstar)
        self.beta = (1 + np.sqrt(1-CT))/(2*np.sqrt(1-CT))  # Eqn. 6

    def amplitude(self,x,Uref=1.0,check_validity=True):
        """Returns the amplitude of the Porte-Agel 1-D Gaussian
        function. By default, this is the velocity deficit (positive
        by convention) normalized by the freestream velocity Uref.
        """
        radic = 1 - self.CT / (8 * (self.sigma(x)/self.d0)**2)
        if check_validity:
            # ensure that the radicand is >= 0
            radic = max(radic,0)
        A = Uref*(1 - np.sqrt(radic))
        if self.verbose: print('Calculated Gaussian amplitude :',A,'m/s')
        return A

    def sigma(self,x=0.0,eps_coeff=0.2):
        """Epsilon is the limit of the normalized wake width as x-->0.
        The theoretical value is 0.25*sqrt(beta). However, Bastankhah
        and Porte-Agel (2014) recommend epsilon = 0.2*sqrt(beta) for 
        better correspondence with LES data.
        """
        sigma = self.kstar*x + eps_coeff*self.d0*np.sqrt(self.beta)
        if self.verbose: print('Calculated Gaussian width :',sigma,'m')
        return sigma

