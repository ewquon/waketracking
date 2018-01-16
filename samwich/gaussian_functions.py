from __future__ import print_function
import numpy as np

class PorteAgel(object):
    """Based on model described in Bastankhah and Porte-Agel, "A new
    analytical model for wind-turbine wakes," Renewable Energy 70 (2014)
    pp.116--123. 
    """

    def __init__(self,
                 CT,d0,kstar=None,
                 TI=None,ka=0.3837,kb=0.003678):
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
            print('Calculated k* = ',self.kstar)
        self.beta = (1 + np.sqrt(1-CT))/(2*np.sqrt(1-CT))  # Eqn. 6

    def amplitude(self,x=0.0,Uref=1.0):
        """Returns the amplitude of the Porte-Agel 1-D Gaussian
        function. By default, this is the velocity deficit (positive
        by convention) normalized by the freestream velocity Uref.
        """
        return Uref*(
            1 - np.sqrt(
                1 - self.CT
                        / (8 * (self.sigma(x)/self.d0)**2)
            )
        )

    def sigma(self,x=0.0):
        return self.kstar*x + 0.2*self.d0*np.sqrt(self.beta)

