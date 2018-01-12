from __future__ import print_function
import numpy as np

class PorteAgel(object):
    """Based on model described in Bastankhah and Porte-Agel, "A new
    analytical model for wind-turbine wakes," Renewable Energy 70 (2014)
    pp.116--123. 
    """

    def __init__(self,CT,kstar,d0):
        """
        Parameters
        ----------
        CT : float
            Reference thrust coefficient.
        kstar : float
            Wake growth rate, equivalent to sigma/d0 for x-->0.
        d0 : float
            Wind turbine rotor diameter.
        """
        self.CT = CT
        self.kstar = kstar
        self.d0 = d0
        self.beta = (1 + np.sqrt(1-CT))/(2*np.sqrt(1-CT))  # Eqn. 6

    def amplitude(self,x=0.0,Uref=1.0):
        """Returns the amplitude of the Porte-Agel 1-D Gaussian
        function. By default, this is the velocity deficit (positive
        by convention) normalized by the freestream velocity Uref.
        """
        return Uref*(
            1 - np.sqrt(
                1 - self.CT
                        / (8 * (self.kstar*x/self.d0
                                + 0.2*np.sqrt(self.beta)
                               )**2
                        )
            )
        )

    def sigma(self,x=0.0):
        return self.kstar*x + 0.2*self.d0*np.sqrt(self.beta)

