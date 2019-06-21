import numpy as np

class Turbine(object):
    """turbine information container"""

    # turbine definitions here
    V27 = {
        'D': 27.0, # rotor diameter [m]
        'zhub': 32.1, # hub height [m]
        'a': 0.3, # induction factor, for momentum theory estimate of mass/momentum flux
    }

    def __init__(self,name):
        self.name = name
        defdict = getattr(self, name)
        for key,val in defdict.items():
            setattr(self, key, val)
        self.rotor_area = np.pi/4 * self.D**2
        self.ref_CT = 4*self.a*(1-self.a) # thrust coefficient
        self.ref_CP = 4*self.a*(1-self.a)**2 # power coefficient

    def __str__(self):
        s = self.name + '\n' + len(self.name)*'-'
        s += '\nhub height: {:g} m'.format(self.zhub)
        s += '\ndiameter: {:g} m'.format(self.D)
        s += '\nref area: {:g} m^2'.format(self.rotor_area)
        s += '\nref CP: {:g}'.format(self.ref_CP)
        s += '\nref CT: {:g}'.format(self.ref_CT)
        return s


