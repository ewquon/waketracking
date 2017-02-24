import os
import importlib

from waketrackers import contourwaketracker
import contour

class ConstantArea(contourwaketracker):
    """ Identifies a wake as a region with velocity contour enclosing an
    area closest to the rotor (or another specified reference area).

    The wake center is identified as the velocity-deficit-weighted 
    "center of mass" of all points within the enclosed region. 

    This is the fastest of the contour-based tracking methods.
    """

    def __init__(self,*args,**kwargs):
        print 'CONSTANT AREA METHOD'
        super(self.__class__,self).__init__(*args,**kwargs)

