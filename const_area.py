from waketracker import WakeTracker

class const_area(WakeTracker):
    """ Identifies a wake as a region with velocity contour enclosing an
    area closest to the rotor (or another specified reference area).

    The wake center is identified as the velocity-deficit-weighted 
    "center of mass" of all points within the enclosed region. 

    This is the fastest of the contour-based tracking methods.
    """

    def __init__(self):
        print 'constant area method'
