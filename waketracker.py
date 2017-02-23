import os

class WakeTracker(object):
    """ A general class for wake tracking operations.

    Tracking algorithms are expected to be implemented as children of
    this class.
    """

    def __init__(self,dataLoaderObject):
        print 'Processing data from',dataLoaderObject
