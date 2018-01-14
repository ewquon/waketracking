from __future__ import print_function
import os

class TimeSeries(object):
    """ Object for holding general time series data which may be stored
    in multiple time subdirectories
    """

    def __init__(self,datadir='.',filename=None,verbose=True):
        """ Collect data from subdirectories, assuming that subdirs
        have a name that can be cast as a float
        """
        self.dataDir = os.path.abspath(datadir)
        self.filename = filename
        self.outputTimes = []
        self.outputNames = []
        self.dirlist = []
        self.filelist = None
        self.lastfile = -1  # for iterator
        self.verbose = verbose

        # process all subdirectories
        subdirs = [ os.path.join(self.dataDir,d)
                    for d in os.listdir(self.dataDir)
                    if os.path.isdir(os.path.join(self.dataDir,d)) ]
        for path in subdirs:
            dname = os.path.split(path)[-1]
            try:
                tval = float(dname)
            except ValueError:
                continue
            self.outputTimes.append(tval)
            self.dirlist.append(path)
        self.Ntimes = len(self.dirlist)
    
        # sort by output time
        iorder = [kv[0] for kv in sorted(enumerate(self.outputTimes),key=lambda x:x[1])]
        self.dirlist = [self.dirlist[i] for i in iorder]
        self.outputTimes = [self.outputTimes[i] for i in iorder]

        # check that all subdirectories contain the same files
        self.outputNames = os.listdir(self.dirlist[0])
        for d in self.dirlist:
            if not os.listdir(d) == self.outputNames:
                print('Warning: not all subdirectories contain the same files')
                break
        if verbose:
            self.outputs() # print available outputs

        # set up file list
        if filename is not None:
            self.setFilename(filename)

    def setFilename(self,filename):
        """Update file list for iteration"""
        self.lastfile = -1  # reset iterator index
        self.filelist = []
        for path in self.dirlist:
            fpath = os.path.join(path,filename)
            if os.path.isfile(fpath):
                self.filelist.append(fpath)
            else:
                raise IOError(fpath+' not found')

    def outputs(self,prefix=''):
        """Print available outputs for the given data directory"""
        selectedOutputNames = [ name for name in self.outputNames if name.startswith(prefix) ]
        if self.verbose:
            if prefix:
                print('Files starting with "{}" in each subdirectory:'.format(prefix))
            else:
                print('Files in each subdirectory:')
            print('\n'.join([ '    '+name for name in selectedOutputNames ]))
        return selectedOutputNames

    def __repr__(self):
        return str(self.Ntimes) + ' time subdirectories located in ' + self.dataDir

    def __getitem__(self,i):
        return self.filelist[i]

    def __iter__(self):
        return self

    def next(self):
        if self.filelist is None:
            raise StopIteration('Need to set filename before iterating')
        self.lastfile += 1
        if self.lastfile >= self.Ntimes:
            raise StopIteration
        else:
            return self.filelist[self.lastfile]
            

