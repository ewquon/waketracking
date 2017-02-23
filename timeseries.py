import os

class TimeSeries(object):
    """ Object for holding general time series data which may be stored
    in multiple time subdirectories
    """

    def __init__(self,datadir='.',filename=None):
        """ Collect data from subdirectories, assuming that subdirs
        have a name that can be cast as a float
        """
        self.dataDir = os.path.abspath(datadir)
        self.filename = filename
        self.outputTimes = []
        self.dirList = []
        self.fileList = []
        self.lastFile = -1 # for iterator

        # process all subdirectories
        dirs = [x[0] for x in os.walk(self.dataDir)]
        for path in dirs:
            if path == self.dataDir: continue
            d = os.path.split(path)[-1]
            try:
                tval = float(d)
            except ValueError:
                continue
            if filename:
                fpath = os.path.join(path,filename)
                if not os.path.isfile(fpath):
                    continue
                self.fileList.append(fpath)
            self.outputTimes.append(tval)
            self.dirList.append(path)
            self.Ntimes = len(self.dirList)
        if filename:
            assert(self.Ntimes == len(self.fileList))
    
        # sort by output time
        iorder = [kv[0] for kv in sorted(enumerate(self.outputTimes),key=lambda x:x[1])]
        self.outputTimes = [self.outputTimes[i] for i in iorder]
        self.dirList = [self.dirList[i] for i in iorder]
        if filename:
            self.fileList = [self.fileList[i] for i in iorder]
        
        # print out a subdirectory listing and check that all subdirectories contain the same files
        if filename is None:
            fileList = os.listdir(self.dirList[0])
            print 'Files in each subdirectory:' # assumed to be identically named
            print ' ','\n  '.join(fileList)
            for d in self.dirList:
                if not os.listdir(d) == fileList:
                    print 'Warning: not all subdirectories contain the same files'
                    break

    def __repr__(self):
        return str(self.Ntimes) + ' time subdirectories located in ' + self.dataDir

    def __getitem__(self,i):
        return self.fileList[i]

    def __iter__(self):
        return self

    def next(self):
        self.lastFile += 1
        if self.lastFile >= self.Ntimes:
            raise StopIteration
        else:
            return self.fileList[self.lastFile]
            

