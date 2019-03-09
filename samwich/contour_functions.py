from __future__ import print_function
import numpy as np
import matplotlib.path as mpath
try:
    import cv2
except ImportError:
    print('Note: Install OpenCV library (cv2) to enable contour trackers, '
          'e.g., `conda install opencv`')
    cv2 = None

class Contours(object):
    """Interface with OpenCV
    
    We used to use the matplotlib._cntr.Cntr class which has been
    deprecated in Python 3. Instead, we'll use OpenCV to create and
    process contours. This involves converting the planar data into
    a grayscale image.
    """
    def __init__(self,x,y,u):
        """Setup an image for manipulation with OpenCV
        x,y,u should be uniformly spaced 2-D arrays
        """
        assert(x.shape == y.shape == u.shape)
        self.Nx, self.Ny = x.shape
        self.x = x
        self.y = y
        self.u = u
        self.u0 = np.min(self.u)
        self.urange = np.max(self.u) - self.u0
        # create single-channel grayscale image, values ranging from 0..255
        self.img = np.array((u-self.u0)/self.urange * 255, dtype=np.uint8)

    def _value_to_uint8(self,val):
        """Convert value to grayscale integer value"""
        return np.uint8((val - self.u0)/self.urange * 255)

    def to_coords(self,path,closed=False,array=False):
        """Convert contour path to coordinates

        If closed is true, first and last point are coincident to form
        a closed loop (e.g., for plotting)
        
        If array is true, return 2-D array with shape (N,2)
        Otherwise, return x and y components
        """
        x = [ self.x[j,i] for i,j in zip(path[:,0,0],path[:,0,1]) ]
        y = [ self.y[j,i] for i,j in zip(path[:,0,0],path[:,0,1]) ]
        if closed:
            if (not x[-1] == x[0]) or (not y[-1] == y[0]):
                x.append(x[0])
                y.append(y[0])
        if array:
            return np.stack((x,y)).T
        else:
            return x,y

    def to_indices(self,coords):
        """Convert coordinates to image points/indices

        NOTE: this assumes that the sampling plane lies on a Cartesian
        grid
        """
        ix = np.argmin(np.abs(coords[0] - self.x[:,0]))
        iy = np.argmin(np.abs(coords[1] - self.y[0,:]))
        return ix, iy

    def _points_in_contour(self,contour):
        inside = np.zeros(self.x.shape)
        for i in range(self.Nx):
            for j in range(self.Ny):
                inside[i,j] = cv2.pointPolygonTest(contour, (j,i),
                                                   measureDist=False)
        return inside

    def _get_all_contours(self, thresh):
        """Get all contours using cv2.findContours()
        API ref: https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#cv2.findContours
        Contour hierarchy ref: https://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html

        Returns list of uint8 image point indices
        """
        # CV_THRESH_BINARY: maxval if src(x,y) > thresh; 0 otherwise
        _, img_bw = cv2.threshold(self.img,thresh=thresh,maxval=255,
                                  type=cv2.THRESH_BINARY)

        # CV_RETR_TREE: retrieves all contours and reconstructs a full
        #   hierarchy of nested contours
        # CV_RETR_CCOMP: retrieves all of the contours and organizes
        #   them into a two-level hierarchy; at the top level, there are
        #   external boundaries of the components. At the second level,
        #   there are boundaries of the holes. If there is another
        #   contour inside a hole of a connected component, it is still
        #   put at the top level.
        # CV_CHAIN_APPROX_SIMPLE: compresses horizontal, vertical,
        #   diagonal segments
        # Note that this modifies the source img_bw
        if cv2.__version__.startswith('3'):
            _, contour_uint8_list, hierarchy = cv2.findContours(
                    img_bw,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
        else:
            contour_uint8_list, hierarchy = cv2.findContours(
                    img_bw,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)

        # hierarchy has shape (1,Ncontours,4) -- expected (Ncontours,4)
        # TODO: may be possible to deduce open contours from the hierarchy
        #     e.g. https://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv
        #   h[0] : next contour at same hierarchical level
        #   h[1] : previous contour at the same hierarchical level
        #   h[2] : first child contour
        #   h[3] : parent contour

        # convert list of points in uint8 space to actual coordinates
        contour_list = []
        is_closed = []
        for ptsvec in contour_uint8_list:
            # ptsvec is a "vector" (from C++) of points representing a single
            # contour, with shape (Npts,1,2)
#            contx, conty = self.to_coords(ptsvec)
            # check if any points are on the boundary; if so, then assume that
            # those contours are open
            if np.any(ptsvec[:,0,0] == 0) or np.any(ptsvec[:,0,1] == 0) or \
                    np.any(ptsvec[:,0,0] == self.Nx) or \
                    np.any(ptsvec[:,0,1] == self.Ny):
                is_closed.append(False)
            else:
                #contx.append(contx[0])
                #conty.append(conty[0])
                is_closed.append(True)
            # coordinates have shape (Npts,2)
#            contx = np.array(contx)
#            conty = np.array(conty)
#            contour_list.append(np.stack((contx,conty)).T)

#        return contour_list, is_closed
        return contour_uint8_list, is_closed

    def get_closed_paths(self, Clevel,
                         close_paths=False,
                         min_points=None,
                         verbose=False):
        """Process all contour paths, returning closed paths only

        NOTE: Closed contour paths in CV2 do not have an overlapped
        start/end point.

        Parameters
        ----------
        Clevel : float
            Contour level for which to identify paths
        close_paths : optional
            If False, open contours will be ignored (determined by
            whether or not any contour points lie on the boundary); if
            True, the contour will be treated as follows:
            - first/last points lie on the same edge and are assumed to
              be connected; no action necessary
            - first/last points lie on different edges, and corner
              points are added to create two new contours
        min_points : int, optional
            Minimum number of points a closed loop must contain for it
            to be considered a candidate path. This is indirectly
            related to the smallest allowable contour region. Set to
            None to return all contours.

        Returns
        -------
        path_list : list
            List of closed contour paths (points are indices)
        """
        path_list = []
        Clevel_uint8 = self._value_to_uint8(Clevel)
        all_contours, is_closed = self._get_all_contours(Clevel_uint8)
        if min_points is None:
            min_points = 1
        for path,closed in zip(all_contours,is_closed):
            if closed:
                if len(path) >= min_points:
                    path_list.append(path)
                elif verbose:
                    print('Ignoring contour with {} points'.format(len(path)))
            elif close_paths:
                # need to close open contour
                xstart = path[0,:]
                xend = path[-1,:]
                if (xstart[0] == xend[0]) or (xstart[1] == xend[1]):
                    # simplest case: both ends point on same edge
                    #path = np.vstack((path,xstart))
                    path_list.append(path)
                    if verbose:
                        print('  closed contour (simple)')
                        print('  {} {}'.format(xstart,xend))
                else:
                    # more complex case, need some additional grid information
                    newpath1 = np.copy(path)
                    newpath2 = np.copy(path)
                    xend = np.array(xend)
                    x0 = 0       #np.min(self.x)
                    x1 = self.Nx #np.max(self.x)
                    y0 = 0       #np.min(self.y)
                    y1 = self.Ny #np.max(self.y)
                    assert((xstart[0] == x0 or xstart[0] == x1) or \
                           (xstart[1] == y0 or xstart[1] == y1))
                    assert((xend[0] == x0 or xend[0] == x1) or \
                           (xend[1] == y0 or xend[1] == y1))
                    TL = [x0,y1]
                    TR = [x1,y1]
                    BR = [x1,y0]
                    BL = [x0,y0]
                    cornersCW = np.array([TL,TR,BR,BL])
                    cornersCCW = cornersCW[::-1,:]
                    # - find nearest corner
                    if xend[0] == x0: # left edge
                        start1,start2 = 0,0
                    elif xend[0] == x1:  # right edge
                        start1,start2 = 2,2
                    elif xend[1] == y0: # bottom edge
                        start1,start2 = 3,1
                    elif xend[1] == y1: # top edge
                        start1,start2 = 1,3
                    if verbose:
                        print('contour ends at {}'.format(xend))
                        print('next CW pt {}'.format(cornersCW[start1,:]))
                        print('next CCW pt {}'.format(cornersCCW[start2,:]))
                    # - reorder list of corners
                    cornersCW  = np.vstack((cornersCW[start1:,:],
                                            cornersCW[:start1,:]))
                    cornersCCW = np.vstack((cornersCCW[start2:,:],
                                            cornersCCW[:start2,:]))
                    # - add points until we get to the same edge as the start
                    #   point if we get an IndexError on corners*[ipop,:], then
                    #   we have a problem with the logic; we should be adding at
                    #   most 3 points, never all 4 corners...
                    def same_edge(pt1,pt2):
                        return (pt1[0] == pt2[0]) or (pt1[1] == pt2[1])
                    if verbose: print('creating CW loop')
                    ipop = 0
                    while not same_edge(newpath1[-1,:],xstart):
                        if verbose:
                            print('{} not on same edge as {}'.format(
                                    newpath1[-1,:],xstart))
                        newpath1 = np.vstack((newpath1,cornersCW[ipop,:]))
                        ipop += 1
                    if verbose: print('creating CCW loop')
                    ipop = 0
                    while not same_edge(newpath2[-1,:],xstart):
                        if verbose:
                            print('{} not on same edge as {}'.format(
                                    newpath2[-1,:],xstart))
                        newpath2 = np.vstack((newpath2,cornersCCW[ipop,:]))
                        ipop += 1
                    # - should have a closed loop now
                    #newpath1 = np.vstack((newpath1,xstart))
                    #newpath2 = np.vstack((newpath2,xstart))
                    path_list.append(newpath1)
                    path_list.append(newpath2)
                    if verbose:
                        print('  closed contour (compound)')
                        print('  {} {}'.format(xstart,xend))

        return path_list

    def calc_area(self, path):
        """Calculate the area enclosed by an arbitrary path using Green's
        Theorem, assuming that the path is closed.
        """
        xp,yp = self.to_coords(path,closed=True)
        dx = np.diff(xp)
        dy = np.diff(yp)
        return 0.5*np.abs(np.sum(yp[:-1]*dx - xp[:-1]*dy))

    def integrate_function(self, path, func, fields, vd=None, Nmin=50):
        """Integrate a specified function within an arbitrary region. This
        is a function of f(x,y) and optionally the contour area.
    
        The area of the enclosed cells is compared to the integrated area
        calculated using Green's Theorem to obtain a correction for the
        discretization error. The resulting integrated quantity is scaled
        by the ratio of the actual area divided by the enclosed cell areas.
        This correction is expected to be negligible if there are "enough"
        cells in the contour region.
    
        Parameters
        ----------
        path : ndarray 
            Contour identified by cv2.findContours().
        func : (lambda) function
            Function over which to integrate.
        fields : list-like of ndarray
            Fields to be used as the independent variable in the specified
            function.
        vd : ndarray, optional
            Velocity deficit; if not None, returns average deficit in the
            enclosed region.
        Nmin : integer, optional
            Minimum number of interior cells to compute; if the contour
            region is too small, skip the contour for efficiency.
    
        Returns
        -------
        fval : float
            Summation of the specified function values in the enclosed
            region, with correction applied. Returns None if the path
            encloses less points than Nmin.
        corr : float
            Scaling factor to correct for discrete integration error.
        vdavg : float
            Average velocity deficit in the contour region.
        """
        A = self.calc_area(path)
    
        in_on_out = self._points_in_contour(path) # >0, 0, <0
        inner = (in_on_out >= 0).ravel()
        Ninner = np.count_nonzero(inner)
        if Ninner < Nmin:
            # contour path is too short
            return None,None
    
        # calculate average velocity deficit if needed (e.g., for more
        # rigorous identification of wake regions)
        if vd is not None:
            vdavg = np.mean(vd.ravel()[inner])
        else:
            vdavg = None
    
        # if just integrating area, we're done at this point
        if func is None:
            fval = A
        else: 
            # correct for errors in area due to discretization
            # - assumes uniform grid spacing
            cell_face_area = (self.x[1,0]-self.x[0,0])*(self.y[0,1]-self.y[0,0])
            corr = A / (Ninner*cell_face_area)
            # evaluate specified function
            func_args = [ field.ravel()[inner] for field in fields ]
            fvals_in_contour = func(*func_args)
            fval = corr * np.sum(fvals_in_contour)*cell_face_area
    
        return fval, vdavg

    def calc_weighted_center(self, path, weighting_function=np.abs):
        """Calculated the velocity-weighted center given an arbitrary path.
        The center is weighted by a specified weighting function (the abs
        function by default) applied to specified field values (e.g.
        velocity). The weighting function should have relatively large
        values in the enclosed wake region.

        Parameters
        ----------
        path : ndarray 
            Output from cv2.findContours().
        weighting_function : (lambda) function
            Univariate weighting function.

        Returns
        -------
        xc,yc : ndarray
            Coordinates of the weighted wake center in the rotor-aligned
            frame.
        """
        in_on_out = self._points_in_contour(path) # >0, 0, <0
        inner = (in_on_out >= 0).ravel()
        xin = self.x.ravel()[inner]
        yin = self.y.ravel()[inner]
        weights = weighting_function(self.u.ravel()[inner])
        denom = np.sum(weights)

        xc = weights.dot(xin) / denom
        yc = weights.dot(yin) / denom
        
        return xc,yc


def resample_outline(outline,origin=(0,0),N=25):
    """First attempt to smooth an arbitrary contour path using
    resampling and cubic spline interpolation. This does _not_ work
    well for smoothing. 
    """
    from scipy.interpolate import interp1d
    x = outline[:,0] - origin[0]
    y = outline[:,1] - origin[1]
    # convert to polar coordinates and sort (for interp1d)
    r0 = np.sqrt(x**2 + y**2)
    theta0 = np.arctan2(y,x)
    isort = np.argsort(theta0)
    r0 = r0[isort]
    theta0 = theta0[isort]
    theta = np.unique(theta0)
    # radially average to get rid of multiple radial values at the same azimuth
    r = [ np.mean(r0[np.where(theta0 == the)]) for the in theta ]
    assert(len(r) == len(theta))
    # interpolate for evenly spaced azimuth angles
    interpfun = interp1d(theta,r,kind='cubic')
    theta = np.linspace(np.min(theta),np.max(theta),N) #np.linspace(-np.pi,np.pi,N)
    r = interpfun(theta)
    # recover coordinates
    x = r*np.cos(theta) + origin[0]
    y = r*np.sin(theta) + origin[1]
    return np.stack((x,y),axis=-1)

def smooth_outline(outline,origin=(0,0),window=3):
    """Perform moving average in polar coordinates"""
    # convert to polar coordinates
    x = outline[:-1,0] - origin[0] # assume last point == first point
    y = outline[:-1,1] - origin[1]
    r0 = np.sqrt(x**2 + y**2)
    the0 = np.arctan2(y,x)
    # add points (since our data is cyclic) to handle convolution edge case
    i = int(window/2)
    r = np.concatenate((r0[-i:],r0,r0[:i]))
    the = np.concatenate((the0[-i:],the0,the0[:i]))
    # now, perform the moving average
    w = np.ones(window)
    w /= w.sum()
    r = np.convolve(w,r,mode='valid')
    assert(len(r)==len(r0))
    # recover coordinates
    x = r*np.cos(the0) + origin[0]
    y = r*np.sin(the0) + origin[1]
    return np.stack((x,y),axis=-1)
