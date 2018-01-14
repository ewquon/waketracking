from __future__ import print_function
import numpy as np
import matplotlib.path as mpath

def get_paths(Cdata, Clevel,
              close_paths=False,
              min_points=50,
              verbose=False):
    """Loops over paths identified by the trace function.

    Parameters
    ----------
    Cdata : contour object
        Instance of matplotlib._cntr.Cntr
    Clevel : float
        Contour level for which to identify paths
    close_paths : tuple, optional
        If None, open contours will be ignored; if True, a simple
        closure will be attempted (assuming that the start and end
        points lie on the same edge); otherwise, specify a tuple with
        (xh_range,xv_range), i.e., the horizontal and vertical range
        (only the first and last elements will be used, so the full list
        of coordinates is not needed)
    min_points : int, optional
        Minimum number of points a closed loop must contain for it to be
        considered a candidate path. This is indirectly related to the 
        smallest allowable contour region.

    Returns
    -------
    path_list : list
        List of closed contour paths
    """
    path_list = []
    for path in Cdata.trace(Clevel):
        # Returns a list of arrays (floats) followed by lists (uint8),
        #   of the contour coordinates and segment descriptors,
        #   respectively
        #NtraceCalls += 1
        if path.dtype=='uint8':
            # done reading paths, don't need the segment connectivity info
            break
        if np.all(path[-1] == path[0]):
            # closed contour
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
                path = np.vstack((path,xstart))
                path_list.append(path)
                if verbose:
                    print('  closed contour (simple)')
                    print('  {} {}'.format(xstart,xend))
            elif isinstance(close_paths, (list,tuple)):
                # more complex case, need some additional grid information
                newpath1 = np.copy(path)
                newpath2 = np.copy(path)
                xend = np.array(xend)
                x0 = close_paths[0][0]
                x1 = close_paths[0][-1]
                y0 = close_paths[1][0]
                y1 = close_paths[1][-1]
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
                #sqdist1 = [ (xend-cornersCW[i,:]).dot(xend-cornersCW[i,:]) for i in range(4) ]
                #sqdist2 = [ (xend-cornersCCW[i,:]).dot(xend-cornersCCW[i,:]) for i in range(4) ]
                #start1 = np.argmin(sqdist1)
                #start2 = np.argmin(sqdist2)
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
                cornersCW  = np.vstack(( cornersCW[start1:,:], cornersCW[:start1,:]))
                cornersCCW = np.vstack((cornersCCW[start2:,:],cornersCCW[:start2,:]))
                # - add points until we get to the same edge as the start point
                #   if we get an IndexError on corners*[ipop,:], then we have a problem with the
                #   logic; we should be adding at most 3 points, never all 4 corners...
                def same_edge(pt1,pt2):
                    return (pt1[0] == pt2[0]) or (pt1[1] == pt2[1])
                if verbose: print('creating CW loop')
                ipop = 0
                while not same_edge(newpath1[-1,:],xstart):
                    if verbose: print('{} not on same edge as {}'.format(newpath1[-1,:],xstart))
                    newpath1 = np.vstack((newpath1,cornersCW[ipop,:]))
                    ipop += 1
                if verbose: print('creating CCW loop')
                ipop = 0
                while not same_edge(newpath2[-1,:],xstart):
                    if verbose: print('{} not on same edge as {}'.format(newpath2[-1,:],xstart))
                    newpath2 = np.vstack((newpath2,cornersCCW[ipop,:]))
                    ipop += 1
                # - should have a closed loop now
                newpath1 = np.vstack((newpath1,xstart))
                newpath2 = np.vstack((newpath2,xstart))
                path_list.append(newpath1)
                path_list.append(newpath2)
                if verbose:
                    print('  closed contour (compound)')
                    print('  {} {}'.format(xstart,xend))
            else:
                # complex closure, but no additional data available
                if verbose:
                    print('  compound closure not performed')
                    print('  {} {}'.format(xstart,xend))
                continue

    return path_list

def calc_area(path):
    """Calculate the area enclosed by an arbitrary path using Green's
    Theorem, assuming that the path is closed.
    """
    if isinstance(path, np.ndarray) and len(path.shape)==2:
        # we have a 2D array
        xp = path[:,0]
        yp = path[:,1]
    else:
        # we have a path instance
        Np = len(path)
        xp = np.zeros(Np)
        yp = np.zeros(Np)
        for i,coords in enumerate(path):
            xp[i] = coords[0]
            yp[i] = coords[1]
    dx = np.diff(xp)
    dy = np.diff(yp)
    return 0.5*np.abs(np.sum(yp[:-1]*dx - xp[:-1]*dy))

def integrate_function(contour_points,
                       func,
                       xg,yg,fields,
                       vd=None,
                       Nmin=50):
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
    contour_points : path 
        Output from matplotlib._cntr.Cntr object's trace function
    xg,yg : ndarray
        Sampling plane coordinates, in the rotor-aligned frame.
    fields : list-like of ndarray
        Fields to be used as the independent variable in the specified
        function.
    vd : ndarray, optional
        Velocity deficit; if not None, returns average deficit in the
        enclosed region.
    func : (lambda) function
        Function over which to integrate.
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
    x = xg.ravel()
    y = yg.ravel()
    gridPts = np.vstack((x,y)).transpose()
    path = mpath.Path(contour_points)
    A = calc_area(contour_points)

    inner = path.contains_points(gridPts)  # <-- most of the processing time is here!
    Ninner = len(np.nonzero(inner)[0])
    if Ninner < Nmin:
        return None,None,None

    if vd is not None:
        vdavg = np.mean(vd.ravel()[inner])
    else:
        vdavg = None

    # correct for errors in area due to discretization
    cell_face_area = (xg[1,0]-xg[0,0])*(yg[0,1]-yg[0,0])
    corr = A / (Ninner*cell_face_area)

    # if integrating area, we're done at this point
    if func is None:
        return A, corr, vdavg

    # evaluate specified function
    func_args = [ field.ravel()[inner] for field in fields ]
    fvals_in_contour = func(*func_args)

    fval = corr * np.sum(fvals_in_contour)*cell_face_area

    return fval, corr, vdavg

def calc_weighted_center(contour_points,
                         xg,yg,fg,
                         weighting_function=np.abs):
    """Calculated the velocity-weighted center given an arbitrary path.
    The center is weighted by a specified weighting function (the abs
    function by default) applied to specified field values (e.g.
    velocity). The weighting function should have relatively large
    values in the enclosed wake region.

    Parameters
    ----------
    contour_points : path 
        Output from matplotlib._cntr.Cntr object's trace function
    xg,yg : ndarray
        Sampling plane coordinates, in the rotor-aligned frame.
    fg : ndarray
        Field function to use for weighting.
    weighting_function : (lambda) function
        Univariate weighting function.

    Returns
    -------
    xc,yc : ndarray
        Coordinates of the weighted wake center in the rotor-aligned
        frame.
    """
    x = xg.ravel()
    y = yg.ravel()
    gridPts = np.vstack((x,y)).transpose()
    path = mpath.Path(contour_points)

    inner = path.contains_points(gridPts)
    xin = x[inner]
    yin = y[inner]
    weights = weighting_function(fg.ravel()[inner])
    denom = np.sum(weights)

    xc = weights.dot(xin) / denom
    yc = weights.dot(yin) / denom
    
    return xc,yc

