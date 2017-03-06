import numpy as np
import matplotlib.path as mpath

def getPaths(Cdata,Clevel,closePaths=False,verbose=True):
    """Loops over paths identified by the trace function.

    Parameters
    ----------
    Cdata : contour object
        Instance of matplotlib._cntr.Cntr
    Clevel : float
        Contour level for which to identify paths
    closePaths : tuple, optional
        If None, open contours will be ignored; if True, a simple
        closure will be attempted (assuming that the start and end
        points lie on the same edge); otherwise, specify a tuple with
        (xh_range,xv_range), i.e., the horizontal and vertical range
        (only the first and last elements will be used, so the full list
        of coordinates is not needed)

    Returns
    -------
    pathList : list
        List of closed contour paths
    """
    pathList = []
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
            pathList.append(path)
        elif closePaths:
            # need to close open contour
            xstart = path[0,:]
            xend = path[-1,:]
            if (xstart[0] == xend[0]) or (xstart[1] == xend[1]):
                # simplest case: both ends point on same edge
                path = np.vstack((path,xstart))
                pathList.append(path)
                if verbose:
                    print '  closed contour (simple)'
                    print ' ',xstart,xend
            elif isinstance(closePaths, (list,tuple)):
                # more complex case, need some additional grid information
                newpath1 = np.copy(path)
                newpath2 = np.copy(path)
                xend = np.array(xend)
                x0 = closePaths[0][0]
                x1 = closePaths[0][-1]
                y0 = closePaths[1][0]
                y1 = closePaths[1][-1]
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
                    print 'contour ends at',xend
                    print 'next CW pt',cornersCW[start1,:]
                    print 'next CCW pt',cornersCCW[start2,:]
                # - reorder list of corners
                cornersCW  = np.vstack(( cornersCW[start1:,:], cornersCW[:start1,:]))
                cornersCCW = np.vstack((cornersCCW[start2:,:],cornersCCW[:start2,:]))
                # - add points until we get to the same edge as the start point
                #   if we get an IndexError on corners*[ipop,:], then we have a problem with the
                #   logic; we should be adding at most 3 points, never all 4 corners...
                def sameEdge(pt1,pt2):
                    return (pt1[0] == pt2[0]) or (pt1[1] == pt2[1])
                if verbose: print 'creating CW loop'
                ipop = 0
                while not sameEdge(newpath1[-1,:],xstart):
                    print newpath1[-1,:],'not on same edge as',xstart
                    newpath1 = np.vstack((newpath1,cornersCW[ipop,:]))
                    ipop += 1
                if verbose: print 'creating CCW loop'
                ipop = 0
                while not sameEdge(newpath2[-1,:],xstart):
                    print newpath2[-1,:],'not on same edge as',xstart
                    newpath2 = np.vstack((newpath2,cornersCCW[ipop,:]))
                    ipop += 1
                # - should have a closed loop now
                newpath1 = np.vstack((newpath1,xstart))
                newpath2 = np.vstack((newpath2,xstart))
                pathList.append(newpath1)
                pathList.append(newpath2)
                if verbose:
                    print '  closed contour (compound)'
                    print ' ',xstart,xend
            else:
                # complex closure, but no additional data available
                if verbose:
                    print '  compound closure not performed'
                    print ' ',xstart,xend
                continue

    return pathList

def calcArea(path):
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

def integrateFunction(contourPts,
                      func,
                      xg,yg,fg,
                      vd=None,
                      Nmin=50):
    """Integrate a specified function within an arbitrary region.

    The area of the enclosed cells is compared to the integrated area
    calculated using Green's Theorem to obtain a correction for the
    discretization error. The resulting integrated quantity is scaled
    by the ratio of the actual area divided by the enclosed cell areas.
    This correction is expected to be negligible if there are "enough"
    cells in the contour region.

    Parameters
    ----------
    contourPts : path 
        Output from matplotlib._cntr.Cntr object's trace function
    xg,yg : ndarray
        Sampling plane coordinates, in the rotor-aligned frame.
    fg : ndarray
        Instantaneous velocity, including shear, used as the
        independent variable in the specificed function.
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
    path = mpath.Path(contourPts)
    A = calcArea(contourPts)

    inner = path.contains_points(gridPts)  # <-- most of the processing time is here!
    Uinner = fg.ravel()[inner]
    Ninner = len(Uinner)
    if Ninner < Nmin:
        return None,None,None

    # evaluate specified function
    if func.func_code.co_argcount==1:
        fvals = func(Uinner)
    elif func.func_code.co_argcount==2: # assume second argument is A
        fvals = func(Uinner, A)
    else:
        print 'Problem with function formulation!'
        return None,None,None

    if vd is not None:
        vdavg = np.mean(vd.ravel()[inner])
    else:
        vdavg = None

    # correct for errors in area
    cellFaceArea = (xg[1,0]-xg[0,0])*(yg[0,1]-yg[0,0])
    corr = A / (Ninner*cellFaceArea)

    fval = corr * np.sum(fvals)*cellFaceArea
    
    return fval, corr, vdavg

def calcWeightedCenter(contourPts,
                       xg,yg,fg,
                       weightingFunc=np.abs):
    """Calculated the velocity-weighted center given an arbitrary path.
    The center is weighted by a specified weighting function (the abs
    function by default) applied to specified field values (e.g.
    velocity). The weighting function should have relatively large
    values in the enclosed wake region.

    Parameters
    ----------
    contourPts : path 
        Output from matplotlib._cntr.Cntr object's trace function
    xg,yg : ndarray
        Sampling plane coordinates, in the rotor-aligned frame.
    fg : ndarray
        Field function to use for weighting.
    weightingFunc : (lambda) function
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
    path = mpath.Path(contourPts)

    inner = path.contains_points(gridPts)
    xin = x[inner]
    yin = y[inner]
    weights = weightingFunc(fg.ravel()[inner])
    denom = np.sum(weights)

    xc = weights.dot(xin) / denom
    yc = weights.dot(yin) / denom
    
    return xc,yc

