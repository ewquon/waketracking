import numpy as np
import matplotlib.path as mpath

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
                      xg,yg,Ug,vd,
                      func,
                      Nmin=50):
    """Integrate a specified function within an arbitrary region.

    The area of the enclosed cells is compared to the integrated area
    calculated using Green's Theorem to obtain a correction for the
    discretization error. The resulting integrated quantity is scaled
    by the ratio of the actual area divided by the enclosed cell areas.
    This correction is expected to be negligible if there are "enough"
    cells in the contour region.

    For the mass flux, func = lambda u: u
    For the momentum flux, func = lambda u: u**2
    The contour area can be referenced as func = lambda u,A: ...

    Parameters
    ----------
    contourPts : path 
        Output from matplotlib._cntr.Cntr object's trace function
    xg,yg : ndarray
        Sampling plane coordinates, in the rotor-aligned frame.
    Ug : ndarray
        Instantaneous velocity, including shear, used as the
        independent variable in the specificed function.
    vd : ndarray
        Velocity deficit, i.e., Ug with shear removed
    func : (lambda) function
        Function over which to integrate.
    Nmin : integer, optional
        Minimum number of interior cells to compute; if the contour
        region is too small, skip the contour for efficiency.

    Returns
    -------
    integ : float
        Summation of the specified function values in the enclosed
        region, with correction applied.
    vdavg : float
        Average velocity deficit in the contour region.
    corr : float
        Scaling factor to correct for discrete integration error.
    """
    x = xg.ravel()
    y = yg.ravel()
    gridPts = np.vstack((x,y)).transpose()
    path = mpath.Path(contourPts)
    A = calcContourArea(contourPts)

    inner = path.contains_points(gridPts)  # <-- most of the processing time is here!
    Uinner = Ug.ravel()[inner]
    Ninner = len(Uinner)
    if Ninner < Nmin:
        return None,None,None

    # evaluate specified function
    if fn.func_code.co_argcount==1:
        fvals = fn(Uinner)
    elif fn.func_code.co_argcount==2: # assume second argument is A
        fvals = fn(Uinner, A)
    else:
        print 'Problem with function formulation!'
        return None,None,None

    vdavg = np.mean(vd.ravel()[inner])

    # correct for errors in area
    cellFaceArea = (xg[1,0]-xg[0,0])*(yg[0,1]-yg[0,0])
    corr = A / (Ninner*cellFaceArea)

    integ = corr * np.sum(fvals)*cellFaceArea
    
    return integ, vdavg, corr

def calcWeightedCenter(contourPts,
                       xg,yg,fg,
                       weightingFunc=np.abs):
    """Calculated the velocity-weighted center given an arbitrary path.
    The center is weighted by a specified weighting function (the abs
    function by default) applied to specified field values (e.g.
    velocity). The weighting function should have relatively large
    values in the enclosed wake region.

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

