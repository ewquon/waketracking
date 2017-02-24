import numpy as np
import matplotlib.path as mpath

def calcContourArea(path):
    """Calculate the area enclosed by an arbitrary path using Green's
    Theorem
    """
    if isinstance( path, np.ndarray ) and len(path.shape)==2:
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

def calcContourFunction(contourPts,xg,yg,Ug,vd,fn):
    """Integrate a specified function within an arbitrary region defined
    by contourPts.
    'vd' is the velocity deficit array

    For the mass flux, fn = lambda u: u
    For the momentum flux, fn = lambda u: u**2
    The contour area can be referenced as fn = lambda u,A: ...
    """
    x = xg.ravel()
    y = yg.ravel()
    gridPts = np.vstack((x,y)).transpose()

    A = calcContourArea(contourPts)

    path = mpath.Path(contourPts)
    inner = path.contains_points(gridPts)  # <-- most of the processing time is here!

    Uinner = Ug.ravel()[inner]
    Ninner = len(Uinner)
    if Ninner < 50:  # minimum number of interior cells; skip if too few for efficiency
        return None,None,None

    # evaluate specified function
    if fn.func_code.co_argcount==1:
        fvals = fn(Uinner)
    elif fn.func_code.co_argcount==2: # assume second argument is A
        fvals = fn(Uinner, A)
    else:
        print 'Problem with lambda function formulation!'
        return None,None,None

    vd_avg = np.mean(vd.ravel()[inner])

    cellFaceArea = (xg[1,0]-xg[0,0])*(yg[0,1]-yg[0,0])

    # correct for errors in area... should be negligible for areas that span more than a few cells
    corr = A / (Ninner*cellFaceArea)
    
    return corr * np.sum( fvals )*cellFaceArea, vd_avg, corr

def calcContourWeightedCenter(contourPts,xg,yg,fg,wfn=None):
    """Calculated the velocity-weighted center given an arbitrary path
    defined by contourPts. Using the function values fg and the
    weighting function wfn, the center is determined from values in grid
    (xg,yg).
    
    By default, the absolute value of the field is used for weighting
    assuming that the function has relatively large (negative) values in
    the enclosed wake region; however, an arbitrary lambda function may
    be specified.
    """# {{{
    x = xg.ravel()
    y = yg.ravel()
    gridPts = np.vstack((x,y)).transpose()

    p = mpath.Path(contourPts)
    inner = p.contains_points(gridPts)
    xin = x[inner]
    yin = y[inner]
    if wfn is None:
        weights = np.abs( fg.ravel()[inner] ) # default
    else:
        weights = wfn( fg.ravel()[inner] )

    denom = np.sum(weights)
    xc = weights.dot(xin) / denom
    yc = weights.dot(yin) / denom
    # }}}
    return xc,yc

