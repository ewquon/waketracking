***********
Methodology
***********

Gaussian-Fit Approach
=====================

The Gaussian test function :math:`\phi` is defined as

.. math::
    \phi(y,z) = -u_{max} \exp \left[ -\frac{1}{2}\left( \frac{(\Delta y)^2}{\sigma_y^2} + \frac{(\Delta z)^2}{\sigma_z^2} \right) \right]

where the :math:`\Delta y` and :math:`\Delta z` are the in-plane horizontal and vertical separation relative to location (y,z).
:math:`\sigma_y` and :math:`\sigma_z` are parameters describing the variances associated with the y- and z-directions.

The magnitude :math:`u_{max}` is calculated from the largest absolute value of the x-velocity with the wind shear removed.
The mean velocity profile can either be a specified array, the precursor inflow profile, or calculated on-the-fly as the time-averaged value from the fringes (y=min(y) and max(y)) of all sampled planes.
*The on-the-fly method is not recommended; preliminary studies have shown inconsistency (overprediction) of the mean velocity profile when estimated based on fringe values compared to the reference inflow profile. This overprediction is likely due to naive averaging that doesn't account for the transient flow at early times.*

First, an idealized 2D wake field is calculated based on :math:`u_{max}`, :math:`\sigma_y`, and :math:`\sigma_z`. Then the idealized wake is translated to points in the sampled plane for comparison. This search only considers locations between 25% and 75% of the width and height of the sampling plane. A squared error is calculated between the u-velocity of the ideal wake and the actual sampled wake. The location with the least squared error is considered to be the wake center.

Note that this approach will never "fail"; it will always return the best possible fit, which may or may not be valid.

Specific fit types:
  * "simple": 1-D Gaussian function (:math:`\sigma_y=\sigma_z`, least-squares fit of wake centers only)
  * "elliptic": Rotated 2-D Gaussian (fit of wake centers and the correlation parameter, only makes sense if :math:`\sigma_y\neq\sigma_z`)
  * "general": General 2-D Gaussian (fit of wake centers, correlation parameters, and width parameters)
  * "double": Double Gaussian, assuming both Gaussians have the same orientation (fit of both wake centers, a single correlation parameter, both sets of wake centers, and both function amplitudes)

Least-squares optimization initial guesses and constraints...

.. Possible improvements:

.. * Estimate and/or optimize :math:`\sigma` on the fly.
.. * Increase resolution of test points--i.e, consider test points that are not necessarily coincident with the sampled points.
.. * Use different sigma in y and z, fit elliptical function
.. * Add correlation term within Gaussian exponent, e.g. Trujillo et al., Wind Energy 2011, Eqn. 2

Weighted Geometric-Center Approach 
==================================

The horizontal position of the wake is determined as follows:

.. math::
    y_{wake} = \frac{ \sum_i u_iy_i }{\sum_i u_i}, \qquad \forall u_i < u_{thresh}

where :math:`u_i` is the instantaneous sampled velocity with mean shear removed, and :math:`u_{thresh}` is a user-specified threshold value. The vertical position is determined analogously by replacing :math:`y` with :math:`z`.

*This detection algorithm fails when* :math:`u_{thresh}` *is set too large and typically fails for larger distances from the rotor when the wake deficit has been significantly reduced by turbulent mixing.*
In this case, the set of points identified as being in the wake (:math:`u_i < u_{thresh}`) is zero and the denominator vanishes. An ad hoc threshold value of -3.0 m/s has been used and invalid wake-center coordinates are corrected with ``fixTrajectoryErrors()`` by piecewise linear interpolation. 

Possible improvements:

* Adaptively set/adjust the threshold value.


Contour-Area Approach
=====================

The contours of the wake velocity are calculated and the closed contour with enclosed area closest to the rotor area is considered representative of the wake location. The wake center is then identified as the geometric center, the average of all the points along the representative contour.
Requires specifying the number of test contours to identify.

Possible improvements:

* Use a weighted average (as in the density approach) to account for wake skew. This introduces additional options, e.g. how to use the velocity as a weighting function. By default, the absolute value is used (expected to be perform better for unstable wakes and more robust by avoiding the possibility of summing to zero); a simpler approach may be to negate the instantaneous velocity (with shear removed) to favor large wake velocity deficits.
* Optimize the contour level selection to obtain the closed contour that has exactly the same area as the reference area; this is not likely to have a significant impact on the estimated wake center location.


Contour-Momentum Approach
=========================

The momentum deficit behind the rotor is estimated...

Contours are selected with the same momentum deficit...



