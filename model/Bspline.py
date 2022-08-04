import numpy as np
import scipy.interpolate as spl

def bsplineBasis(n, k,deg):
    """B-spline type matrix for splines.

    Returns a matrix whose columns are the values of the b-splines of deg
    `deg` as sociated with the knot sequence `knots` evaluated at the points
    `x`.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the b-splines.
    deg : int
        Degree of the splines.
    knots : array_like
        List of knots. The convention here is that the interior knots have
        been extended at both ends by ``deg + 1`` extra knots.

    Returns
    -------
    vander : ndarray
        Vandermonde like matrix of shape (m,n), where ``m = len(x)`` and
        ``m = len(knots) - deg - 1``

    Notes
    -----
    The knots exending the interior points are usually taken to be the same
    as the endpoints of the interval on which the spline will be evaluated.

    """
    knots = np.r_[np.zeros(deg),np.linspace(0,n-1,k),(n-1) * np.ones(deg)]
    x = np.arange(n)
    m = len(knots) - deg - 1
    v = np.zeros((m, len(x)))
    d = np.eye(m, len(knots))
    for i in range(m):
        v[i] = spl.splev(x, (knots, d[i], deg))
    return v.T