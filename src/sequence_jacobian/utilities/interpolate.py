"""Efficient linear interpolation exploiting monotonicity.

    Interpolates increasing query points xq against increasing data points x.

    - interpolate_y: (x, xq, y) -> yq
        get interpolated values of yq at xq

    - interpolate_coord: (x, xq) -> (xqi, xqpi)
        get representation xqi, xqpi of xq interpolated against x
        xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    - apply_coord: (xqi, xqpi, y) -> yq
        use representation xqi, xqpi to get yq at xq
        yq = xqpi * y[xqi] + (1-xqpi) * y[xqi+1]

    Composing interpolate_coord and apply_coord gives interpolate_y.

    All three functions are written for vectors but can be broadcast to other dimensions
    since we use Numba's guvectorize decorator. In these cases, interpolation is always
    done on the final dimension.
"""

import numpy as np
from numba import njit, guvectorize


@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi


@guvectorize(['void(int64[:], float64[:], float64[:], float64[:])',
              'void(uint32[:], float64[:], float64[:], float64[:])'], '(nq),(nq),(n)->(nq)')
def apply_coord(x_i, x_pi, y, yq):
    """Use representation xqi, xqpi to get yq at xq:
    yq = xqpi * y[xqi] + (1-xqpi) * y[xqi+1]

    Parameters
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nq = x_i.shape[0]
    for iq in range(nq):
        y_low = y[x_i[iq]]
        y_high = y[x_i[iq]+1]
        yq[iq] = x_pi[iq]*y_low + (1-x_pi[iq])*y_high


'''Part 2: More robust linear interpolation that does not require monotonicity in query points.

    Intended for general use in interpolating policy rules that we cannot be sure are monotonic.
    Only get xqi, xqpi representation, for case where x is one-dimensional, in this application.
'''


def interpolate_coord_robust(x, xq, check_increasing=False):
    """Linear interpolation exploiting monotonicity only in data x, not in query points xq.
    Simple binary search, less efficient but more robust.
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Main application intended to be universally-valid interpolation of policy rules.
    Dimension k is optional.

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (k, nq), query points (in any order)

    Returns
    ----------
    xqi  : array (k, nq), indices of lower bracketing gridpoints
    xqpi : array (k, nq), weights on lower bracketing gridpoints
    """
    if x.ndim != 1:
        raise ValueError('Data input to interpolate_coord_robust must have exactly one dimension')

    if check_increasing and np.any(x[:-1] >= x[1:]):
        raise ValueError('Data input to interpolate_coord_robust must be strictly increasing')

    if xq.ndim == 1:
        return interpolate_coord_robust_vector(x, xq)
    else:
        i, pi = interpolate_coord_robust_vector(x, xq.ravel())
        return i.reshape(xq.shape), pi.reshape(xq.shape)


@njit
def interpolate_coord_robust_vector(x, xq):
    """Does interpolate_coord_robust where xq must be a vector, more general function is wrapper"""

    n = len(x)
    nq = len(xq)
    xqi = np.empty(nq, dtype=np.uint32)
    xqpi = np.empty(nq)

    for iq in range(nq):
        if xq[iq] < x[0]:
            ilow = 0
        elif xq[iq] > x[-2]:
            ilow = n-2
        else:
            # start binary search
            # should end with ilow and ihigh exactly 1 apart, bracketing variable
            ihigh = n-1
            ilow = 0
            while ihigh - ilow > 1:
                imid = (ihigh + ilow) // 2
                if xq[iq] > x[imid]:
                    ilow = imid
                else:
                    ihigh = imid

        xqi[iq] = ilow
        xqpi[iq] = (x[ilow+1] - xq[iq]) / (x[ilow+1] - x[ilow])

    return xqi, xqpi
