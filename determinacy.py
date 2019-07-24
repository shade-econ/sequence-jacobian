import numpy as np
from numpy.fft import fftn, rfftn
import jacobian
from numba import njit


def detA_path(A, N=4096):
    """Same as detA_path_simple but uses conjugate symmetry across real axis to halve work"""
    # preliminary: assume and verify shape 2*T-1, k, k for A
    T = (A.shape[0]+1) // 2
    k = A.shape[1]
    assert T == (A.shape[0]+1)/2 and N >= 2*T-1 and k == A.shape[2]

    # step 1: use FFT to calculate A(lambda) for each lambda = 2*pi*{0, 1/N, ..., 1/2} (last if N even)
    Alambda = rfftn(A[::-1,...], axes=(0,), s=(N,))

    # step 2: take determinant of each and adjust for T-1 displacement
    det_Alambda = np.empty(N+1, dtype=np.complex128)
    det_Alambda[:N//2+1] = np.linalg.det(Alambda)*np.exp(2j*np.pi*k*(T-1)/N*np.arange(N//2+1))
    
    # step 3: use conjugate symmetry to fill in rest (only not duplicating 1/2 if N even)
    det_Alambda[N//2+1:] = det_Alambda[:(N+1)//2][::-1].conj()

    return det_Alambda


@njit
def winding_number(x, y):
    """compute winding number of (x,y) coordinates that make closed path by counting
    number of counterclockwise crossings of ray from (0,0) -> (infty,0) on x axis"""
    # ensure closed path!
    assert x[-1] == x[0] and y[-1] == y[0]

    winding_number = 0
    cur_sign = (y[0] >= 0)
    for i in range(1, len(x)):
        if (y[i] >= 0) != cur_sign:
            # we only enter this if statement rarely (when x axis crossed)
            # so efficiency no biggie
            cur_sign = (y[i] >= 0)
            
            # possible crossing, let's test the three cases
            if x[i] > 0 and x[i-1] > 0:
                # entirely on right half-plane, definite crossing
                winding_number += 2*cur_sign-1
            elif not (x[i] <= 0 and x[i-1] <= 0):
                # if not entirely on left half-plane, ambiguous, must check criterion
                # this step is intended to be rare
                cross_coord = (x[i-1]*y[i] - x[i]*y[i-1])/(y[i]-y[i-1])
                if cross_coord > 0:
                    winding_number += 2*cur_sign-1
    return winding_number


def winding_criterion(A, N=4096):
    """Build path of det A(lambda) and obtain its winding number, implementing extended
    Onatski criterion for determinacy: 0 for determinate solution, -1 (or lower)
    for indeterminacy, 1 (or higher) for no solution"""
    det_Alambda = detA_path(A, N)
    return winding_number(det_Alambda.real, det_Alambda.imag)
