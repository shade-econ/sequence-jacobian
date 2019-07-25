import numpy as np
from numpy.fft import fftn, rfftn
import jacobian
from numba import njit


def winding_criterion(A, N=4096):
    """Build path of det A(lambda) and obtain its winding number, implementing winding number
    criterion for determinacy that generalizes Onatski (2006).

    Parameters
    ----------
    A : array ((2T-1)*k*k)
            asymptotic H_U matrix, where A[t,i,j] gives Jacobian of target i vs. unknown j
            at t-(T-1) above the main diagonal
    N : [optional] int
            number of equispaced points lambda on interval [0,2pi] for evaluating det A(lambda)

    Returns
    ----------
    winding_number : int
            winding number that characterizes existence and uniqueness of solutions:
                0 for determinate solution
                -1 (or lower) for indeterminacy 
                1 (or higher) for no solution
    """
    det_Alambda = detA_path(A, N)
    return winding_number(det_Alambda.real, det_Alambda.imag)


def detA_path(A, N=4096):
    """Evaluates det A(lambda) at N equispaced points lambda on interval [0,2pi].

    A brief derivation of how this function uses FFT to rapidly evaluate det A(lambda) follows.

    We have, letting A_(-j) denote the k*k matrix A[-j,:,:]:

        det A(lambda) = det sum_(j=-(T-1))^(T-1) A_(-j)e^(i*j*lambda)
    
    which, flipping the order and realigning j, can be rewritten as

        e^(lambda*i*k*(T-1)) det sum_(j=0)^(2T-2) A_(-j+(T-1))e^(-i*j*lambda)   (***)

    Taking the sum in (***) for the values lambda=0,2*pi/N,...,2*pi*(N-1)/N, assuming N >= (2T-1),
    is just taking the discrete Fourier transform of the sequence A_(T-1),...,A_(-(T-1)),0,...,0
    right-padded with zeros to length N.
    
    Hence we can rapidly, simultaneously evaluate (***) at all points lambda equispaced from lambda=0
    to lambda=2*pi using the FFT. This is implemented below, with additional efficiency from fact that
    A(lambda) and A(2*pi-lambda) are conjugate.
    """
    # preliminary: assume and verify shape 2*T-1, k, k for A
    T = (A.shape[0]+1) // 2
    k = A.shape[1]
    if not (T == (A.shape[0]+1)/2 and N >= 2*T-1 and k == A.shape[2]):
        raise ValueError(f'Asymptotic A matrix has improper shape {A.shape}')

    # step 1: use FFT to calculate A(lambda) for each lambda = 2*pi*{0, 1/N, ..., 1/2} (last if N even)
    # note that we need to reverse order of A_t to get sequence A_(T-1),...,A_(-(T-1)),0,...,0
    Alambda = rfftn(A[::-1,...], axes=(0,), s=(N,))

    # step 2: take determinant of each, then multiply by e^(i*k*(T-1)*lambda) to get (***)
    det_Alambda = np.empty(N+1, dtype=np.complex128)
    det_Alambda[:N//2+1] = np.linalg.det(Alambda)*np.exp(2j*np.pi*k*(T-1)/N*np.arange(N//2+1))
    
    # step 3: use conjugate symmetry to fill in rest
    det_Alambda[N//2+1:] = det_Alambda[:(N+1)//2][::-1].conj()

    return det_Alambda


@njit
def winding_number(x, y):
    """Compute winding number around origin of (x,y) coordinates that make closed path by
    counting number of counterclockwise crossings of ray from (0,0) -> (infty,0) on x axis"""
    # ensure closed path!
    assert x[-1] == x[0] and y[-1] == y[0]

    winding_number = 0

    # we iterate through coordinates (x[i], y[i]), where cur_sign is flag for
    # whether current coordinate is above the x axis
    cur_sign = (y[0] >= 0)
    for i in range(1, len(x)):
        if (y[i] >= 0) != cur_sign:
            # if we're here, this means the x axis has been crossed
            # this generally happens rarely, so efficiency no biggie
            cur_sign = (y[i] >= 0)
            
            # crossing of x axis implies possible crossing of ray (0,0) -> (infty,0)
            # we will evaluate three possible cases to see if this is indeed the case
            if x[i] > 0 and x[i-1] > 0:
                # case 1: both (x[i-1],y[i-1]) and (x[i],y[i]) on right half-plane, definite crossing
                # increment winding number if counterclockwise (negative to positive y)
                # decrement winding number if clockwise (positive to negative y)
                winding_number += 2*cur_sign-1
            elif not (x[i] <= 0 and x[i-1] <= 0):
                # here we've ruled out case 2: both (x[i-1],y[i-1]) and (x[i],y[i]) in left 
                # half-plane, where there is definitely no crossing

                # thus we're in ambiguous case 3, where points (x[i-1],y[i-1]) and (x[i],y[i]) in
                # different half-planes: here we must analytically check whether we crossed
                # x-axis to the right or the left of the origin
                # [this step is intended to be rare]
                cross_coord = (x[i-1]*y[i] - x[i]*y[i-1])/(y[i]-y[i-1])
                if cross_coord > 0:
                    winding_number += 2*cur_sign-1
    return winding_number


