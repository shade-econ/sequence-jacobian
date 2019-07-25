import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import rfft, rfftn, irfft, irfftn
from numba import njit
import jacobian as jac
import determinacy

class AsymptoticTimeInvariant:
    """Represents the asymptotic behavior of infinite matrix that is asymptotically time invariant,
    given by vector v of -(tau-1), ... , 0, ..., tau-1 asymptotic column entries around main diagonal.
    
    Conveniently overloads matrix multiplication operator @, addition operator +, etc., so that we
    can use the same code on these as for ordinary matrices: if A and B are of the ATI class,
    then A @ B is also of the ATI class and gives the asymptotic columns around diagonal of the
    product of matrices whose asymptotic columns are given respectively by A and B."""

    # give higher priority than simple_block.SimpleSparse, which when mixed with ATI is converted
    # to it using .asymptotic_time_invariant property and then handled by methods in this class
    __array_priority__ = 2000

    def __init__(self, v):
        self.v = v

        # v should be -(tau-1), ... , 0, ..., tau-1 asymp column around main diagonal
        self.tau = (len(v)+1) // 2
        assert self.tau*2 - 1 == len(v), f'{len(v)}'

    @property
    def vfft(self):
        """FFT of v padded on the right with 2*tau-1 0s, used for multiplication below"""
        # we could cache this, but so fast it isn't really necessary
        # TODO: maybe it should be cached after all now that we don't need other stuff?
        return rfft(self.v, 4*self.tau-3)

    def changetau(self, tau):
        """Return new with lower or higher tau, trimming or padding with zeros as needed"""
        if tau == self.tau:
            return self
        elif tau < self.tau:
            return AsymptoticTimeInvariant(self.v[self.tau - tau: tau + self.tau - 1])
        else:
            v = np.zeros(2*tau-1)
            v[tau - self.tau: tau + self.tau - 1] = self.v
            return AsymptoticTimeInvariant(v)

    def __getitem__(self, i):
        """Get convenient slice of v, properly centered so -2 maps to entry v_(-2), etc."""
        if isinstance(i, slice):
            return self.v[slice(i.start+self.tau-1, i.stop+self.tau-1, i.step)]
        else:
            return self.v[i+self.tau-1]

    @property
    def T(self):
        """Transpose"""
        return AsymptoticTimeInvariant(self.v[::-1])

    def __pos__(self):
        return self

    def __neg__(self):
        return AsymptoticTimeInvariant(-self.v)

    def __matmul__(self, other):
        """If the vectors v and w represent the asymptotic diagonals of ATI matrices, their
        the product of the matrices is ATI, with asymptotic diagonals represented by vector x
        that is *convolution* of v and w:
        
        x[i] = sum_(j=-infty)^infty v[j]*w[i-j]

        If v and w both have nonzero elements with indices -(tau-1),...,(tau-1), then x[i]
        will be nonzero for indices -(2*tau-2),...,(2*tau-2).
        
        We could obtain this full vector x using, e.g., np.convolve(v, w).
        
        When tau is large it is more efficient, however, to use the FFT:
        irfft(rfft(v, 4*tau-3), rfft(w, 4*tau-3), 4*tau-3) is identical to np.convolve(v, w).

        By convention, to prevent exploding dimensionality, we then return the middle
        -(tau-1), ..., (tau-1) elements of the convolution, dropping the extra (tau-1) on each side.
        """
        if isinstance(other, AsymptoticTimeInvariant):
            # make sure the two arguments have equal tau by enlarging the smaller
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)

            # convolve using FFT, then drop first and last (tau-1) entries
            return AsymptoticTimeInvariant(irfft(newself.vfft*other.vfft, 4*newself.tau-3)[newself.tau-1:-(newself.tau-1)])
        elif hasattr(other, 'asymptotic_time_invariant'):
            # if one of the arguments can be converted to ATI (for now, just SimpleSparse)
            # do so and then take product
            return self @ other.asymptotic_time_invariant
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        return self @ other

    def __add__(self, other):
        if isinstance(other, AsymptoticTimeInvariant):
            # make sure the two arguments have equal tau (same as matmul)
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)

            # now just add the corresponding vectors v
            return AsymptoticTimeInvariant(newself.v + other.v)
        elif hasattr(other, 'asymptotic_time_invariant'):
            # convert non-ATI argument to ATI if possible (same as matmul)
            return self + other.asymptotic_time_invariant
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return AsymptoticTimeInvariant(a*self.v)

    def __rmul__(self, a):
        return self * a

    def __repr__(self):
        return f'AsymptoticTimeInvariant({self.v!r})'

    def __eq__(self, other):
        return np.array_equal(self.v, other.v) if isinstance(other, AsymptoticTimeInvariant) else False
    

def invert_jacdict(jacdict, unknowns, targets, tau, test_invertible=False):
    """Given a nested dict of ATI Jacobians that maps unknowns -> targets, e.g. an asymptotic
    H_U matrix, get the inverse H_U^(-1) as a nested dict.

    This is implemented by inverting the FFT-based multiplication that was implemented above
    for ATI, making use of the linearity of the FFT:
        - We take the FFT of each ATI Jacobian, padded out to 4*tau-3 as above
            (This is done by first packing all Jacobians into a single array A)
        - Then, we take the FFT of the identity, centered aroun d2*tau-1 since
            we intend it to be the result of a product
        - We solve frequency-by-frequency, i.e. for each of 4*tau-3 omegas we solve a k*k
            linear system to get A_rfft[omega,...]^(-1)*id_rfft[omega,...]
        - We take the inverse FFT of the results, then take only the first 2*tau-1 elements
            to get (approximate) inverse Jacobians with times -(tau-1),...,(tau-1), same as
            original Jacobians
        - We unpack these to get a nested dict of ATI Jacobians that inverts original 'jacdict'

    Parameters
    ----------
    jacdict  : dict of dict, ATI (or convertible to ATI) Jacobians where jacdict[t][u] gives
                    asymptotic mapping from unknowns u to targets t in H_U
    unknowns : list, names of unknowns in H_U
    targets  : list, names of targets in H_U
    tau      : int, convert all ATI Jacobians to size tau and provide inverse in size tau
    test_invertible : [optional] bool, use winding number criterion to test whether we should
                    really be inverting this system (i.e. whether determinate solution)

    Returns
    -------
    inv_jacdict : dict of dict, ATI Jacobians where inv_jacdict[u][t] gives asymptotic mapping
                    from targets t to unknowns u in H_U^(-1)
    """

    k = len(unknowns)
    assert k == len(targets)

    # stack the k^2 Jacobians relating unknowns to targets into an A matrix
    A = jac.pack_asymptotic_jacobians(jacdict, unknowns, targets, tau)

    if test_invertible:
        # use winding number criterion to test invertibility
        if determinacy.winding_criterion(A, N=4096) != 0:
            raise ValueError('Trying to invert asymptotic time invariant system of Jacobians' + 
                             ' but winding number test says that it is not uniquely invertible!')

    # take FFT of first dimension (time) of A (i.e. take FFT separtely of all k^2 Jacobians)
    A_rfft = rfftn(A, s=(4*tau-3,), axes=(0,))
    
    # take FFT of identity operator (for efficiency, reuse smaller calc)
    id_vec_rfft = rfft(np.arange(4*tau-3)==(2*tau-2))
    id_rfft = np.zeros((2*tau-1, k, k), dtype=np.complex128)
    for i in range(k):
        id_rfft[:, i, i] = id_vec_rfft
    
    # now solve the linear system to invert A frequency-by-frequency
    # (since frequency is leading dimension, np.linalg.solve automatically does this)
    A_rfft_inv = np.linalg.solve(A_rfft, id_rfft)

    # take inverse FFT of this to get full A
    # then take first 2*tau-1 entries to get approximate A from -(tau-1),...,0,...,(tau-1)
    A_inv = irfftn(A_rfft_inv, s=(4*tau-3,), axes=(0,))[:2*tau-1, :, :]

    # unstack this
    return jac.unpack_asymptotic_jacobians(A_inv, targets, unknowns, tau)
