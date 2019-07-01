import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import rfft, rfftn, irfft, irfftn
from numba import njit


class AsymptoticVector:
    """Represents the asymptotic behavior of infinite matrix that is asymptotically Toeplitz, given
    by vector v of -(tau-1), ... , 0, ..., tau-1 asymptotic column entries around main diagonal"""

    def __init__(self, v):
        self.v = v

        # v should be -(tau-1), ... , 0, ..., tau-1 asymp column around main diagonal
        self.tau = (len(v) + 1) // 2
        assert self.tau * 2 - 1 == len(v)

    # note: no longer doing caching of these because FFT so quick that complexity unnecessary
    @property
    def vfft(self):
        return rfft(self.v, 4 * self.tau - 1)

    @property
    def vfft_leftzero(self):
        zeroed = self.v.copy()
        zeroed[:self.tau + 1] = 0
        return rfft(zeroed, 4 * self.tau - 1)

    @property
    def vfft_rightzero(self):
        zeroed = self.v.copy()
        zeroed[self.tau:] = 0
        return rfft(zeroed, 4 * self.tau - 1)

    def make_toeplitz(self, T):
        return AsymptoticToeplitz(np.empty(0, 0), self).changeT(T)

    def changetau(self, tau):
        """return new AsymptoticVector with lower or higher tau, trimming or padding with zeros as needed"""
        if tau == self.tau:
            return self
        elif tau < self.tau:
            return AsymptoticVector(self.v[self.tau - tau: tau + self.tau - 1])
        else:
            v = np.zeros(2 * tau - 1)
            v[tau - self.tau: tau + self.tau - 1] = self.v
            return AsymptoticVector(v)

    @property
    def T(self):
        """Transpose"""
        return AsymptoticVector(self.v[::-1])

    def __pos__(self):
        return self

    def __neg__(self):
        return AsymptoticVector(-self.v)

    def __matmul__(self, other):
        if isinstance(other, AsymptoticVector):
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)
            return irfft(newself.vfft * other.vfft)[newself.tau:-newself.tau]
        elif hasattr(other, 'asymptotic_vector'):
            return self @ other.asymptotic_vector
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        return self @ other

    def __add__(self, other):
        if isinstance(other, AsymptoticVector):
            newself = self
            if other.tau < self.tau:
                other = other.changetau(self.tau)
            elif other.tau > self.tau:
                newself = self.changetau(other.tau)
            return AsymptoticVector(newself.v + other.v)
        elif hasattr(other, 'asymptotic_vector'):
            return self + other.asymptotic_vector
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
        return AsymptoticVector(a * self.v)

    def __rmul__(self, a):
        return self * a

    def __repr__(self):
        return f'AsymptoticVector({self.v!r})'

    def __eq__(self, other):
        return np.array_equal(self.v, other.v) if isinstance(other, AsymptoticVector) else False

    @staticmethod
    def stack_invert(J, unknowns, targets, tau=None):
        n = len(unknowns)
        assert n == len(targets)

        # first cast to asymptotic vector if not
        Jnew = {}
        for t in targets:
            Jnew[t] = {}
            for u in unknowns:
                if isinstance(J[t][u], AsymptoticVector):
                    Jnew[t][u] = J[t][u]
                else:
                    Jnew[t][u] = J[t][u].asymptotic_vector
        J = Jnew

        # now go through and find max length of everything
        max_tau = max(J[t][u].tau for t in targets for u in unknowns)
        if tau is not None:
            max_tau = max(max_tau, tau)
        else:
            tau = max_tau

        # stack all these together in an array
        stacked_rfft = np.empty((2 * max_tau, n, n), dtype=np.complex128)
        for i, t in enumerate(targets):
            for j, u in enumerate(unknowns):
                stacked_rfft[:, i, j] = J[t][u].changetau(max_tau).vfft

        print(stacked_rfft[:5, 0, 1].real)
        print(stacked_rfft.shape)

        # now stack the identity under similar conditions
        id_rfft = rfft(np.arange(4 * tau - 1) == (2 * tau - 1))
        stacked_id_rfft = np.zeros((2 * max_tau, n, n), dtype=np.complex128)
        for i in range(n):
            stacked_id_rfft[:, i, i] = id_rfft

        # now solve for everything and extract
        stacked_results = np.linalg.solve(stacked_rfft, stacked_id_rfft)
        invJ = {}
        for i, u in enumerate(unknowns):
            invJ[u] = {}
            for j, t in enumerate(targets):
                invJ[u][t] = irfft(stacked_results)[max_tau - tau + 1:max_tau + tau, i, j]

        return invJ


class AsymptoticToeplitz:
    """Represents infinite matrix that is asymptotically Toeplitz, with a finite explicit matrix M
    and then assumed asymptotic behavior beyond that given by AsymptoticVector asymp"""

    def __init__(self, M, asymp):
        self.M = M
        self.asymp = asymp if isinstance(asymp, AsymptoticVector) else AsymptoticVector(asymp)

        # M has to be square matrix, T*T
        self.T = M.shape[0]
        assert self.T == M.shape[1]

    def changeT(self, T):
        """return new AsymptoticToeplitz with lower or higher T, if higher filling in with v"""
        if T <= self.T:
            return AsymptoticToeplitz(self.M[:T, :T], self.asymp)
        else:
            # easiest way to do extension: use built-in Toeplitz to fill in matrix, then overwrite submatrix with M
            toeplitz_col1 = np.zeros(T)
            toeplitz_col1[:self.asymp.tau] = self.asymp.v[self.asymp.tau - 1:]
            toeplitz_row1 = np.zeros(T)
            toeplitz_row1[:self.asymp.tau] = self.asymp.v[:self.asymp.tau][::-1]
            M = toeplitz(toeplitz_col1, toeplitz_row1)
            M[:self.T, :self.T] = self.M
            return AsymptoticToeplitz(M, self.asymp)

    @property
    def T(self):
        """Transpose"""
        return AsymptoticToeplitz(self.M.T, self.asymp.T)

    def __pos__(self):
        return self

    def __neg__(self):
        return AsymptoticToeplitz(-self.M, -self.asymp)

    # def __matmul__(self, other):
    #     if isinstance(other, AsymptoticToeplitz):
    #         newself = self
    #         if other.tau

    #     if isinstance(other, AsymptoticVector):
    #         newself = self
    #         if other.tau < self.tau:
    #             other = other.changetau(self.tau)
    #         elif other.tau > self.tau:
    #             newself = self.changetau(other.tau)
    #         return irfft(newself.vfft*other.vfft)[newself.tau:-newself.tau]
    #     elif hasattr(other, 'asymptotic_vector'):
    #         return self @ other.asymptotic_vector
    #     else:
    #         return NotImplemented


def correction(av, bv, av_rz_fft, bv_lz_fft, T):
    """Correction matrix for right (tau-1)*(tau-1) submatrix of T*T matrix, possibly entire thing,
    given we're multiplying Toeplitz av times bv"""
    tau = (len(av) + 1) // 2
    assert len(av) == 2 * tau - 1 and len(bv) == 2 * tau - 1
    assert len(av_rz_fft) == 2 * tau and len(bv_lz_fft) == 2 * tau

    # initialize correction C as (tau-1)*(tau-1) submatrix of T*T matrix
    C = np.empty((min(tau - 1, T), min(tau - 1, T)))

    # compute full correction vector from asymptotics
    cv_full = irfft(av_rz_fft * bv_lz_fft, 4 * tau - 1)

    # split into correction for final column and final row, length tau-1 each
    cv_column = cv_full[tau:2 * tau - 1]
    cv_row = cv_full[2 * tau - 2:3 * tau - 3][::-1]

    # store these in C's actual final column and final row, sizing appropriately
    C[:, -1] = cv_column[-T:]
    C[-1, :] = cv_row[-T:]

    # now call simple correction matrix routine
    # given a_(-1), ... , a_(-S) and b_1, ..., b_S where S = min(tau-1, T)
    a = av[:tau - 1][::-1][:T]
    b = bv[tau:][:T]
    recursive_correction(C, a, b)

    return C


@njit
def recursive_correction(C, a, b):
    # C is S*S matrix with last row and column already filled
    # a is length-S vector with a_(-1), ... , a_(-S) asymptotics
    # b is length-S vector with b_1, ... , b_S asymptotics
    S = C.shape[0]
    for s in range(S - 2, -1, -1):
        for t in range(S - 1):
            C[s, t] = C[s + 1, t + 1] - a[S - s - 2] * b[S - t - 2]
