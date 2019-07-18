import numpy as np
from numba import njit
import toeplitz
import utils

'''Part 1: SimpleBlock class and @simple decorator to generate it'''


def simple(f):
    return SimpleBlock(f)


class SimpleBlock:
    """Generated from simple block written in Dynare-ish style and decorated with @simple"""

    def __init__(self, f):
        self.f = f
        self.input_list = utils.input_list(f)
        self.output_list = utils.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<SimpleBlock '{self.f.__name__}'>"

    def ss(self, *args, **kwargs):
        args = [Ignore(x) for x in args]
        kwargs = {k: Ignore(v) for k, v in kwargs.items()}
        return self.f(*args, **kwargs)

    def td(self, ss, **kwargs):
        kwargs_new = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            kwargs_new[k] = Displace(v, ss=ss.get(k, None), name=k)

        for k in self.input_list:
            if k not in kwargs_new:
                kwargs_new[k] = Ignore(ss[k])

        return dict(zip(self.output_list, utils.make_tuple(self.f(**kwargs_new))))

    def jac(self, ss, T=None, shock_list=None, h=1E-5):
        """
        Assemble nested dict of Jacobians

        Parameters
        ----------
        ss : dict,
            steady state values
        T : int, optional
            number of time periods for explicit T*T Jacobian; if omitted, more efficient SimpleSparse objects returned
        shock_list : list of str, optional
            names of input variables to differentiate wrt; if omitted, assume all inputs
        h : float, optional
            radius for symmetric numerical differentiation

        Returns
        -------
        J : dict,
            Jacobians as nested dict of SimpleSparse objects or, if T specified, (T*T) matrices,
            with zero derivatives omitted by convention
        """
        if shock_list is None:
            shock_list = self.input_list

        raw_derivatives = {o: {} for o in self.output_list}
        x_ss_new = {k: Ignore(ss[k]) for k in self.input_list}

        # loop over all inputs to differentiate
        for i in shock_list:
            # detect all indices with which i appears
            reporter = Reporter(ss[i])
            x_ss_new[i] = reporter
            self.f(**x_ss_new)
            relevant_indices = reporter.myset
            relevant_indices.add(0)

            # evaluate derivative with respect to each and store in dict
            for index in relevant_indices:
                x_ss_new[i] = Perturb(ss[i], h, index)
                y_up_all = utils.make_tuple(self.f(**x_ss_new))

                x_ss_new[i] = Perturb(ss[i], -h, index)
                y_down_all = utils.make_tuple(self.f(**x_ss_new))
                for y_up, y_down, o in zip(y_up_all, y_down_all, self.output_list):
                    if y_up != y_down:
                        sparsederiv = raw_derivatives[o].setdefault(i, {})
                        sparsederiv[index] = (y_up - y_down) / (2 * h)
            x_ss_new[i] = Ignore(ss[i])

        # process raw_derivatives to return either SimpleSparse objects or matrices
        J = {o: {} for o in self.output_list}
        for o in self.output_list:
            for i in raw_derivatives[o].keys():
                if T is None:
                    J[o][i] = SimpleSparse.from_simple_diagonals(raw_derivatives[o][i])
                else:
                    J[o][i] = SimpleSparse.from_simple_diagonals(raw_derivatives[o][i]).matrix(T)

        return J


'''Part 2: SimpleSparse class to represent and work with sparse Jacobians of SimpleBlocks'''


class SimpleSparse:
    __array_priority__ = 1000

    def __init__(self, elements):
        """elements is dict mapping (i, m) -> x
        where i is diagonal, m is number of initial entries missing, x is value along the diagonal"""
        self.elements = elements
        self.indices, self.xs = None, None

    @staticmethod
    def from_simple_diagonals(elements):
        """Take in dict 'elements' just mapping i -> x where i is diagonal and x is value on diagonal, no initial
         entries missing, and produces SimpleSparse object. Arises from differentiation of SimpleBlocks."""
        return SimpleSparse({(i, 0): x for i, x in elements.items()})

    def matrix(self, T):
        return self + np.zeros((T, T))

    def array(self):
        """Combine im and x into ndarrays (for Numba), cache."""
        if self.indices is not None:
            return self.indices, self.xs
        else:
            indices, xs = zip(*self.elements.items())
            self.indices, self.xs = np.array(indices), np.array(xs)
            return self.indices, self.xs

    @property
    def asymptotic_vector(self):
        indices, xs = self.array()
        tau = np.max(np.abs(indices[:, 0])) + 1  # how far out do we go?
        v = np.zeros(2 * tau - 1)
        # v[indices[:, 0]+tau-1] = xs
        v[-indices[:, 0] + tau - 1] = xs  # switch from asymptotic ROW to asymptotic COLUMN
        return toeplitz.AsymptoticVector(v)

    @property
    def asymptotic_toeplitz(self):
        return self.asymptotic_vector.make_toeplitz()

    @property
    def T(self):
        """Transpose"""
        return SimpleSparse({(-i, m): x for (i, m), x in self.elements.items()})

    def __pos__(self):
        return self

    def __neg__(self):
        return SimpleSparse({im: -x for im, x in self.elements.items()})

    def __matmul__(self, A):
        if isinstance(A, SimpleSparse):
            return multiply_rs_rs(self, A)
        elif isinstance(A, np.ndarray):
            indices, xs = self.array()
            if A.ndim == 2:
                return multiply_rs_matrix(indices, xs, A)
            elif A.ndim == 1:
                return multiply_rs_matrix(indices, xs, A[:, np.newaxis])[:, 0]
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __rmatmul__(self, A):
        return (self.T @ A.T).T

    def __add__(self, A):
        if isinstance(A, SimpleSparse):
            elements = self.elements.copy()
            for im, x in A.elements.items():
                if im in elements:
                    elements[im] += x
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x
            return SimpleSparse(elements)
        else:
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
                return NotImplemented
            T = A.shape[0]
            A = A.ravel()
            for (i, m), x in self.elements.items():
                if i < 0:
                    A[T * (-i) + (T + 1) * m::T + 1] += x
                else:
                    A[i + (T + 1) * m:(T - i) * T:T + 1] += x
            return A.reshape((T, T))

    def __radd__(self, A):
        return self + A

    def __sub__(self, A):
        """slightly inefficient implementation with temporary for ease, avoid this"""
        return self + (-A)

    def __rsub__(self, A):
        return -self + A

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return SimpleSparse({im: a * x for im, x in self.elements.items()})

    def __rmul__(self, a):
        return self * a

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x:.3f}' for (i, m), x in self.elements.items()) + '}'
        return f'SimpleSparse({formatted})'

    def __eq__(self, s):
        return self.elements == s.elements


def multiply_basis(t1, t2):
    """Matrix multiplication operation mapping two sparse basis elements to another."""
    i, m = t1
    j, n = t2
    k = i + j
    if i >= 0:
        if j >= 0:
            l = max(m, n - i)
        elif k >= 0:
            l = max(m, n - k)
        else:
            l = max(m + k, n)
    else:
        if j <= 0:
            l = max(m + j, n)
        else:
            l = max(m, n) + min(-i, j)
    return k, l


def multiply_rs_rs(s1, s2):
    """Matrix multiplication operation on two SimpleSparse objects."""
    elements = {}
    for im, x in s1.elements.items():
        for jn, y in s2.elements.items():
            kl = multiply_basis(im, jn)
            if kl in elements:
                elements[kl] += x * y
            else:
                elements[kl] = x * y
    return SimpleSparse(elements)


@njit
def multiply_rs_matrix(indices, xs, A):
    n = indices.shape[0]
    T = A.shape[0]
    S = A.shape[1]
    Aout = np.zeros((T, S))

    for count in range(n):
        i = indices[count, 0]
        m = indices[count, 1]
        x = xs[count]

        if i == 0:
            for t in range(m, T):
                for s in range(S):
                    Aout[t, s] += x * A[t, s]
        elif i > 0:
            for t in range(m, T - i):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
        else:
            for t in range(m - i, T):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
    return Aout


'''Part 3: helper classes used by SimpleBlock for .ss, .td, and .jac evaluation'''


class Ignore(float):
    """This class ignores time displacements of a scalar."""

    def __call__(self, index):
        return self


class Displace(np.ndarray):
    """This class makes time displacements of a time path, given the steady-state value.
    Needed for SimpleBlock.td()"""

    def __new__(cls, x, ss=None, name='UNKNOWN'):
        obj = np.asarray(x).view(cls)
        obj.ss = ss
        obj.name = name
        return obj

    def __call__(self, index):
        if index != 0:
            if self.ss is None:
                raise KeyError(f'Trying to call {self.name}({index}), but steady-state {self.name} not given!')
            newx = np.empty_like(self)
            if index > 0:
                newx[:-index] = self[index:]
                newx[-index:] = self.ss
            else:
                newx[-index:] = self[:index]
                newx[:-index] = self.ss
            return newx
        else:
            return self


class Reporter(float):
    """This class adds to a shared set to tell us what x[i] are accessed.
    Needed for differentiation in SimpleBlock.jac()"""

    def __init__(self, value):
        self.myset = set()

    def __call__(self, index):
        self.myset.add(index)
        return self


class Perturb(float):
    """This class uses the shared set to perturb each x[i] separately, starting at steady-state values.
    Needed for differentiation in SimpleBlock.jac()"""

    def __new__(cls, value, h, index):
        if index == 0:
            return float.__new__(cls, value + h)
        else:
            return float.__new__(cls, value)

    def __init__(self, value, h, index):
        self.h = h
        self.index = index

    def __call__(self, index):
        if self.index == 0:
            if index == 0:
                return self
            else:
                return self - self.h
        else:
            if self.index == index:
                return self + self.h
            else:
                return self
