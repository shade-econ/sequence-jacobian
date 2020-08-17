import numpy as np
import numbers
import copy
from numba import njit
from warnings import warn

from .. import utils
from .. import asymptotic

'''Part 1: SimpleBlock class and @simple decorator to generate it'''


def simple(f):
    return SimpleBlock(f)


class SimpleBlock:
    """Generated from simple block written in Dynare-ish style and decorated with @simple, e.g.
    
    @simple
    def production(Z, K, L, alpha):
        Y = Z * K(-1) ** alpha * L ** (1 - alpha)
        return Y

    which is a SimpleBlock that takes in Z, K, L, and alpha, all of which can be either constants
    or series, and implements a Cobb-Douglas production function, noting that for production today
    we use the capital K(-1) determined yesterday.
    
    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        self.f = f
        self.input_list = utils.input_list(f)
        self.output_list = utils.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<SimpleBlock '{self.f.__name__}'>"

    def _output_in_ss_format(self, *args, **kwargs):
        """Returns output of the method ss as either a tuple of numeric primitives (scalars/vectors) or a single
        numeric primitive, as opposed to Ignore/IgnoreVector objects"""
        if len(self.output_list) > 1:
            return tuple([numeric_primitive(o) for o in self.f(*args, **kwargs)])
        else:
            return numeric_primitive(self.f(*args, **kwargs))

    def ss(self, *args, **kwargs):
        # Wrap args and kwargs in Ignore/IgnoreVector classes to be passed into the function "f"
        args = [ignore(x) for x in args]
        kwargs = {k: ignore(v) for k, v in kwargs.items()}

        return self._output_in_ss_format(*args, **kwargs)

    def _output_in_td_format(self, **kwargs_new):
        """Returns output of the method td as a dict mapping output names to numeric primitives (scalars/vectors)
        or a single numeric primitive of output values, as opposed to Ignore/IgnoreVector/Displace objects.

        Also accounts for the fact that for outputs of block.td that were *not* affected by a Displace object, i.e.
        variables that remained at their ss value in spite of other variables within that same block being
        affected by the Displace object (e.g. I in the mkt_clearing block of the two_asset model
        is unchanged by a shock to rstar, being only a function of K's ss value and delta),
        we still want to return them as paths (i.e. vectors, if they were
        previously scalars) to impose uniformity on the dimensionality of the td returned values.
        """
        out = self.f(**kwargs_new)
        if len(self.output_list) > 1:
            # Because we know at least one of the outputs in `out` must be of length T
            T = np.max([np.size(o) for o in out])
            out_unif_dim = [np.full(T, numeric_primitive(o)) if np.isscalar(o) else numeric_primitive(o) for o in out]
            return dict(zip(self.output_list, utils.make_tuple(out_unif_dim)))
        else:
            return dict(zip(self.output_list, utils.make_tuple(numeric_primitive(out))))

    def td(self, ss, **kwargs):
        kwargs_new = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            kwargs_new[k] = Displace(v, ss=ss.get(k, None), name=k)

        for k in self.input_list:
            if k not in kwargs_new:
                kwargs_new[k] = ignore(ss[k])

        return self._output_in_td_format(**kwargs_new)

    def jac(self, ss, T=None, shock_list=[]):
        """Assemble nested dict of Jacobians

        Parameters
        ----------
        ss : dict,
            steady state values
        T : int, optional
            number of time periods for explicit T*T Jacobian
            if omitted, more efficient SimpleSparse objects returned
        shock_list : list of str, optional
            names of input variables to differentiate wrt; if omitted, assume all inputs
        h : float, optional
            radius for symmetric numerical differentiation

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives Jacobian of o with respect to i
            This Jacobian is a SimpleSparse object or, if T specific, a T*T matrix, omitted by convention
            if zero
        """

        relevant_shocks = [i for i in self.inputs if i in shock_list]

        # If none of the shocks passed in shock_list are relevant to this block (i.e. none of the shocks
        # are an input into the block), then return an empty dict
        if not relevant_shocks:
            return {}
        else:
            invertedJ = {shock_name: {} for shock_name in relevant_shocks}

            # Loop over all inputs/shocks which we want to differentiate with respect to
            for shock in relevant_shocks:
                invertedJ[shock] = compute_single_shock_curlyJ(self.f, ss, shock, T=T)

            # Because we computed the Jacobian of all outputs with respect to each shock (invertedJ[i][o]),
            # we need to loop back through to have J[o][i] to map for a given output `o`, shock `i`,
            # the Jacobian curlyJ^{o,i}.
            J = {o: {} for o in self.output_list}
            for o in self.output_list:
                for i in relevant_shocks:
                    # Remove empty Jacobians corresponding to outputs of a block.
                    # This occurs when one of the block's outputs is not a function of any of the shocks
                    # and hence does not change with respect to them.
                    if not invertedJ[i][o]:
                        continue
                    else:
                        J[o][i] = invertedJ[i][o]

            return J


def compute_single_shock_curlyJ(f, steady_state_dict, shock_name, T=None):
    """Find the Jacobian of the function `f` with respect to a single shocked argument, `shock_name`"""
    input_args = {i: ignore(steady_state_dict[i]) for i in utils.input_list(f)}
    input_args[shock_name] = DerivativeMap(ss=steady_state_dict[shock_name])

    J = {o: {} for o in utils.output_list(f)}
    for o, o_name in zip(utils.make_tuple(f(**input_args)), utils.output_list(f)):
        if isinstance(o, DerivativeMap):
            J[o_name] = SimpleSparse(o.elements) if T is None else SimpleSparse(o.elements).matrix(T)

    return J


'''Part 2: SimpleSparse class to represent and work with sparse Jacobians of SimpleBlocks'''


class SimpleSparse:
    """Efficient representation of sparse linear operators, which are linear combinations of basis
    operators represented by pairs (i, m), where i is the index of diagonal on which there are 1s
    (measured by # above main diagonal) and m is number of initial entries missing.

    Examples of such basis operators:
        - (0, 0) is identity operator
        - (0, 2) is identity operator with first two '1's on main diagonal missing
        - (1, 0) has 1s on diagonal above main diagonal: "left-shift" operator
        - (-1, 1) has 1s on diagonal below main diagonal, except first column

    The linear combination of these basis operators that makes up a given SimpleSparse object is
    stored as a dict 'elements' mapping (i, m) -> x.

    The Jacobian of a SimpleBlock is a SimpleSparse operator combining basis elements (i, 0). We need
    the more general basis (i, m) to ensure closure under multiplication.

    These (i, m) correspond to the Q_(-i, m) operators defined for Proposition 2 of the Sequence Space
    Jacobian paper. The flipped sign in the code is so that the index 'i' matches the k(i) notation
    for writing SimpleBlock functions.

    The "dunder" methods x.__add__(y), x.__matmul__(y), x.__rsub__(y), etc. in Python implement infix
    operations x + y, x @ y, y - x, etc. Defining these allows us to use these more-or-less
    interchangeably with ordinary NumPy matrices.
    """

    # when performing binary operations on SimpleSparse and a NumPy array, use SimpleSparse's rules
    __array_priority__ = 1000

    def __init__(self, elements):
        self.elements = elements
        self.indices, self.xs = None, None

    @staticmethod
    def from_simple_diagonals(elements):
        """Take dict i -> x, i.e. from SimpleBlock differentiation, convert to SimpleSparse (i, 0) -> x"""
        return SimpleSparse({(i, 0): x for i, x in elements.items()})

    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""
        return self + np.zeros((T, T))

    def array(self):
        """Rewrite dict (i, m) -> x as pair of NumPy arrays, one size-N*2 array of ints with rows (i, m)
        and one size-N array of floats with entries x.

        This is needed for Numba to take as input. Cache for efficiency.
        """
        if self.indices is not None:
            return self.indices, self.xs
        else:
            indices, xs = zip(*self.elements.items())
            self.indices, self.xs = np.array(indices), np.array(xs)
            return self.indices, self.xs

    @property
    def asymptotic_time_invariant(self):
        indices, xs = self.array()
        tau = np.max(np.abs(indices[:, 0]))+1 # how far out do we go?
        v = np.zeros(2*tau-1)
        #v[indices[:, 0]+tau-1] = xs
        v[-indices[:, 0]+tau-1] = xs # switch from asymptotic ROW to asymptotic COLUMN
        return asymptotic.AsymptoticTimeInvariant(v)

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
            # multiply SimpleSparse by SimpleSparse, simple analytical rules in multiply_rs_rs
            return multiply_rs_rs(self, A)
        elif isinstance(A, np.ndarray):
            # multiply SimpleSparse by matrix or vector, multiply_rs_matrix uses slicing
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
        # multiplication rule when this object is on right (will only be called when left is matrix)
        # for simplicity, just use transpose to reduce this to previous cases
        return (self.T @ A.T).T

    def __add__(self, A):
        if isinstance(A, SimpleSparse):
            # add SimpleSparse to SimpleSparse, combining dicts, summing x when (i, m) overlap
            elements = self.elements.copy()
            for im, x in A.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x
            return SimpleSparse(elements)
        else:
            # add SimpleSparse to T*T matrix
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
                return NotImplemented
            T = A.shape[0]

            # fancy trick to do this efficiently by writing A as flat vector
            # then (i, m) can be mapped directly to NumPy slicing!
            A = A.flatten()     # use flatten, not ravel, since we'll modify A and want a copy
            for (i, m), x in self.elements.items():
                if i < 0:
                    A[T * (-i) + (T + 1) * m::T + 1] += x
                else:
                    A[i + (T + 1) * m:(T - i) * T:T + 1] += x
            return A.reshape((T, T))

    def __radd__(self, A):
        try:
            return self + A
        except:
            print(self)
            print(A)
            raise

    def __sub__(self, A):
        # slightly inefficient implementation with temporary for simplicity
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
    # equivalent to formula in Proposition 2 of Sequence Space Jacobian paper, but with
    # signs of i and j flipped to reflect different sign convention used here
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
    # iterate over all pairs (i, m) -> x and (j, n) -> y in objects,
    # add all pairwise products to get overall product
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
    """Matrix multiplication of SimpleSparse object ('indices' and 'xs') and matrix A.
    Much more computationally demanding than multiplying two SimpleSparse (which is almost
    free with simple analytical formula), so we implement as jitted function."""
    n = indices.shape[0]
    T = A.shape[0]
    S = A.shape[1]
    Aout = np.zeros((T, S))

    for count in range(n):
        # for Numba to jit easily, SimpleSparse with basis elements '(i, m)' with coefs 'x'
        # was stored in 'indices' and 'xs'
        i = indices[count, 0]
        m = indices[count, 1]
        x = xs[count]

        # loop faster than vectorized when jitted
        # directly use def of basis element (i, m), displacement of i and ignore first m
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


def ignore(x):
    if isinstance(x, numbers.Real):
        return Ignore(x)
    elif isinstance(x, np.ndarray):
        return IgnoreVector(x)
    else:
        raise TypeError(f"{type(x)} is not supported. Must provide either a float or an nd.array as an argument")


class Ignore(float):
    """This class ignores time displacements of a scalar.
    Standard arithmetic operators including +, -, x, /, ** all overloaded to "promote" the result of
    any arithmetic operation with an Ignore type to an Ignore type. e.g. type(Ignore(1) + 1) is Ignore
    """

    def __call__(self, index):
        return self

    def apply(self, f, **kwargs):
        return ignore(f(numeric_primitive(self), **kwargs))

    def __pos__(self):
        return self

    def __neg__(self):
        return ignore(-numeric_primitive(self))

    # Tried using the multipledispatch package but @dispatch requires the classes being dispatched on to be defined
    # prior to the use of the decorator @dispatch("ClassName"), hence making it impossible to overload in this way,
    # as opposed to how isinstance() is evaluated at runtime, so it is valid to check isinstance even if in this module
    # the class is defined later on in the module.
    # Thus, we need to specially overload the left operations to check if `other` is a Displace to promote properly
    def __add__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__radd__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) + other)

    def __radd__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__add__(numeric_primitive(self))
        else:
            return ignore(other + numeric_primitive(self))

    def __sub__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rsub__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) - other)

    def __rsub__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__sub__(numeric_primitive(self))
        else:
            return ignore(other - numeric_primitive(self))

    def __mul__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rmul__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) * other)

    def __rmul__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__mul__(numeric_primitive(self))
        else:
            return ignore(other * numeric_primitive(self))

    def __truediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rtruediv__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self)/other)

    def __rtruediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__truediv__(numeric_primitive(self))
        else:
            return ignore(other/numeric_primitive(self))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Displace) or isinstance(power, DerivativeMap):
            return power.__rpow__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self)**power)

    def __rpow__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__pow__(numeric_primitive(self))
        else:
            return ignore(other**numeric_primitive(self))


class IgnoreVector(np.ndarray):
    """This class ignores time displacements of a np.ndarray.
       See NumPy documentation on "Subclassing ndarray" for more details on the use of __new__
       for this implementation."""

    def __new__(cls, x):
        obj = np.asarray(x).view(cls)
        return obj

    def __call__(self, index):
        return self

    def apply(self, f, **kwargs):
        return ignore(f(numeric_primitive(self), **kwargs))

    def __add__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__radd__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) + other)

    def __radd__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__add__(numeric_primitive(self))
        else:
            return ignore(other + numeric_primitive(self))

    def __sub__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rsub__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) - other)

    def __rsub__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__sub__(numeric_primitive(self))
        else:
            return ignore(other - numeric_primitive(self))

    def __mul__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rmul__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) * other)

    def __rmul__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__mul__(numeric_primitive(self))
        else:
            return ignore(other * numeric_primitive(self))

    def __truediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__rtruediv__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self)/other)

    def __rtruediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__truediv__(numeric_primitive(self))
        else:
            return ignore(other/numeric_primitive(self))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Displace) or isinstance(power, DerivativeMap):
            return power.__rpow__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self)**power)

    def __rpow__(self, other):
        if isinstance(other, Displace) or isinstance(other, DerivativeMap):
            return other.__pow__(numeric_primitive(self))
        else:
            return ignore(other**numeric_primitive(self))


class Displace(np.ndarray):
    """This class makes time displacements of a time path, given the steady-state value.
    Needed for SimpleBlock.td()"""

    def __new__(cls, x, ss=None, name='UNKNOWN'):
        obj = np.asarray(x).view(cls)
        obj.ss = ss
        obj.name = name
        return obj

    # TODO: Implemented a very preliminary generalization of Displace to higher-dimensional (>1) ndarrays
    #   however the rigorous operator overloading/testing has not been checked for higher dimensions.
    #   Should also implement some checks for the dimension of .ss, to ensure that it's always N-1
    #   where we also assume that the *last* dimension is the time dimension
    def __call__(self, index):
        if index != 0:
            if self.ss is None:
                raise KeyError(f'Trying to call {self.name}({index}), but steady-state {self.name} not given!')
            newx = np.zeros(np.shape(self))
            if index > 0:
                newx[..., :-index] = numeric_primitive(self)[..., index:]
                newx[..., -index:] = self.ss
            else:
                newx[..., -index:] = numeric_primitive(self)[..., :index]
                newx[..., :-index] = self.ss
            return Displace(newx, ss=self.ss)
        else:
            return self

    def apply(self, f, **kwargs):
        return Displace(f(numeric_primitive(self), **kwargs), ss=f(self.ss))

    def __pos__(self):
        return self

    def __neg__(self):
        return Displace(-numeric_primitive(self), ss=-self.ss)

    def __add__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(self) + numeric_primitive(other),
                            ss=self.ss + other.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(self) + numeric_primitive(other),
                            ss=self.ss + numeric_primitive(other))
        else:
            # TODO: See if there is a different, systematic way we want to handle this case.
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(self) + numeric_primitive(other),
                            ss=self.ss)

    def __radd__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(other) + numeric_primitive(self),
                            ss=other.ss + self.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(other) + numeric_primitive(self),
                            ss=numeric_primitive(other) + self.ss)
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(other) + numeric_primitive(self),
                            ss=self.ss)

    def __sub__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(self) - numeric_primitive(other),
                            ss=self.ss - other.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(self) - numeric_primitive(other),
                            ss=self.ss - numeric_primitive(other))
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(self) - numeric_primitive(other),
                            ss=self.ss)

    def __rsub__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(other) - numeric_primitive(self),
                            ss=other.ss - self.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(other) - numeric_primitive(self),
                            ss=numeric_primitive(other) - self.ss)
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(other) - numeric_primitive(self),
                            ss=self.ss)

    def __mul__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(self) * numeric_primitive(other),
                            ss=self.ss * other.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(self) * numeric_primitive(other),
                            ss=self.ss * numeric_primitive(other))
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(self) * numeric_primitive(other),
                            ss=self.ss)

    def __rmul__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(other) * numeric_primitive(self),
                            ss=other.ss * self.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(other) * numeric_primitive(self),
                            ss=numeric_primitive(other) * self.ss)
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(other) * numeric_primitive(self),
                            ss=self.ss)

    def __truediv__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(self) / numeric_primitive(other),
                            ss=self.ss / other.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(self) / numeric_primitive(other),
                            ss=self.ss / numeric_primitive(other))
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(self) / numeric_primitive(other),
                            ss=self.ss)

    def __rtruediv__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(other) / numeric_primitive(self),
                            ss=other.ss / self.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(other) / numeric_primitive(self),
                            ss=numeric_primitive(other) / self.ss)
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(other) / numeric_primitive(self),
                            ss=self.ss)

    def __pow__(self, power):
        if isinstance(power, Displace):
            return Displace(numeric_primitive(self) ** numeric_primitive(power),
                            ss=self.ss ** power.ss)
        elif np.isscalar(power):
            return Displace(numeric_primitive(self) ** numeric_primitive(power),
                            ss=self.ss ** numeric_primitive(power))
        else:
            warn("\n" + f"Applying operation to {power}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(self) ** numeric_primitive(power),
                            ss=self.ss)

    def __rpow__(self, other):
        if isinstance(other, Displace):
            return Displace(numeric_primitive(other) ** numeric_primitive(self),
                            ss=other.ss ** self.ss)
        elif np.isscalar(other):
            return Displace(numeric_primitive(other) ** numeric_primitive(self),
                            ss=numeric_primitive(other) ** self.ss)
        else:
            warn("\n" + f"Applying operation to {other}, a vector, and {self}, a Displace." + "\n" +
                 f"The resulting Displace object will retain the steady-state value of the original Displace object.")
            return Displace(numeric_primitive(other) ** numeric_primitive(self),
                            ss=self.ss)


class DerivativeMap:
    """A mapping (i, m) -> x, where i is the index of the non-zero diagonal relative to the main diagonal (0), where
    m is the number of initial entries missing from the diagonal (same conceptually as in SimpleSparse).
    The purpose of this object is to be an efficient, flexible container for accumulating derivative information
    while calculating the Jacobian of a SimpleBlock.
    """

    def __init__(self, elements={(0, 0): 1.}, ss=1.):
        self.elements = elements
        self.ss = ss  # Track the ss value of the DerivativeMap so we can properly apply the chain rule if needed
        self._keys = list(self.elements.keys())
        self._values = np.fromiter(self.elements.values(), dtype=float)

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x:.3f}' for (i, m), x in self.elements.items()) + '}'
        return f'DerivativeMap({formatted})'

    # Treat it as if the operator Q_(i, 0) is being applied to Q_(j, n), following the notation in the paper
    # s.t. Q_(i, 0) Q_(j, n) = Q(k,l)
    def __call__(self, i):
        keys = [(i + j, compute_l(i, 0, j, n)) for j, n in self._keys]
        return DerivativeMap(elements=dict(zip(keys, self._values)), ss=self.ss)

    def apply(self, f, **kwargs):
        return DerivativeMap(elements=dict(zip(self._keys, [f(x, **kwargs) for x in self._values])),
                             ss=f(self.ss, **kwargs))

    def __pos__(self):
        return DerivativeMap(elements=dict(zip(self._keys, +self._values)), ss=+self.ss)

    def __neg__(self):
        return DerivativeMap(elements=dict(zip(self._keys, -self._values)), ss=-self.ss)

    def __add__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, self._values)),
                                 ss=self.ss + numeric_primitive(other))
        elif isinstance(other, DerivativeMap):
            elements = self.elements.copy()
            for im, x in other.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return DerivativeMap(elements=elements, ss=self.ss + other.ss)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __radd__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, self._values)),
                                 ss=numeric_primitive(other) + self.ss)
        elif isinstance(other, DerivativeMap):
            elements = other.elements.copy()
            for im, x in self.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return DerivativeMap(elements=elements, ss=other.ss + self.ss)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __sub__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, self._values)),
                                 ss=self.ss - numeric_primitive(other))
        elif isinstance(other, DerivativeMap):
            elements = self.elements.copy()
            for im, x in other.elements.items():
                if im in elements:
                    elements[im] -= x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return DerivativeMap(elements=elements, ss=self.ss - other.ss)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rsub__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, -self._values)),
                                 ss=numeric_primitive(other) - self.ss)
        elif isinstance(other, DerivativeMap):
            elements = other.elements.copy()
            for im, x in self.elements.items():
                if im in elements:
                    elements[im] -= x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return DerivativeMap(elements=elements, ss=other.ss - self.ss)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __mul__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, self._values * numeric_primitive(other))),
                                 ss=self.ss * numeric_primitive(other))
        elif isinstance(other, DerivativeMap):
            return self * other.ss + other * self.ss
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rmul__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, numeric_primitive(other) * self._values)),
                                 ss=numeric_primitive(other) * self.ss)
        elif isinstance(other, DerivativeMap):
            return other * self.ss + self * other.ss
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __truediv__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, self._values/numeric_primitive(other))),
                                 ss=self.ss/numeric_primitive(other))
        elif isinstance(other, DerivativeMap):
            return (other.ss * self - self.ss * other.ss)/(other.ss**2)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, numeric_primitive(other)/self._values)),
                                 ss=numeric_primitive(other)/self.ss)
        elif isinstance(other, DerivativeMap):
            return (self.ss * other - other.ss * self)/(self.ss**2)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __pow__(self, power, modulo=None):
        if np.isscalar(power):
            return DerivativeMap(elements=dict(zip(self._keys, numeric_primitive(power) *
                                                   self.ss**numeric_primitive(power - 1) * self._values)),
                                 ss=self.ss**numeric_primitive(power))
        elif isinstance(power, DerivativeMap):
            return NotImplemented
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rpow__(self, other):
        if np.isscalar(other):
            return DerivativeMap(elements=dict(zip(self._keys, np.log(other) * numeric_primitive(other)**self._values)),
                                 ss=numeric_primitive(other)**self.ss)
        elif isinstance(other, DerivativeMap):
            return NotImplemented
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")


def compute_l(i, m, j, n):
    """Computes the `l` index from the composition of shift operators, Q_{i, m} Q_{j, n} = Q_{k, l} in Proposition 2
    of the paper (regarding efficient multiplication of simple Jacobians)."""
    if i >= 0 and j >= 0:
        return max(m - j, n)
    elif i >= 0 and j <= 0:
        return max(m, n) + min(i, -j)
    elif i <= 0 and j >= 0 and i + j >= 0:
        return max(m - i - j, n)
    elif i <= 0 and j >= 0 and i + j <= 0:
        return max(n + i + j, m)
    else:
        return max(m, n + i)


def numeric_primitive(instance):
    # If it is already a primitive, just return it
    if type(instance) in {int, float, np.ndarray}:
        return instance
    else:
        return instance.real if np.isscalar(instance) else instance.base


def shift_first_dim_to_last(v):
    """For `v`, an np.ndarray, shift the first dimension to the last dimension,
    e.g. if np.shape(v) = (3, 4, 5), then shift_first_dim_to_last(v) returns the same data as v
    but in shape (4, 5, 3).
    This is useful since in apply_function, the default behavior of iterating across the time dimension and then
    calling np.array on the resulting list of arrays returns the time dimension in the first index (but we want
    it in the last index)
    """
    if np.ndim(v) > 1:
        return np.moveaxis(v, 0, -1)
    else:
        return v


def vectorize_func_over_time(func, *args):
    """In `args` some arguments will be Displace objects and others will be Ignore/IgnoreVector objects.
    The Displace objects will have an extra time dimension (as its last dimension).
    We need to ensure that `func` is evaluated at the non-time dependent steady-state value of
    the Ignore/IgnoreVectors and at each of the time-dependent values, t, of the Displace objects or in other
    words along its time path.
    """
    d_inds = [i for i in range(len(args)) if isinstance(args[i], Displace)]
    x_path = []
    # np.shape(args[d_inds[0]])[-1] is T, the size of the last dimension of the first Displace object
    # provided in args (assume all Displaces are the same shape s.t. they're conformable)
    for t in range(np.shape(args[d_inds[0]])[-1]):
        x_path.append(func(*[args[i][t] if i in d_inds else args[i] for i in range(len(args))]))

    # Need an extra call to shift_first_dim_to_last, since the way Python collects a list of arrays through np.array
    # (the list's elements are across the time dimension) by default is to add it as the first
    # dimension, but we want time to be the last dimension so need to move that axis
    return shift_first_dim_to_last(np.array(x_path))


def apply_function(func, *args, **kwargs):
    """Ensure that for generic functions called within a block and acting on a Displace object
    properly instantiates the steady state value of the created Displace object"""
    if np.any([isinstance(x, Displace) for x in args]):
        x_path = vectorize_func_over_time(func, *args)
        return Displace(x_path, ss=func(*[x.ss if isinstance(x, Displace) else numeric_primitive(x) for x in args]))
    elif np.any([isinstance(x, DerivativeMap) for x in args]):
        raise NotImplementedError("Have not yet implemented general apply_function functionality for DerivativeMaps")
    else:
        return func(*args, **kwargs)
