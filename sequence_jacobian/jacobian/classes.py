"""Various classes to support the computation of Jacobians"""

import copy
import numpy as np

from . import support


class IdentityMatrix:
    """Simple identity matrix class, cheaper than using actual np.eye(T) matrix,
    use to initialize Jacobian of a variable wrt itself"""
    __array_priority__ = 10_000

    def sparse(self):
        """Equivalent SimpleSparse representation, less efficient operations but more general."""
        return SimpleSparse({(0, 0): 1})

    def matrix(self, T):
        return np.eye(T)

    def __matmul__(self, other):
        """Identity matrix knows to simply return 'other' whenever it's multiplied by 'other'."""
        return copy.deepcopy(other)

    def __rmatmul__(self, other):
        return copy.deepcopy(other)

    def __mul__(self, a):
        return a*self.sparse()

    def __rmul__(self, a):
        return self.sparse()*a

    def __add__(self, x):
        return self.sparse() + x

    def __radd__(self, x):
        return x + self.sparse()

    def __sub__(self, x):
        return self.sparse() - x

    def __rsub__(self, x):
        return x - self.sparse()

    def __neg__(self):
        return -self.sparse()

    def __pos__(self):
        return self

    def __repr__(self):
        return 'IdentityMatrix'


class ZeroMatrix:
    """Simple zero matrix class, cheaper than using actual np.zeros((T,T)) matrix,
    use in common case where some outputs don't depend on inputs"""
    __array_priority__ = 10_000

    def sparse(self):
        return SimpleSparse({(0, 0): 0})

    def matrix(self, T):
        return np.zeros((T,T))

    def __matmul__(self, other):
        if isinstance(other, np.ndarray) and other.ndim == 1:
            return np.zeros_like(other)
        else:
            return self

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, a):
        return self

    def __rmul__(self, a):
        return self

    # copies seem inefficient here, try to live without them
    def __add__(self, x):
        return x

    def __radd__(self, x):
        return x

    def __sub__(self, x):
        return -x

    def __rsub__(self, x):
        return x

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    def __repr__(self):
        return 'ZeroMatrix'


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
    def T(self):
        """Transpose"""
        return SimpleSparse({(-i, m): x for (i, m), x in self.elements.items()})

    @property
    def iszero(self):
        return not self.nonzero().elements

    def nonzero(self):
        elements = self.elements.copy()
        for im, x in self.elements.items():
            # safeguard to retain sparsity: disregard extremely small elements (num error)
            if abs(elements[im]) < 1E-14:
                del elements[im]
        return SimpleSparse(elements)

    def __pos__(self):
        return self

    def __neg__(self):
        return SimpleSparse({im: -x for im, x in self.elements.items()})

    def __matmul__(self, A):
        if isinstance(A, SimpleSparse):
            # multiply SimpleSparse by SimpleSparse, simple analytical rules in multiply_rs_rs
            return SimpleSparse(support.multiply_rs_rs(self, A))
        elif isinstance(A, np.ndarray):
            # multiply SimpleSparse by matrix or vector, multiply_rs_matrix uses slicing
            indices, xs = self.array()
            if A.ndim == 2:
                return support.multiply_rs_matrix(indices, xs, A)
            elif A.ndim == 1:
                return support.multiply_rs_matrix(indices, xs, A[:, np.newaxis])[:, 0]
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


class NestedDict:
    def __init__(self, nesteddict, outputs=None, inputs=None):
        if isinstance(nesteddict, NestedDict):
            self.nesteddict = nesteddict.nesteddict
            self.outputs = nesteddict.outputs
            self.inputs = nesteddict.inputs
        else:
            self.nesteddict = nesteddict
            if outputs is None:
                outputs = list(nesteddict.keys())
            if inputs is None:
                inputs = []
                for v in nesteddict.values():
                    inputs.extend(list(v))
                inputs = deduplicate(inputs)

            self.outputs = list(outputs)
            self.inputs = list(inputs)

    def __repr__(self):
        return f'<{type(self).__name__} outputs={self.outputs}, inputs={self.inputs}>'

    def __iter__(self):
        return iter(self.outputs)

    def __or__(self, other):
        # non-in-place merge: make a copy, then update
        merged = type(self)(self.nesteddict, self.outputs, self.inputs)
        merged.update(other)
        return merged

    def __getitem__(self, x):
        if isinstance(x, str):
            # case 1: just a single output, give subdict
            return self.nesteddict[x]
        elif isinstance(x, tuple):
            # case 2: tuple, referring to output and input
            o, i = x
            o = self.outputs if o == slice(None, None, None) else o
            i = self.inputs if i == slice(None, None, None) else i
            if isinstance(o, str):
                if isinstance(i, str):
                    # case 2a: one output, one input, return single Jacobian
                    return self.nesteddict[o][i]
                else:
                    # case 2b: one output, multiple inputs, return dict
                    return {ii: self.nesteddict[o][ii] for ii in i}
            else:
                # case 2c: multiple outputs, one or more inputs, return NestedDict with outputs o and inputs i
                i = (i,) if isinstance(i, str) else i
                return type(self)({oo: {ii: self.nesteddict[oo][ii] for ii in i} for oo in o}, o, i)
        elif isinstance(x, list) or isinstance(x, set):
            # case 3: assume that list or set refers just to outputs, get all of those
            return type(self)({oo: self.nesteddict[oo] for oo in x}, x, self.inputs)
        else:
            raise ValueError(f'Tried to get impermissible item {x}')

    def get(self, *args, **kwargs):
        # this is for compatibility, not a huge fan
        return self.nesteddict.get(*args, **kwargs)

    def update(self, J):
        if set(self.inputs) != set(J.inputs):
            raise ValueError \
                (f'Cannot merge {type(self).__name__}s with non-overlapping inputs {set(self.inputs) ^ set(J.inputs)}')
        if not set(self.outputs).isdisjoint(J.outputs):
            raise ValueError \
                (f'Cannot merge {type(self).__name__}s with overlapping outputs {set(self.outputs) & set(J.outputs)}')
        self.outputs = self.outputs + J.outputs
        self.nesteddict = {**self.nesteddict, **J.nesteddict}

    def complete(self, filler):
        nesteddict = {}
        for o in self.outputs:
            nesteddict[o] = dict(self.nesteddict[o])
            for i in self.inputs:
                if i not in nesteddict[o]:
                    nesteddict[o][i] = filler
        return type(self)(nesteddict, self.outputs, self.inputs)


def deduplicate(mylist):
    """Remove duplicates while otherwise maintaining order"""
    return list(dict.fromkeys(mylist))


class JacobianDict(NestedDict):
    @staticmethod
    def identity(ks):
        return JacobianDict({k: {k: IdentityMatrix()} for k in ks}, ks, ks).complete()

    def complete(self):
        return super().complete(ZeroMatrix())

    def addinputs(self):
        """Add any inputs that were not already in output list as outputs, with the identity"""
        inputs = [x for x in self.inputs if x not in self.outputs]
        return self | JacobianDict.identity(inputs)

    def __matmul__(self, x):
        if isinstance(x, JacobianDict):
            return self.compose(x)
        else:
            return self.apply(x)

    def compose(self, J):
        o_list = self.outputs
        m_list = tuple(set(self.inputs) & set(J.outputs))
        i_list = J.inputs

        J_om = self.nesteddict
        J_mi = J.nesteddict
        J_oi = {}

        for o in o_list:
            J_oi[o] = {}
            for i in i_list:
                Jout = ZeroMatrix()
                for m in m_list:
                    J_om[o][m]
                    J_mi[m][i]
                    Jout += J_om[o][m] @ J_mi[m][i]
                J_oi[o][i] = Jout

        return JacobianDict(J_oi, o_list, i_list)

    def apply(self, x):
        # assume that all entries in x have some length T, and infer it
        T = len(next(iter(x.values())))

        inputs = x.keys() & set(self.inputs)
        J_oi = self.nesteddict
        y = {}

        for o in self.outputs:
            y[o] = np.zeros(T)
            for i in inputs:
                y[o] += J_oi[o][i] @ x[i]

        return y

    def pack(self, T):
        J = np.empty((len(self.outputs) * T, len(self.inputs) * T))
        for iO, O in enumerate(self.outputs):
            for iI, I in enumerate(self.inputs):
                J[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = support.make_matrix(self[O, I], T)
        return J

    @staticmethod
    def unpack(bigjac, outputs, inputs, T):
        """If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary"""
        jacdict = {}
        for iO, O in enumerate(outputs):
            jacdict[O] = {}
            for iI, I in enumerate(inputs):
                jacdict[O][I] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
        return JacobianDict(jacdict, outputs, inputs)

