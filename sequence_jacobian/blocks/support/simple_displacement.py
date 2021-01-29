"""Displacement handler classes used by SimpleBlock for .ss, .td, and .jac evaluation to have Dynare-like syntax"""

import numpy as np
import numbers
from warnings import warn


def ignore(x):
    if isinstance(x, int):
        return IgnoreInt(x)
    elif isinstance(x, numbers.Real) and not isinstance(x, int):
        return IgnoreFloat(x)
    elif isinstance(x, np.ndarray):
        return IgnoreVector(x)
    else:
        raise TypeError(f"{type(x)} is not supported. Must provide either a float or an nd.array as an argument")


class IgnoreInt(int):
    """This class ignores time displacements of a scalar.
    Standard arithmetic operators including +, -, x, /, ** all overloaded to "promote" the result of
    any arithmetic operation with an Ignore type to an Ignore type. e.g. type(Ignore(1) + 1) is Ignore
    """
    @property
    def ss(self):
        return self

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
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__radd__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) + other)

    def __radd__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__add__(numeric_primitive(self))
        else:
            return ignore(other + numeric_primitive(self))

    def __sub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rsub__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) - other)

    def __rsub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__sub__(numeric_primitive(self))
        else:
            return ignore(other - numeric_primitive(self))

    def __mul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rmul__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) * other)

    def __rmul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__mul__(numeric_primitive(self))
        else:
            return ignore(other * numeric_primitive(self))

    def __truediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rtruediv__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) / other)

    def __rtruediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__truediv__(numeric_primitive(self))
        else:
            return ignore(other / numeric_primitive(self))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Displace) or isinstance(power, AccumulatedDerivative):
            return power.__rpow__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) ** power)

    def __rpow__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__pow__(numeric_primitive(self))
        else:
            return ignore(other ** numeric_primitive(self))


class IgnoreFloat(float):
    """This class ignores time displacements of a scalar.
    Standard arithmetic operators including +, -, x, /, ** all overloaded to "promote" the result of
    any arithmetic operation with an Ignore type to an Ignore type. e.g. type(Ignore(1) + 1) is Ignore
    """

    @property
    def ss(self):
        return self

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
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__radd__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) + other)

    def __radd__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__add__(numeric_primitive(self))
        else:
            return ignore(other + numeric_primitive(self))

    def __sub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rsub__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) - other)

    def __rsub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__sub__(numeric_primitive(self))
        else:
            return ignore(other - numeric_primitive(self))

    def __mul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rmul__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) * other)

    def __rmul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__mul__(numeric_primitive(self))
        else:
            return ignore(other * numeric_primitive(self))

    def __truediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rtruediv__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) / other)

    def __rtruediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__truediv__(numeric_primitive(self))
        else:
            return ignore(other / numeric_primitive(self))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Displace) or isinstance(power, AccumulatedDerivative):
            return power.__rpow__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) ** power)

    def __rpow__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__pow__(numeric_primitive(self))
        else:
            return ignore(other ** numeric_primitive(self))


class IgnoreVector(np.ndarray):
    """This class ignores time displacements of a np.ndarray.
       See NumPy documentation on "Subclassing ndarray" for more details on the use of __new__
       for this implementation."""

    def __new__(cls, x):
        obj = np.asarray(x).view(cls)
        return obj

    @property
    def ss(self):
        return self

    def __call__(self, index):
        return self

    def apply(self, f, **kwargs):
        return ignore(f(numeric_primitive(self), **kwargs))

    def __add__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__radd__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) + other)

    def __radd__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__add__(numeric_primitive(self))
        else:
            return ignore(other + numeric_primitive(self))

    def __sub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rsub__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) - other)

    def __rsub__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__sub__(numeric_primitive(self))
        else:
            return ignore(other - numeric_primitive(self))

    def __mul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rmul__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) * other)

    def __rmul__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__mul__(numeric_primitive(self))
        else:
            return ignore(other * numeric_primitive(self))

    def __truediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__rtruediv__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) / other)

    def __rtruediv__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__truediv__(numeric_primitive(self))
        else:
            return ignore(other / numeric_primitive(self))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Displace) or isinstance(power, AccumulatedDerivative):
            return power.__rpow__(numeric_primitive(self))
        else:
            return ignore(numeric_primitive(self) ** power)

    def __rpow__(self, other):
        if isinstance(other, Displace) or isinstance(other, AccumulatedDerivative):
            return other.__pow__(numeric_primitive(self))
        else:
            return ignore(other ** numeric_primitive(self))


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


class AccumulatedDerivative:
    """A container for accumulated derivative information to help calculate the sequence space Jacobian
    of the outputs of a SimpleBlock with respect to its inputs.
    Uses common (i, m) -> x notation as in SimpleSparse (see its docs for more details) as a sparse representation of
    a Jacobian of outputs Y at any time t with respect to inputs X at any time s.

    Attributes:
    `.elements`: `dict`
      A mapping from tuples, (i, m), to floats, x, where i is the index of the non-zero diagonal
      relative to the main diagonal (0), where m is the number of initial entries missing from the diagonal
      (same conceptually as in SimpleSparse), and x is the value of the accumulated derivatives.
    `.f_value`: `float`
      The function value of the AccumulatedDerivative to be used when applying the chain rule in finding a subsequent
      simple derivative. We can think of a SimpleBlock is a composition of simple functions
      (either time displacements, arithmetic operators, etc.), i.e. f_i(f_{i-1}(...f_2(f_1(y))...)), where
      at each step i as we are accumulating the derivatives through each simple function, if the derivative of any
      f_i requires the chain rule, we will need the function value of the previous f_{i-1} to calculate that derivative.
    `._keys`: `list`
      The keys from the `.elements` attribute for convenience.
    `._fp_values`: `list`
      The values from the `.elements` attribute for convenience. `_fp_values` stands for f prime values, i.e. the actual
      values of the accumulated derivative themselves.
    """

    def __init__(self, elements={(0, 0): 1.}, f_value=1.):
        self.elements = elements
        self.f_value = f_value
        self._keys = list(self.elements.keys())
        self._fp_values = np.fromiter(self.elements.values(), dtype=float)

    @property
    def ss(self):
        return ignore(self.f_value)

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x:.3f}' for (i, m), x in self.elements.items()) + '}'
        return f'AccumulatedDerivative({formatted})'

    # TODO: Rewrite this comment for clarity once confirmed that the paper's notation will change
    #   (i, m)/(j, n) correspond to the Q_(-i, m), Q_(-j, n) operators defined for
    #   Proposition 2 of the Sequence Space Jacobian paper.
    #   The flipped sign in the code is so that the index 'i' matches the k(i) notation
    #   for writing SimpleBlock functions. Thus, it follows the same convention as SimpleSparse.
    #   Also because __call__ on a AccumulatedDerivative is a simple shift operator, it will take the form
    #   Q_(-i, 0) being applied to Q_(-j, n) (following the notation in the paper)
    #   s.t. Q_(-i, 0) Q_(-j, n) = Q(k,l)
    def __call__(self, i):
        keys = [(i + j, compute_l(-i, 0, -j, n)) for j, n in self._keys]
        return AccumulatedDerivative(elements=dict(zip(keys, self._fp_values)), f_value=self.f_value)

    def apply(self, f, h=1e-5, **kwargs):
        if f == np.log:
            return AccumulatedDerivative(elements=dict(zip(self._keys,
                                                           [1 / self.f_value * x for x in self._fp_values])),
                                         f_value=np.log(self.f_value))
        else:
            return AccumulatedDerivative(elements=dict(zip(self._keys, [(f(self.f_value + h, **kwargs) -
                                                                         f(self.f_value - h, **kwargs)) / (2 * h) * x
                                                                        for x in self._fp_values])),
                                         f_value=f(self.f_value, **kwargs))

    def __pos__(self):
        return AccumulatedDerivative(elements=dict(zip(self._keys, +self._fp_values)), f_value=+self.f_value)

    def __neg__(self):
        return AccumulatedDerivative(elements=dict(zip(self._keys, -self._fp_values)), f_value=-self.f_value)

    def __add__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, self._fp_values)),
                                         f_value=self.f_value + numeric_primitive(other))
        elif isinstance(other, AccumulatedDerivative):
            elements = self.elements.copy()
            for im, x in other.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return AccumulatedDerivative(elements=elements, f_value=self.f_value + other.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __radd__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, self._fp_values)),
                                         f_value=numeric_primitive(other) + self.f_value)
        elif isinstance(other, AccumulatedDerivative):
            elements = other.elements.copy()
            for im, x in self.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x

            return AccumulatedDerivative(elements=elements, f_value=other.f_value + self.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __sub__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, self._fp_values)),
                                         f_value=self.f_value - numeric_primitive(other))
        elif isinstance(other, AccumulatedDerivative):
            elements = self.elements.copy()
            for im, x in other.elements.items():
                if im in elements:
                    elements[im] -= x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = -x

            return AccumulatedDerivative(elements=elements, f_value=self.f_value - other.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rsub__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, -self._fp_values)),
                                         f_value=numeric_primitive(other) - self.f_value)
        elif isinstance(other, AccumulatedDerivative):
            elements = other.elements.copy()
            for im, x in self.elements.items():
                if im in elements:
                    elements[im] -= x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = -x

            return AccumulatedDerivative(elements=elements, f_value=other.f_value - self.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __mul__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, self._fp_values * numeric_primitive(other))),
                                         f_value=self.f_value * numeric_primitive(other))
        elif isinstance(other, AccumulatedDerivative):
            return AccumulatedDerivative(elements=(self * other.f_value + other * self.f_value).elements,
                                         f_value=self.f_value * other.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rmul__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, numeric_primitive(other) * self._fp_values)),
                                         f_value=numeric_primitive(other) * self.f_value)
        elif isinstance(other, AccumulatedDerivative):
            return AccumulatedDerivative(elements=(other * self.f_value + self * other.f_value).elements,
                                         f_value=other.f_value * self.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __truediv__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, self._fp_values / numeric_primitive(other))),
                                         f_value=self.f_value / numeric_primitive(other))
        elif isinstance(other, AccumulatedDerivative):
            return AccumulatedDerivative(elements=((other.f_value * self - self.f_value * other) /
                                                   (other.f_value ** 2)).elements,
                                         f_value=self.f_value / other.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rtruediv__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, -numeric_primitive(other) /
                                                           self.f_value ** 2 * self._fp_values)),
                                         f_value=numeric_primitive(other) / self.f_value)
        elif isinstance(other, AccumulatedDerivative):
            return AccumulatedDerivative(elements=((self.f_value * other - other.f_value * self) /
                                                   (self.f_value ** 2)).elements, f_value=other.f_value / self.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __pow__(self, power, modulo=None):
        if np.isscalar(power):
            return AccumulatedDerivative(elements=dict(zip(self._keys, numeric_primitive(power) * self.f_value
                                                           ** numeric_primitive(power - 1) * self._fp_values)),
                                         f_value=self.f_value ** numeric_primitive(power))
        elif isinstance(power, AccumulatedDerivative):
            return AccumulatedDerivative(elements=(self.f_value ** (power.f_value - 1) * (
                    power.f_value * self + power * self.f_value * np.log(self.f_value))).elements,
                                         f_value=self.f_value ** power.f_value)
        else:
            raise NotImplementedError("This operation is not yet supported for non-scalar arguments")

    def __rpow__(self, other):
        if np.isscalar(other):
            return AccumulatedDerivative(elements=dict(zip(self._keys, np.log(other) * numeric_primitive(other) **
                                                           self.f_value * self._fp_values)),
                                         f_value=numeric_primitive(other) ** self.f_value)
        elif isinstance(other, AccumulatedDerivative):
            return AccumulatedDerivative(elements=(other.f_value ** (self.f_value - 1) * (
                    self.f_value * other + self * other.f_value * np.log(other.f_value))).elements,
                                         f_value=other.f_value ** self.f_value)
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


# TODO: This needs its own unit test
def vectorize_func_over_time(func, *args):
    """In `args` some arguments will be Displace objects and others will be Ignore/IgnoreVector objects.
    The Displace objects will have an extra time dimension (as its first dimension).
    We need to ensure that `func` is evaluated at the non-time dependent steady-state value of
    the Ignore/IgnoreVectors and at each of the time-dependent values, t, of the Displace objects or in other
    words along its time path.
    """
    d_inds = [i for i in range(len(args)) if isinstance(args[i], Displace)]
    x_path = []

    # np.shape(args[d_inds[0]])[0] is T, the size of the first dimension of the first Displace object
    # provided in args (assume all Displaces are the same shape s.t. they're conformable)
    for t in range(np.shape(args[d_inds[0]])[0]):
        x_path.append(func(*[args[i][t] if i in d_inds else args[i] for i in range(len(args))]))

    return np.array(x_path)


def apply_function(func, *args, **kwargs):
    """Ensure that for generic functions called within a block and acting on a Displace object
    properly instantiates the steady state value of the created Displace object"""
    if np.any([isinstance(x, Displace) for x in args]):
        x_path = vectorize_func_over_time(func, *args)
        return Displace(x_path, ss=func(*[x.ss if isinstance(x, Displace) else numeric_primitive(x) for x in args]))
    elif np.any([isinstance(x, AccumulatedDerivative) for x in args]):
        raise NotImplementedError(
            "Have not yet implemented general apply_function functionality for AccumulatedDerivatives")
    else:
        return func(*args, **kwargs)
