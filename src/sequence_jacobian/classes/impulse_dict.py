"""ImpulseDict class for manipulating impulse responses."""

import numpy as np

from .result_dict import ResultDict

from ..utilities.ordered_set import OrderedSet
from ..utilities.bijection import Bijection
from .steady_state_dict import SteadyStateDict

class ImpulseDict(ResultDict):
    def __init__(self, data, internals=None, T=None):
        if isinstance(data, ImpulseDict):
            if internals is not None or T is not None:
                raise ValueError('Supplying ImpulseDict and also internal or T to constructor not allowed')
            super().__init__(data)
            self.T = data.T
        else:
            if not isinstance(data, dict):
                raise ValueError('ImpulseDicts are initialized with a `dict` of top-level impulse responses.')
            super().__init__(data, internals)
            self.T = (T if T is not None else self.infer_length())

    def __getitem__(self, k):
        return super().__getitem__(k, T=self.T)

    def __add__(self, other):
        return self.binary_operation(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.binary_operation(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self.binary_operation(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self.binary_operation(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.binary_operation(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self.binary_operation(other, lambda a, b: b / a)

    def __neg__(self):
        return self.unary_operation(lambda a: -a)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.unary_operation(lambda a: abs(a))

    def binary_operation(self, other, op):
        if isinstance(other, (SteadyStateDict, ImpulseDict)):
            toplevel = {k: op(v, other[k]) for k, v in self.toplevel.items()}
            internals = {}
            for b in self.internals:
                other_internals = other.internals[b]
                internals[b] = {k: op(v, other_internals[k]) for k, v in self.internals[b].items()} 
            return ImpulseDict(toplevel, internals, self.T)
        elif isinstance(other, (float, int)):
            toplevel = {k: op(v, other) for k, v in self.toplevel.items()}
            internals = {}
            for b in self.internals:
                internals[b] = {k: op(v, other) for k, v in self.internals[b].items()} 
            return ImpulseDict(toplevel, internals, self.T)
        else:
            return NotImplementedError(f'Can only perform operations with ImpulseDicts and other ImpulseDicts, SteadyStateDicts, or numbers, not {type(other).__name__}')

    def unary_operation(self, op):
        toplevel = {k: op(v) for k, v in self.toplevel.items()}
        internals = {}
        for b in self.internals:
            internals[b] = {k: op(v) for k, v in self.internals[b].items()} 
        return ImpulseDict(toplevel, internals, self.T)
        
    def pack(self):
        T = self.T
        bigv = np.empty(T*len(self.toplevel))
        for i, v in enumerate(self.toplevel.values()):
            bigv[i*T:(i+1)*T] = v
        return bigv

    @staticmethod
    def unpack(bigv, outputs, T):
        impulse = {}
        for i, o in enumerate(outputs):
            impulse[o] = bigv[i*T:(i+1)*T]
        return ImpulseDict(impulse, T=T)

    def infer_length(self):
        lengths = [len(v) for v in self.toplevel.values()]
        length = max(lengths)
        if length != min(lengths):
            raise ValueError(f'Building ImpulseDict with inconsistent lengths {max(lengths)} and {min(lengths)}')
        return length

    def get(self, k):
        """Like __getitem__ but with default of zero impulse"""
        if isinstance(k, str):
            return self.toplevel.get(k, np.zeros(self.T))
        elif isinstance(k, tuple):
            raise TypeError(f'Key {k} to {type(self).__name__} cannot be tuple')
        else:
            try:
                return type(self)({ki: self.toplevel.get(ki, np.zeros(self.T)) for ki in k}, T=self.T)
            except TypeError:
                raise TypeError(f'Key {k} to {type(self).__name__} needs to be a string or an iterable (list, set, etc) of strings')
