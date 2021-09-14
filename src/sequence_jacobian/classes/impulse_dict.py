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

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v + other for k, v in self.toplevel.items()})
        elif isinstance(other, (SteadyStateDict, ImpulseDict)):
            return type(self)({k: v + other[k] for k, v in self.toplevel.items()})
        else:
            return NotImplementedError('Only a number or a SteadyStateDict can be added from an ImpulseDict.')

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v - other for k, v in self.toplevel.items()})
        elif isinstance(other, (SteadyStateDict, ImpulseDict)):
            return type(self)({k: v - other[k] for k, v in self.toplevel.items()})
        else:
            return NotImplementedError('Only a number or a SteadyStateDict can be subtracted from an ImpulseDict.')

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v * other for k, v in self.toplevel.items()})
        elif isinstance(other, (SteadyStateDict, ImpulseDict)):
            return type(self)({k: v * other[k] for k, v in self.toplevel.items()})
        else:
            return NotImplementedError('An ImpulseDict can only be multiplied by a number or a SteadyStateDict.')

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v * other for k, v in self.toplevel.items()})
        elif isinstance(other, SteadyStateDict):
            return type(self)({k: v * other[k] for k, v in self.toplevel.items()})
        else:
            return NotImplementedError('An ImpulseDict can only be multiplied by a number or a SteadyStateDict.')

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v / other for k, v in self.toplevel.items()})
        # ImpulseDict[['C, 'Y']] / ss[['C', 'Y']]: matches steady states; don't divide by zero
        elif isinstance(other, SteadyStateDict):
            return type(self)({k: v / other[k] if not np.isclose(other[k], 0) else v for k, v in self.toplevel.items()})
        else:
            return NotImplementedError('An ImpulseDict can only be divided by a number or a SteadyStateDict.')

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
        return ImpulseDict(impulse)

    def infer_length(self):
        lengths = [len(v) for v in self.toplevel.values()]
        length = max(lengths)
        if length != min(lengths):
            raise ValueError(f'Building ImpulseDict with inconsistent lengths {max(lengths)} and {min(lengths)}')
        return length
