"""ImpulseDict class for manipulating impulse responses."""

import numpy as np
from copy import deepcopy

from ...steady_state.classes import SteadyStateDict
from .bijection import Bijection


class ImpulseDict:
    def __init__(self, impulse):
        if isinstance(impulse, ImpulseDict):
            self.impulse = impulse.impulse
        else:
            if not isinstance(impulse, dict):
                raise ValueError('ImpulseDicts are initialized with a `dict` of impulse responses.')
            self.impulse = impulse

    def __repr__(self):
        return f'<ImpulseDict: {list(self.impulse.keys())}>'

    def __iter__(self):
        return iter(self.impulse.items())

    def __or__(self, other):
        if not isinstance(other, ImpulseDict):
            raise ValueError('Trying to merge an ImpulseDict with something else.')
        # Union returns a new ImpulseDict
        merged = type(self)(self.impulse)
        merged.impulse.update(other.impulse)
        return merged

    def __getitem__(self, item):
        # Behavior similar to pandas
        if isinstance(item, str):
            # Case 1: ImpulseDict['C'] returns array
            return self.impulse[item]
        if isinstance(item, list):
            # Case 2: ImpulseDict[['C']] or ImpulseDict[['C', 'Y']] return smaller ImpulseDicts
            return type(self)({k: self.impulse[k] for k in item})

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v + other for k, v in self.impulse.items()})
        if isinstance(other, SteadyStateDict):
            return type(self)({k: v + other[k] for k, v in self.impulse.items()})

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v - other for k, v in self.impulse.items()})
        if isinstance(other, (SteadyStateDict, ImpulseDict)):
            return type(self)({k: v - other[k] for k, v in self.impulse.items()})

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v * other for k, v in self.impulse.items()})
        if isinstance(other, SteadyStateDict):
            return type(self)({k: v * other[k] for k, v in self.impulse.items()})

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v * other for k, v in self.impulse.items()})
        if isinstance(other, SteadyStateDict):
            return type(self)({k: v * other[k] for k, v in self.impulse.items()})

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return type(self)({k: v / other for k, v in self.impulse.items()})
        # ImpulseDict[['C, 'Y']] / ss[['C', 'Y']]: matches steady states; don't divide by zero
        if isinstance(other, SteadyStateDict):
            return type(self)({k: v / other[k] if not np.isclose(other[k], 0) else v for k, v in self.impulse.items()})

    def __matmul__(self, x):
        # remap keys in toplevel
        if isinstance(x, Bijection):
            new = deepcopy(self)
            new.impulse = x @ self.impulse
            return new

    def __rmatmul__(self, x):
        return self.__matmul__(x)
