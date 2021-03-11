"""ImpulseDict class for manipulating impulse responses."""

import numpy as np


class ImpulseDict:
    def __init__(self, impulse, ss):
        if isinstance(impulse, ImpulseDict):
            self.impulse = impulse.impulse
            self.ss = impulse.ss
        else:
            if not isinstance(impulse, dict) or not isinstance(ss, dict):
                raise ValueError('ImpulseDicts are initialized with two dicts.')
            self.impulse = impulse
            self.ss = ss

    def __iter__(self):
        return iter(self.impulse.items())

    def __mul__(self, x):
        return type(self)({k: x * v for k, v in self.impulse.items()}, self.ss)

    def __rmul__(self, x):
        return type(self)({k: x * v for k, v in self.impulse.items()}, self.ss)

    def __or__(self, other):
        if not isinstance(other, ImpulseDict):
            raise ValueError('Trying to merge an ImpulseDict with something else.')
        if self.ss != other.ss:
            raise ValueError('Trying to merge ImpulseDicts with different steady states.')
        # make a copy, then add additional impulses
        merged = type(self)(self.impulse, self.ss)
        merged.impulse.update(other.impulse)
        return merged

    def __getitem__(self, x):
        # Behavior similar to pandas
        if isinstance(x, str):
            # case 1: ImpulseDict['C'] returns array
            return self.impulse[x]
        if isinstance(x, list):
            # case 2: ImpulseDict[['C']] or ImpulseDict[['C', 'Y']] return smaller ImpulseDicts
            return type(self)({k: self.impulse[k] for k in x}, self.ss)

    def normalize(self, x=None):
        if x is None:
            # default: normalize by steady state if not zero
            impulse = {k: v/self.ss[k] if not np.isclose(self.ss[k], 0) else v for k, v in self.impulse.items()}
        else:
            # normalize by steady state of x
            if x not in self.ss.keys():
                raise ValueError(f'Cannot normalize with {x}: steady state is unknown.')
            elif np.isclose(self.ss[x], 0):
                raise ValueError(f'Cannot normalize with {x}: steady state is zero.')
            else:
                impulse = {k: v / self.ss[x] for k, v in self.impulse.items()}
        return type(self)(impulse, self.ss)

    def levels(self):
        return type(self)({k: v + self.ss[k] for k, v in self.impulse.items()}, self.ss)

    def deviations(self):
        return type(self)({k: v - self.ss[k] for k, v in self.impulse.items()}, self.ss)
