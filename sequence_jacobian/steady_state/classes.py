"""Various classes to support the computation of steady states"""

from copy import deepcopy
import numpy as np


class SteadyStateDict:
    def __init__(self, toplevel, internal):
        self.toplevel = toplevel
        self.internal = internal

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.toplevel[k]
        else:
            try:
                return {ki: self.toplevel[ki] for ki in k}
            except TypeError:
                raise TypeError(f'Key {k} needs to be a string or an iterable (list, set, etc) of strings')

    def __setitem__(self, k, v):
        self.toplevel[k] = v
