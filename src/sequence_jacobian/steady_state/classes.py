"""Various classes to support the computation of steady states"""

from copy import deepcopy

from ..utilities.misc import dict_diff
from ..blocks.support.bijection import Bijection

from numbers import Real
from typing import Any, Dict, Union
Array = Any

class SteadyStateDict:
    # TODO: should this just subclass dict so we can avoid a lot of boilerplate?
    # Really this is just a top-level dict (with all the usual functionality) with "internal" bolted on

    def __init__(self, data, internal=None):
        if isinstance(data, SteadyStateDict):
            if internal is not None:
                raise ValueError('Supplying SteadyStateDict and also internal to constructor not allowed')
            self.toplevel = data
            self.internal = {}
        
        self.toplevel: dict = data
        self.internal: dict = {} if internal is None else internal

    def __repr__(self):
        if self.internal:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}, internal={list(self.internal.keys())}>"
        else:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}>"

    def __iter__(self):
        return iter(self.toplevel)

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.toplevel[k]
        else:
            try:
                return SteadyStateDict({ki: self.toplevel[ki] for ki in k})
            except TypeError:
                raise TypeError(f'Key {k} needs to be a string or an iterable (list, set, etc) of strings')

    def __setitem__(self, k, v):
        self.toplevel[k] = v

    def __matmul__(self, x):
        # remap keys in toplevel
        if isinstance(x, Bijection):
            new = deepcopy(self)
            new.toplevel = x @ self.toplevel
            return new
        else:
            return NotImplemented

    def __rmatmul__(self, x):
        return self.__matmul__(x)

    def __len__(self):
        return len(self.toplevel)

    def keys(self):
        return self.toplevel.keys()

    def values(self):
        return self.toplevel.values()

    def items(self):
        return self.toplevel.items()

    def update(self, ssdict):
        if isinstance(ssdict, SteadyStateDict):
            self.toplevel.update(ssdict.toplevel)
            self.internal.update(ssdict.internal)
        else:
            self.toplevel.update(dict(ssdict))

    def difference(self, data_to_remove):
        return SteadyStateDict(dict_diff(self.toplevel, data_to_remove), deepcopy(self.internal))

UserProvidedSS = Dict[str, Union[Real, Array]]
