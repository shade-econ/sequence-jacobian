"""Various classes to support the computation of steady states"""

from copy import deepcopy
import numpy as np

from ..utilities.misc import dict_diff


class SteadyStateDict:
    def __init__(self, data, internal=None):
        if isinstance(data, SteadyStateDict):
            self.toplevel = deepcopy(data.toplevel)
            self.internal = deepcopy(data.internal)
        else:
            self.toplevel = data
            self.internal = internal if internal is not None else {}

    def __repr__(self):
        if self.internal:
            return f"{type(self).__name__}: {list(self.toplevel.keys())}, internal={list(self.internal.keys())}"
        else:
            return f"{type(self).__name__}: {list(self.toplevel.keys())}"

    def __iter__(self):
        return iter(self.toplevel)

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

    def keys(self):
        return self.toplevel.keys()

    def values(self):
        return self.toplevel.values()

    def update(self, new_data):
        if isinstance(new_data, SteadyStateDict):
            self.toplevel.update(new_data.toplevel)
            self.internal.update(new_data.internal)
        else:
            # TODO: This is assuming new_data only contains aggregates. Upgrade in later commit to handle the case of
            #   vector-valued variables/collection into internal namespaces
            self.toplevel.update(new_data)

    def difference(self, data_to_remove):
        return SteadyStateDict(dict_diff(self.toplevel, data_to_remove), internal=deepcopy(self.internal))
