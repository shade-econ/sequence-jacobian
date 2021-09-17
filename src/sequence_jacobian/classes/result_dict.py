import copy

from ..utilities.bijection import Bijection

class ResultDict:
    def __init__(self, data, internals=None):
        if isinstance(data, ResultDict):
            if internals is not None:
                raise ValueError(f'Supplying {type(self).__name__} and also internals to constructor not allowed')
            self.toplevel = data.toplevel.copy()
            self.internals = data.internals.copy()
        else:
            self.toplevel: dict = data.copy()
            self.internals: dict = {} if internals is None else internals.copy()
        
    def __repr__(self):
        if self.internals:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}, internals={list(self.internals.keys())}>"
        else:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}>"

    def __iter__(self):
        return iter(self.toplevel)

    def __getitem__(self, k, **kwargs):
        if isinstance(k, str):
            return self.toplevel[k]
        elif isinstance(k, tuple):
            raise TypeError(f'Key {k} to {type(self).__name__} cannot be tuple')
        else:
            try:
                return type(self)({ki: self.toplevel[ki] for ki in k}, **kwargs)
            except TypeError:
                raise TypeError(f'Key {k} to {type(self).__name__} needs to be a string or an iterable (list, set, etc) of strings')
    
    def __setitem__(self, k, v):
        self.toplevel[k] = v

    def __matmul__(self, x):
        # remap keys in toplevel
        if isinstance(x, Bijection):
            new = copy.deepcopy(self)
            new.toplevel = x @ self.toplevel
            return new
        else:
            return NotImplemented
    
    def __rmatmul__(self, x):
        return self.__matmul__(x)

    def __len__(self):
        return len(self.toplevel)

    def __or__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f'Trying to merge a {type(self).__name__} with a {type(other).__name__}.')
        merged = self.copy()
        merged.update(other)
        return merged

    def keys(self):
        return self.toplevel.keys()

    def values(self):
        return self.toplevel.values()

    def items(self):
        return self.toplevel.items()

    def update(self, rdict):
        if isinstance(rdict, ResultDict):
            self.toplevel.update(rdict.toplevel)
            self.internals.update(rdict.internals)
        else:
            self.toplevel.update(dict(rdict))
    
    def copy(self):
        return type(self)(self)
