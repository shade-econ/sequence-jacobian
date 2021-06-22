class Bijection:
    def __init__(self, map):
        # identity always implicit, remove if there explicitly
        self.map = {k: v for k, v in map.items() if k != v}
        invmap = {}
        for k, v in map.items():
            if v in invmap:
                raise ValueError(f'Duplicate value {v}, for keys {invmap[v]} and {k}')
            invmap[v] = k
        self.invmap = invmap
    
    @property
    def inv(self):
        invmap = Bijection.__new__(Bijection)  # better way to do this?
        invmap.map = self.invmap
        invmap.invmap = self.map
        return invmap

    def __repr__(self):
        return f'Bijection({repr(self.map)})'

    def __getitem__(self, k):
        return self.map.get(k, k)

    def __matmul__(self, x):
        if isinstance(x, Bijection):
            # compose self: v -> u with x: w -> v
            # assume everything missing in either is the identity
            M = {}
            for v, u in self.map.items():
                w = x.invmap.get(v, v)
                M[w] = u
            for w, v in x.map.items():
                if v not in self.map:
                    M[w] = v
            return Bijection(M)
        elif isinstance(x, dict):
            return {self[k]: v for k, v in x.items()}
        elif isinstance(x, list):
            return [self[k] for k in x]
        elif isinstance(x, set):
            return {self[k] for k in x}
        elif isinstance(x, tuple):
            return tuple(self[k] for k in x)
        else:
            return NotImplemented

    def __rmatmul__(self, x):
        if isinstance(x, dict):
            return {self[k]: v for k, v in x.items()}
        elif isinstance(x, list):
            return [self[k] for k in x]
        elif isinstance(x, set):
            return {self[k] for k in x}
        elif isinstance(x, tuple):
            return tuple(self[k] for k in x)
        else:
            return NotImplemented