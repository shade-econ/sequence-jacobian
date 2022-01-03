from typing import Iterable

class OrderedSet:
    """Ordered set implemented as dict (where key insertion order is preserved) mapping all to None.
    
    Operations on multiple ordered sets (e.g. union) order all members of first argument first, then
    second argument. If a member is in both, order is as early as possible.
    
    See test_misc_support.test_ordered_set() for examples."""

    def __init__(self, members: Iterable = []):
        self.d = {k: None for k in members}

    def dict_from(self, s):
        return dict(zip(self, s))

    def __iter__(self):
        return iter(self.d)

    def __reversed__(self):
        return OrderedSet(list(self)[::-1])

    def __repr__(self):
        return f"OrderedSet({list(self)})"

    def __str__(self):
        return str(list(self.d))

    def __contains__(self, k):
        return k in self.d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return list(self.d)[i]

    def add(self, x):
        self.d[x] = None
    
    def difference(self, s):
        return OrderedSet(k for k in self if k not in s)

    def difference_update(self, s):
        self.d = self.difference(s).d
        return self

    def discard(self, k):
        self.d.pop(k, None)
    
    def intersection(self, s):
        return OrderedSet(k for k in self if k in s)

    def intersection_update(self, s):
        self.d = self.intersection(s).d
        return self

    def isdisjoint(self, s):
        return len(self.intersection(s)) == 0
    
    def issubset(self, s):
        return len(self.difference(s)) == 0

    def issuperset(self, s):
        return len(self.intersection(s)) == len(s)
    
    def remove(self, k):
        self.d.pop(k)

    def symmetric_difference(self, s):
        diff = self.difference(s)
        for k in s:
            if k not in self:
                diff.add(k)
        return diff

    def symmetric_difference_update(self, s):
        self.d = self.symmetric_difference(s).d
        return self

    def union(self, s):
        return self.copy().update(s)
    
    def update(self, s):
        for k in s:
            self.add(k)
        return self
    
    def copy(self):
        return OrderedSet(self)

    def __eq__(self, s):
        if isinstance(s, OrderedSet):
            return list(self) == list(s)
        else:
            return False
    
    def __le__(self, s):
        return self.issubset(s)
    
    def __lt__(self, s):
        return self.issubset(s) and (len(self) != len(s))

    def __ge__(self, s):
        return self.issuperset(s)

    def __gt__(self, s):
        return self.issuperset(s) and (len(self) != len(s))

    def __or__(self, s):
        return self.union(s)

    def __ior__(self, s):
        return self.update(s)

    def __ror__(self, s):
        return self.union(s)

    def __and__(self, s):
        return self.intersection(s)
    
    def __iand__(self, s):
        return self.intersection_update(s)

    def __rand__(self, s):
        return self.intersection(s)

    def __sub__(self, s):
        return self.difference(s)

    def __isub__(self, s):
        return self.difference_update(s)

    def __rsub__(self, s):
        return OrderedSet(s).difference(self)

    def __xor__(self, s):
        return self.symmetric_difference(s)

    def __ixor__(self, s):
        return self.symmetric_difference_update(s)

    def __rxor__(self, s):
        return OrderedSet(s).symmetric_difference(self)

    """Compatibility methods, regular use not advised"""

    def pop(self):
        k = self.top()
        del self.d[k]
        return k

    def top(self):
        return list(self.d)[-1]

    def index(self, k):
        return list(self.d).index(k)
