class Parent:
    # see tests in test_parent_block.py

    def __init__(self, blocks, name=None):
        # dict from names to immediate kid blocks themselves
        # dict from descendants to the names of kid blocks through which to access them
        # "descendants" of a block include itself 
        if not hasattr(self, 'name') and name is not None:
            self.name = name

        kids = {}
        descendants = {}

        for block in blocks:
            kids[block.name] = block

            if isinstance(block, Parent):
                for k in block.descendants:
                    if k in descendants:
                        raise ValueError(f'Overlapping block name {k}')
                    descendants[k] = block.name
            else:
                descendants[block.name] = block.name

        # add yourself to descendants too! but you don't belong to any kid...
        if self.name in descendants:
            raise ValueError(f'Overlapping block name {self.name}')
        descendants[self.name] = None

        self.kids = kids
        self.descendants = descendants
    
    def __getitem__(self, k):
        if k == self.name:
            return self
        elif k in self.kids:
            return self.kids[k]
        else:
            return self.kids[self.descendants[k]][k]

    def select(self, d, kid):
        """If d is a dict with block names as keys and kid is a kid, select only the entries in d that are descendants of kid"""
        return {k: v for k, v in d.items() if k in self.kids[kid].descendants}

    def path(self, k, reverse=True):
        if k not in self.descendants:
            raise KeyError(f'Cannot get path to {k} because it is not a descendant of current block')
        
        if k != self.name:
            kid = self.kids[self.descendants[k]]
            if isinstance(kid, Parent):
                p = kid.path(k, reverse=False)
            else:
                p = [k]
        else:
            p = []
        p.append(self.name)

        if reverse:
            return list(reversed(p))
        else:
            return p

    def get_attribute(self, k, attr):
        """Gets attribute attr from descendant k, respecting any remapping
        along the way (requires that attr is list, dict, set)"""
        if k == self.name:
            inner = getattr(self, attr)
        else:
            kid = self.kids[self.descendants[k]]
            if isinstance(kid, Parent):
                inner = kid.get_attribute(k, attr)
            else:
                inner = getattr(kid, attr)
                if hasattr(kid, 'M'):
                    inner = kid.M @ inner

        if hasattr(self, 'M'):
            return self.M @ inner
        else:
            return inner
