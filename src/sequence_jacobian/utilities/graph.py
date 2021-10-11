"""Topological sort and related code"""
from .ordered_set import OrderedSet
from .bijection import Bijection

class DAG:
    """Represents "blocks" that each have inputs and outputs, where output-input relationships between
    blocks form a DAG. Fundamental DAG object intended to underlie CombinedBlock and CombinedExtendedFunction.
    
    Initialized with list of blocks, which are then topologically sorted"""
    
    def __init__(self, blocks):
        inmap = get_input_map(blocks)
        outmap = get_output_map(blocks)
        adj = get_block_adjacency_list(blocks, inmap)
        revadj = get_block_reverse_adjacency_list(blocks, outmap)
        topsort = topological_sort(adj, revadj)

        M = Bijection({i: t for i, t in enumerate(topsort)})

        self.blocks = [blocks[t] for t in topsort]
        self.inmap = {k: M @ v for k, v in inmap.items()}
        self.outmap = {k: M @ v for k, v in outmap.items()}
        self.adj = [M @ adj[t] for t in topsort]
        self.revadj = [M @ revadj[t] for t in topsort]

        self.inputs = OrderedSet(k for k in inmap if k not in outmap)
        self.outputs = OrderedSet(outmap)


    def visit_from_inputs(self, inputs):
        """Which block numbers are ultimately dependencies of 'inputs'?"""
        inputs = inputs & self.inputs
        visited = OrderedSet()
        for n, (block, parentset) in enumerate(zip(self.blocks, self.revadj)):
            # first see if block has its input directly changed
            for i in inputs:
                if i in block.inputs:
                    visited.add(n)
                    break
            else:
                if not parentset.isdisjoint(visited):
                    visited.add(n)

        return visited

    def visit_from_outputs(self, outputs):
        """Which block numbers are 'outputs' ultimately dependent on?"""
        outputs = outputs & self.outputs
        visited = OrderedSet()
        for n in reversed(range(len(self.blocks))):
            block = self.blocks[n]
            childset = self.adj[n]

            # first see if block has its output directly used
            for o in outputs:
                if o in block.outputs:
                    visited.add(n)
                    break
            else:
                if not childset.isdisjoint(visited):
                    visited.add(n)

        return reversed(visited)


def block_sort(blocks):
    """Given list of blocks (either blocks themselves or dicts of Jacobians), find a topological sort.

    Relies on blocks having 'inputs' and 'outputs' attributes (unless they are dicts of Jacobians, in which case it's
    inferred) that indicate their aggregate inputs and outputs

    blocks: `list`
        A list of the blocks (SimpleBlock, HetBlock, etc.) to sort
    """
    inmap = get_input_map(blocks)
    outmap = get_output_map(blocks)
    adj = get_block_adjacency_list(blocks, inmap)
    revadj = get_block_reverse_adjacency_list(blocks, outmap)
    return topological_sort(adj, revadj)


def topological_sort(adj, revadj, names=None):
    """Given directed graph pointing from each node to the nodes it depends on, topologically sort nodes"""
    # get complete set version of dep, and its reversal, and build initial stack of nodes with no dependencies
    revdep = adj
    dep = [s.copy() for s in revadj]
    nodeps = [n for n, depset in enumerate(dep) if not depset]
    topsorted = []

    # Kahn's algorithm: find something with no dependency, delete its edges and update
    while nodeps:
        n = nodeps.pop()
        topsorted.append(n)
        for n2 in revdep[n]:
            dep[n2].remove(n)
            if not dep[n2]:
                nodeps.append(n2)

    # should be done: topsorted should be topologically sorted with same # of elements as original graphs!
    if len(topsorted) != len(dep):
        cycle_ints = find_cycle(dep, dep.keys() - set(topsorted))
        assert cycle_ints is not None, 'topological sort failed but no cycle, THIS SHOULD NEVER EVER HAPPEN'
        cycle = [names[i] for i in cycle_ints] if names else cycle_ints
        raise Exception(f'Topological sort failed: cyclic dependency {" -> ".join([str(n) for n in cycle])}')

    return topsorted


def get_input_map(blocks: list):
    """inmap[i] gives set of block numbers where i is an input"""
    inmap = dict()
    for num, block in enumerate(blocks):
        for i in block.inputs:
            inset = inmap.setdefault(i, OrderedSet())
            inset.add(num)

    return inmap


def get_output_map(blocks: list):
    """outmap[o] gives unique block number where o is an output"""
    outmap = dict()
    for num, block in enumerate(blocks):
        for o in block.outputs:
            if o in outmap:
                raise ValueError(f'{o} is output twice')
            outmap[o] = num

    return outmap


def get_block_adjacency_list(blocks, inmap):
    """adj[n] for block number n gives set of block numbers which this block points to"""
    adj = []
    for block in blocks:
        current_adj = OrderedSet()
        for o in block.outputs:
            # for each output, if that output is used as an input by some blocks, add those blocks to adj
            if o in inmap:
                current_adj |= inmap[o]
        adj.append(current_adj)
    return adj


def get_block_reverse_adjacency_list(blocks, outmap):
    """revadj[n] for block number n gives set of block numbers that point to this block"""
    revadj = []
    for block in blocks:
        current_revadj = OrderedSet()
        for i in block.inputs:
            if i in outmap:
                current_revadj.add(outmap[i])
        revadj.append(current_revadj)
    return revadj


def find_intermediate_inputs(blocks):
    # TODO: should be deprecated
    """Find outputs of the blocks in blocks that are inputs to other blocks in blocks.
    This is useful to ensure that all of the relevant curlyJ Jacobians (of all inputs to all outputs) are computed.
    """
    required = OrderedSet()
    outmap = get_output_map(blocks)
    for num, block in enumerate(blocks):
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = OrderedSet(i for o in block for i in block[o])
        for i in inputs:
            if i in outmap:
                required.add(i)
    return required


def find_cycle(dep, onlyset=None):
    """Return list giving cycle if there is one, otherwise None"""

    # supposed to look only within 'onlyset', so filter out everything else
    if onlyset is not None:
        dep = {k: (set(v) & set(onlyset)) for k, v in dep.items() if k in onlyset}

    tovisit = set(dep.keys())
    stack = SetStack()
    while tovisit or stack:
        if stack:
            # if stack has something, still need to proceed with DFS
            n = stack.top()
            if dep[n]:
                # if there are any dependencies left, let's look at them
                n2 = dep[n].pop()
                if n2 in stack:
                    # we have a cycle, since this is already in our stack
                    i2loc = stack.index(n2)
                    return stack[i2loc:] + [stack[i2loc]]
                else:
                    # no cycle, visit this node only if we haven't already visited it
                    if n2 in tovisit:
                        tovisit.remove(n2)
                        stack.add(n2)
            else:
                # if no dependencies left, then we're done with this node, so let's forget about it
                stack.pop(n)
        else:
            # nothing left on stack, let's start the DFS from something new
            n = tovisit.pop()
            stack.add(n)

    # if we never find a cycle, we're done
    return None


class SetStack:
    """Stack implemented with list but tests membership with set to be efficient in big cases"""

    def __init__(self):
        self.myset = set()
        self.mylist = []

    def add(self, x):
        self.myset.add(x)
        self.mylist.append(x)

    def pop(self):
        x = self.mylist.pop()
        self.myset.remove(x)
        return x

    def top(self):
        return self.mylist[-1]

    def index(self, x):
        return self.mylist.index(x)

    def __contains__(self, x):
        return x in self.myset

    def __len__(self):
        return len(self.mylist)

    def __getitem__(self, i):
        return self.mylist.__getitem__(i)

    def __repr__(self):
        return self.mylist.__repr__()
