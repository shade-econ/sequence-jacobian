"""Topological sort and related code"""


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
    #dep, revdep = complete_reverse_graph(dep)
    dep, revdep = revadj, adj
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
            inset = inmap.setdefault(i, set())
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
        current_adj = set()
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
        current_revadj = set()
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
    required = set()
    outmap = get_output_map(blocks)
    for num, block in enumerate(blocks):
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = set(i for o in block for i in block[o])
        for i in inputs:
            if i in outmap:
                required.add(i)
    return required


def block_sort_w_helpers(blocks, helper_blocks=None, calibration=None, return_io=False):
    """Given list of blocks (either blocks themselves or dicts of Jacobians), find a topological sort.

    Relies on blocks having 'inputs' and 'outputs' attributes (unless they are dicts of Jacobians, in which case it's
    inferred) that indicate their aggregate inputs and outputs

    Importantly, because including helper blocks in a blocks without additional measures
    can introduce cycles within the DAG, allow the user to provide the calibration that will be used in the
    steady_state computation to resolve these cycles.
    e.g. Consider Krusell Smith:
    Suppose one specifies a helper block based on a calibrated value for "r", which outputs "K" (among other vars).
    Normally block_sort would include the "firm" block as a dependency of the helper block
    because the "firm" block outputs "r", which the helper block takes as an input.
    However, it would also include the helper block as a dependency of the "firm" block because the "firm" block takes
    "K" as an input.
    This would result in a cycle. However, if a "calibration" is provided in which "r" is included, then
    "firm" could be removed as a dependency of helper block and the cycle would be resolved.

    blocks: `list`
        A list of the blocks (SimpleBlock, HetBlock, etc.) to sort
    helper_blocks: `list`
        A list of helper blocks
    calibration: `dict` or `None`
        An optional dict of variable/parameter names and their pre-specified values to help resolve any cycles
        introduced by using helper blocks. Read above docstring for more detail
    return_io: `bool`
        A boolean indicating whether to return the full set of input and output arguments from `blocks`
    """
    if return_io:
        # step 1: map outputs to blocks for topological sort
        outmap, outargs = construct_output_map_w_helpers(blocks, return_output_args=True,
                                                         helper_blocks=helper_blocks, calibration=calibration)

        # step 2: dependency graph for topological sort and input list
        dep, inargs = construct_dependency_graph_w_helpers(blocks, outmap, return_input_args=True, outargs=outargs,
                                                           helper_blocks=helper_blocks, calibration=calibration)

        return topological_sort(map_to_list(complete_reverse_graph(dep)), map_to_list(dep)), inargs, outargs
    else:
        # step 1: map outputs to blocks for topological sort
        outmap = construct_output_map_w_helpers(blocks, helper_blocks=helper_blocks)

        # step 2: dependency graph for topological sort and input list
        dep = construct_dependency_graph_w_helpers(blocks, outmap, helper_blocks=helper_blocks, calibration=calibration)

        return topological_sort(map_to_list(complete_reverse_graph(dep)), map_to_list(dep))


def map_to_list(m):
    return [m[i] for i in range(len(m))]


def construct_output_map_w_helpers(blocks, helper_blocks=None, calibration=None, return_output_args=False):
    """Mirroring construct_output_map functionality in utilities.graph module but augmented to support
    helper blocks"""
    if calibration is None:
        calibration = {}
    if helper_blocks is None:
        helper_blocks = []

    helper_inputs = set().union(*[block.inputs for block in helper_blocks])

    outmap = dict()
    outargs = set()
    for num, block in enumerate(blocks):
        # Find the relevant set of outputs corresponding to a block
        if hasattr(block, "outputs"):
            outputs = block.outputs
        elif isinstance(block, dict):
            outputs = block.keys()
        else:
            raise ValueError(f'{block} is not recognized as block or does not provide outputs')

        for o in outputs:
            # Because some of the outputs of a helper block are, by construction, outputs that also appear in the
            # standard blocks that comprise a DAG, ignore the fact that an output is repeated when considering
            # throwing this ValueError
            if o in outmap and block not in helper_blocks:
                raise ValueError(f'{o} is output twice')

            # Priority sorting for standard blocks:
            # Ensure that the block "outmap" maps "o" to is the actual block and not a helper block if both share
            # a given output, such that the dependency graph is constructed on the standard blocks, where possible
            if o not in outmap:
                outmap[o] = num
                if return_output_args and not (o in helper_inputs and o in calibration):
                    outargs.add(o)
            else:
                continue
    if return_output_args:
        return outmap, outargs
    else:
        return outmap


def construct_dependency_graph_w_helpers(blocks, outmap, helper_blocks=None,
                                         calibration=None, return_input_args=False, outargs=None):
    """Mirroring construct_dependency_graph functionality in utilities.graph module but augmented to support
    helper blocks"""
    if calibration is None:
        calibration = {}
    if helper_blocks is None:
        helper_blocks = []
    if outargs is None:
        outargs = {}

    dep = {num: set() for num in range(len(blocks))}
    inargs = set()
    for num, block in enumerate(blocks):
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = set(i for o in block for i in block[o])
        for i in inputs:
            # Each potential input to a given block will either be 1) output by another block,
            # 2) an unknown or exogenous variable, or 3) a pre-specified variable/parameter passed into
            # the steady-state computation via the `calibration' dict.
            # If the block is a helper block, then we want to check the calibration to see if the potential
            # input is a pre-specified variable/parameter, and if it is then we will not add the block that
            # produces that input as an output as a dependency.
            # e.g. Krusell Smith's firm_steady_state_solution helper block and firm block would create a cyclic
            # dependency, if it were not for this resolution.
            if i in outmap and not (i in calibration and block in helper_blocks):
                dep[num].add(outmap[i])
            if return_input_args and not (i in outargs):
                inargs.add(i)
    if return_input_args:
        return dep, inargs
    else:
        return dep


def find_intermediate_inputs_w_helpers(blocks, helper_blocks=None, **kwargs):
    """Mirroring find_outputs_that_are_intermediate_inputs functionality in utilities.graph module
    but augmented to support helper blocks"""
    required = set()
    outmap = construct_output_map_w_helpers(blocks, helper_blocks=helper_blocks, **kwargs)
    for num, block in enumerate(blocks):
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = set(i for o in block for i in block[o])
        for i in inputs:
            if i in outmap:
                required.add(i)
    return required


def complete_reverse_graph(gph):
    """Given directed graph represented as a dict from nodes to iterables of nodes, return representation of graph that
    is complete (i.e. has each vertex pointing to some iterable, even if empty), and a complete version of reversed too.
    Have returns be sets, for easy removal"""

    revgph = {n: set() for n in gph}
    for n, e in gph.items():
        for n2 in e:
            n2_edges = revgph.setdefault(n2, set())
            n2_edges.add(n)

    return revgph


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
