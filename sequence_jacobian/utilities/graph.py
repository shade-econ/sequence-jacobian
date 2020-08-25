"""Topological sort and related code"""

from ..blocks.helper_block import HelperBlock


def block_sort(block_list, ignore_helpers=False, calibration=None):
    """Given list of blocks (either blocks themselves or dicts of Jacobians), find a topological sort.

    Relies on blocks having 'inputs' and 'outputs' attributes (unless they are dicts of Jacobians, in which case it's
    inferred) that indicate their aggregate inputs and outputs

    Importantly, because including HelperBlocks in a block_list without additional measures
    can introduce cycles within the DAG, allow the user to provide the calibration that will be used in the
    steady_state computation to resolve these cycles.
    e.g. Consider Krusell Smith:
    Suppose one specifies a HelperBlock based on a calibrated value for "r", which outputs "K" (among other vars).
    Normally block_sort would include the "firm" block as a dependency of the HelperBlock
    because the "firm" block outputs "r", which the HelperBlock takes as an input.
    However, it would also include the HelperBlock as a dependency of the "firm" block because the "firm" block takes
    "K" as an input.
    This would result in a cycle. However, if a "calibration" is provided in which "r" is included, then
    "firm" could be removed as a dependency of HelperBlock and the cycle would be resolved.

    block_list: `list`
        A list of the blocks (SimpleBlock, HetBlock, HelperBlock, etc.) to sort
    ignore_helpers: `bool`
        A boolean indicating whether to account for/return the indices of HelperBlocks contained in block_list
        Set to true when sorting for td and jac calculations
    calibration: `dict` or `None`
        An optional dict of variable/parameter names and their pre-specified values to help resolve any cycles
        introduced by using HelperBlocks. Read above docstring for more detail
    """

    # step 1: map outputs to blocks for topological sort
    outmap = construct_output_map(block_list, calibration=calibration)

    # step 2: dependency graph for topological sort and input list
    dep = construct_dependency_graph(block_list, outmap, ignore_helpers=ignore_helpers)
    if ignore_helpers:
        return ignore_helper_block_indices(topological_sort(dep), block_list)
    else:
        return topological_sort(dep)


def topological_sort(dep, names=None):
    """Given directed graph pointing from each node to the nodes it depends on, topologically sort nodes"""

    # get complete set version of dep, and its reversal, and build initial stack of nodes with no dependencies
    dep, revdep = complete_reverse_graph(dep)
    nodeps = [n for n in dep if not dep[n]]
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


def ignore_helper_block_indices(topsorted, blocks):
    return [i for i in topsorted if not isinstance(blocks[i], HelperBlock)]


def construct_output_map(block_list, ignore_helpers=False, calibration=None):
    """Construct a map of outputs to the indices of the blocks that produce them.

    block_list: `list`
        A list of the blocks (SimpleBlock, HetBlock, HelperBlock, etc.) to sort
    ignore_helpers: `bool`
        A boolean indicating whether to account for/return the indices of HelperBlocks contained in block_list
        Set to true when sorting for td and jac calculations
    calibration: `dict` or `None`
        An optional dict of variable/parameter names and their pre-specified values to help resolve any cycles
        introduced by using HelperBlocks. Read above docstring for more detail
    """
    outmap = dict()
    for num, block in enumerate(block_list):
        if ignore_helpers and isinstance(block, HelperBlock):
            continue

        # Find the relevant set of outputs corresponding to a block
        # TODO: This is temporary to force the DAG to account for heterogeneous outputs (e.g. the individual
        #   household policy functions). Later just generalize those to always be accounted for as potential
        #   objects to be passed around in the DAG
        if hasattr(block, 'all_outputs_order'):
            # Include the backward iteration variable, individual household policy functions, and aggregate equivalents
            # when adding a HetBlock's outputs to the output map
            outputs = set(block.all_outputs_order) | set({k.upper() for k in block.non_back_outputs})
        elif hasattr(block, 'outputs'):
            outputs = block.outputs
        elif isinstance(block, dict):
            outputs = block.keys()
        else:
            raise ValueError(f'{block} is not recognized as block or does not provide outputs')

        for o in outputs:
            # If a block's output is also present in the provided calibration, then it is not required for the
            # construction of a dependency graph and hence we omit it
            if calibration is not None and o in calibration:
                continue

            # Because some of the outputs of a HelperBlock are, by construction, outputs that also appear in the
            # standard blocks that comprise a DAG, ignore the fact that an output is repeated when considering
            # throwing this ValueError
            if o in outmap and not (isinstance(block, HelperBlock) or isinstance(block_list[outmap[o]], HelperBlock)):
                raise ValueError(f'{o} is output twice')

            # Ensure that the block "outmap" maps "o" to is the actual block and not a HelperBlock if both share
            # a given output, such that the dependency graph is constructed on the standard blocks, where possible
            if o not in outmap or (o in outmap and not isinstance(block, HelperBlock)):
                outmap[o] = num
            else:
                continue
    return outmap


def construct_dependency_graph(block_list, outmap, ignore_helpers=False):
    """Construct a dependency graph dictionary, with block indices as keys and a set of block indices as values, where
    this set is the set of blocks that the key block is dependent on.

    outmap is the output map (output to block index mapping) created by construct_output_map.

    See the docstring of block_sort for more details about the other arguments.
    """
    dep = {num: set() for num in range(len(block_list))}
    for num, block in enumerate(block_list):
        if ignore_helpers and isinstance(block, HelperBlock):
            continue
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = set(i for o in block for i in block[o])
        for i in inputs:
            if i in outmap:
                dep[num].add(outmap[i])
    return dep


def find_outputs_that_are_intermediate_inputs(block_list, ignore_helpers=False, calibration=None):
    """Find outputs of the blocks in block_list that are inputs to other blocks in block_list.
    This is useful to ensure that all of the relevant curlyJ Jacobians (of all inputs to all outputs) are computed.

    See the docstring of construct_output_map for more details about the arguments.
    """
    required = set()
    outmap = construct_output_map(block_list, ignore_helpers=ignore_helpers, calibration=calibration)
    for block in block_list:
        if ignore_helpers and isinstance(block, HelperBlock):
            continue
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

    gph_missing_n = revgph.keys() - gph.keys()
    gph = {**{k: set(v) for k, v in gph.items()}, **{n: set() for n in gph_missing_n}}
    return gph, revgph


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


