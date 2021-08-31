"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from ..primitives import Block
from ..utilities.misc import dict_diff
from ..utilities.graph import block_sort, find_intermediate_inputs
from ..utilities.graph import topological_sort
from ..utilities.ordered_set import OrderedSet
from ..blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from ..blocks.parent import Parent
from ..steady_state.support import subset_helper_block_unknowns
from ..jacobian.classes import JacobianDict


def combine(blocks, name="", model_alias=False):
    return CombinedBlock(blocks, name=name, model_alias=model_alias)


# Useful functional alias
def create_model(blocks, **kwargs):
    return combine(blocks, model_alias=True, **kwargs)


class CombinedBlock(Block, Parent):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, blocks, name="", model_alias=False, sorted_indices=None, intermediate_inputs=None):
        super().__init__()

        self._blocks_unsorted = [b if isinstance(b, Block) else JacobianDictBlock(b) for b in blocks]
        self._sorted_indices = block_sort(blocks) if sorted_indices is None else sorted_indices
        self._required = find_intermediate_inputs(blocks) if intermediate_inputs is None else intermediate_inputs
        self.blocks = [self._blocks_unsorted[i] for i in self._sorted_indices]

        if not name:
            self.name = f"{self.blocks[0].name}_to_{self.blocks[-1].name}_combined"
        else:
            self.name = name

        # now that it has a name, do Parent initialization
        Parent.__init__(self, blocks)

        # Find all outputs (including those used as intermediary inputs)
        self.outputs = set().union(*[block.outputs for block in self.blocks])

        # Find all inputs that are *not* intermediary outputs
        all_inputs = set().union(*[block.inputs for block in self.blocks])
        self.inputs = all_inputs.difference(self.outputs)

        # If the create_model() is used instead of combine(), we will have __repr__ show this object as a 'Model'
        self._model_alias = model_alias

    def __repr__(self):
        if self._model_alias:
            return f"<Model '{self.name}'>"
        else:
            return f"<CombinedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""

        ss = deepcopy(calibration)
        for block in self.blocks:
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **kwargs)
            ss.update(outputs)

        return ss

    def _impulse_nonlinear(self, ss, inputs, outputs, Js):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()

        irf_nonlin_partial_eq = deepcopy(inputs)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_nonlin_partial_eq.update(block.impulse_nonlinear(ss, input_args, outputs & block.outputs, Js))

        return irf_nonlin_partial_eq[original_outputs]

    def _impulse_linear(self, ss, inputs, outputs, Js):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()
        
        irf_lin_partial_eq = deepcopy(inputs)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs} 

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update(block.impulse_linear(ss, input_args, outputs & block.outputs, Js))

        return irf_lin_partial_eq[original_outputs]

    def _partial_jacobians(self, ss, inputs, outputs, T, Js):
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued

        curlyJs = {}
        for block in self.blocks:
            descendants = block.descendants if isinstance(block, Parent) else {block.name: None}
            Js_block = {k: v for k, v in Js.items() if k in descendants}

            curlyJ = block.partial_jacobians(ss, inputs & block.inputs, outputs & block.outputs, T, Js_block)
            curlyJs.update(curlyJ)
            
        return curlyJs

    def _jacobian(self, ss, inputs, outputs, T, Js={}):
        Js = self._partial_jacobians(ss, inputs, outputs, T=T, Js=Js)

        original_outputs = outputs
        total_Js = JacobianDict.identity(inputs)

        # TODO: horrible, redoing work from partial_jacobians, also need more efficient sifting of intermediates!
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued
        for block in self.blocks:
            descendants = block.descendants if isinstance(block, Parent) else {block.name: None}
            Js_block = {k: v for k, v in Js.items() if k in descendants}
            J = block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T, Js_block)
            total_Js.update(J @ total_Js)

        return total_Js[original_outputs, :]

    def solve_steady_state(self, calibration, unknowns, targets, solver="", helper_blocks=None, helper_targets=None,
                           **kwargs):
        if helper_blocks is not None:
            if helper_targets is None:
                raise ValueError("Must provide the dict of targets and their values that the `helper_blocks` solve"
                                 " in the `helper_targets` keyword argument.")
            else:
                targets = {t: 0. for t in targets} if isinstance(targets, list) else targets
                helper_targets = {t: targets[t] for t in helper_targets} if isinstance(helper_targets, list) else helper_targets
                helper_unknowns = subset_helper_block_unknowns(unknowns, helper_blocks, helper_targets)

                calibration_w_help = calibration.copy()
                calibration_w_help.update(helper_targets)
                ss_calibration_block = SSCalibrationBlock(self.blocks + helper_blocks, helper_blocks=helper_blocks,
                                                          calibration=calibration_w_help)
                return ss_calibration_block.solve_steady_state(calibration_w_help, dict_diff(unknowns, helper_unknowns),
                                                               dict_diff(targets, helper_targets), solver=solver,
                                                               **kwargs)
        else:
            return super().solve_steady_state(calibration, unknowns, targets, solver=solver, **kwargs)


# Useful type aliases
Model = CombinedBlock

# A CombinedBlock sub-class specifically for steady state calibration with helper blocks
class SSCalibrationBlock(CombinedBlock):
    """An SSCalibrationBlock is a Block object, which includes a set of 'helper' blocks to be used for altering
    the behavior of .steady_state and .solve_steady_state methods. In practice, the common use-case for an
    SSCalibrationBlock is to help .solve_steady_state solve for a subset of the unknowns/targets analytically."""
    def __init__(self, blocks, helper_blocks, calibration, name=""):
        sorted_indices, inputs, outputs = block_sort_w_helpers(blocks, helper_blocks, calibration, return_io=True)
        intermediate_inputs = find_intermediate_inputs_w_helpers(blocks, helper_blocks=helper_blocks)

        super().__init__(blocks, name=name, sorted_indices=sorted_indices, intermediate_inputs=intermediate_inputs)

        self.helper_blocks = helper_blocks
        self.inputs, self.outputs = OrderedSet(inputs), OrderedSet(outputs)

        self.outputs_orig = set().union(*[block.outputs for block in self.blocks if block not in helper_blocks])
        self.inputs_orig = set().union(*[block.inputs for block in self.blocks if block not in helper_blocks]) - self.outputs_orig

    def __repr__(self):
        return f"<SSCalibrationBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], helper_targets={}, evaluate_helpers=True, **block_kwargs):
        """Evaluate a partial equilibrium steady state of the RedirectedBlock given a `calibration`"""
        ss = calibration.copy()
        helper_outputs = {}
        for block in self.blocks:
            if not evaluate_helpers and block in self.helper_blocks:
                continue
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **block_kwargs)
            if evaluate_helpers and block in self.helper_blocks:
                helper_outputs.update({k: v for k, v in outputs.toplevel.items() if k in block.outputs | set(helper_targets.keys())})
                ss.update(outputs)
            else:
                # Don't overwrite entries in ss_values corresponding to what has already
                # been solved for in helper_blocks so we can check for consistency after-the-fact
                ss.update(outputs) if evaluate_helpers else ss.update(outputs.difference(helper_outputs))
        return ss


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

        return topological_sort(dep), inargs, outargs
    else:
        # step 1: map outputs to blocks for topological sort
        outmap = construct_output_map_w_helpers(blocks, helper_blocks=helper_blocks)

        # step 2: dependency graph for topological sort and input list
        dep = construct_dependency_graph_w_helpers(blocks, outmap, helper_blocks=helper_blocks, calibration=calibration)

        return topological_sort(dep)


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
