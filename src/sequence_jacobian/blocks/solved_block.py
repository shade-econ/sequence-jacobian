import warnings

from ..primitives import Block
from ..blocks.simple_block import simple
from ..utilities import graph
from .support.bijection import Bijection


def solved(unknowns, targets, block_list=[], solver=None, solver_kwargs={}, name=""):
    """Creates SolvedBlocks. Can be applied in two ways, both of which return a SolvedBlock:
        - as @solved(unknowns=..., targets=...) decorator on a single SimpleBlock
        - as function solved(blocklist=..., unknowns=..., targets=...) where blocklist
            can be any list of blocks
    """
    if block_list:
        if not name:
            name = f"{block_list[0].name}_to_{block_list[-1].name}_solved"
        # ordinary call, not as decorator
        return SolvedBlock(block_list, name, unknowns, targets, solver=solver, solver_kwargs=solver_kwargs)
    else:
        # call as decorator, return function of function
        def singleton_solved_block(f):
            return SolvedBlock([simple(f)], f.__name__, unknowns, targets, solver=solver, solver_kwargs=solver_kwargs)
        return singleton_solved_block


class SolvedBlock(Block):
    """SolvedBlocks are mini SHADE models embedded as blocks inside larger SHADE models.

    When creating them, we need to provide the basic ingredients of a SHADE model: the list of
    blocks comprising the model, the list on unknowns, and the list of targets.

    When we use .jac to ask for the Jacobian of a SolvedBlock, we are really solving for the 'G'
    matrices of the mini SHADE models, which then become the 'curlyJ' Jacobians of the block.

    Similarly, when we use .td to evaluate a SolvedBlock on a path, we are really solving for the
    nonlinear transition path such that all internal targets of the mini SHADE model are zero.
    """

    def __init__(self, blocks, name, unknowns, targets, solver=None, solver_kwargs={}):
        # Store the actual blocks in ._blocks_unsorted, and use .blocks_w_helpers and .blocks to index from there.
        self._blocks_unsorted = blocks
        self.M = Bijection({})  # don't inherit membrane from parent blocks (think more about this later)

        # Upon instantiation, we only have enough information to conduct a sort ignoring HelperBlocks
        # since we need a `calibration` to resolve cyclic dependencies when including HelperBlocks in a topological sort
        # Hence, we will cache that info upon first invocation of the steady_state
        self._sorted_indices_w_o_helpers = graph.block_sort(blocks)
        self._sorted_indices_w_helpers = None  # These indices are cached the first time steady state is evaluated
        self._required = graph.find_outputs_that_are_intermediate_inputs(blocks)

        # User-facing attributes for accessing blocks
        # .blocks_w_helpers meant to only interface with steady_state functionality
        # .blocks meant to interface with dynamic functionality (impulses and jacobian calculations)
        self.blocks_w_helpers = None
        self.blocks = [self._blocks_unsorted[i] for i in self._sorted_indices_w_o_helpers]

        self.name = name
        self.unknowns = unknowns
        self.targets = targets
        self.solver = solver
        self.solver_kwargs = solver_kwargs

        # need to have inputs and outputs!!!
        self.outputs = (set.union(*(b.outputs for b in blocks)) | set(list(self.unknowns.keys()))) - set(self.targets)
        self.inputs = set.union(*(b.inputs for b in blocks)) - self.outputs

    def __repr__(self):
        return f"<SolvedBlock '{self.name}'>"

    def _steady_state(self, calibration, unknowns=None, helper_blocks=None, solver=None,
                      consistency_check=False, ttol=1e-9, ctol=1e-9, verbose=False):
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for HelperBlocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = graph.block_sort(self._blocks_unsorted, helper_blocks=helper_blocks,
                                                              calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        # Allow override of unknowns/solver, if one wants to evaluate the SolvedBlock at a particular set of
        # unknown values akin to the steady_state method of Block
        if unknowns is None:
            unknowns = self.unknowns
        if solver is None:
            solver = self.solver

        return super().solve_steady_state(calibration, unknowns, self.targets, solver=solver,
                                          consistency_check=consistency_check, ttol=ttol, ctol=ctol, verbose=verbose)

    def _impulse_nonlinear(self, ss, exogenous=None, monotonic=False, Js=None, returnindividual=False, verbose=False):
        return super().solve_impulse_nonlinear(ss, exogenous=exogenous,
                                               unknowns=list(self.unknowns.keys()), Js=Js,
                                               targets=self.targets if isinstance(self.targets, list) else list(self.targets.keys()),
                                               monotonic=monotonic, returnindividual=returnindividual, verbose=verbose)

    def _impulse_linear(self, ss, exogenous, T=None, Js=None):
        return super().solve_impulse_linear(ss, exogenous=exogenous, unknowns=list(self.unknowns.keys()),
                                            targets=self.targets if isinstance(self.targets, list) else list(self.targets.keys()),
                                            T=T, Js=Js)

    def _jacobian(self, ss, exogenous=None, T=300, outputs=None, Js=None):
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None:
            outputs = list(self.outputs)
        relevant_shocks = [i for i in self.inputs if i in exogenous]

        return super().solve_jacobian(ss, relevant_shocks, unknowns=list(self.unknowns.keys()),
                                      targets=self.targets if isinstance(self.targets, list) else list(self.targets.keys()),
                                      T=T, outputs=outputs, Js=Js)
