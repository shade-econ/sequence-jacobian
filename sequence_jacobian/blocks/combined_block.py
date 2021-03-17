"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from .support.impulse import ImpulseDict
from ..primitives import Block
from .. import utilities as utils
from ..steady_state.drivers import eval_block_ss
from ..steady_state.support import provide_solver_default
from ..jacobian.classes import JacobianDict


def combine(*args, name="", helper_indices=None, model_alias=False):
    # TODO: Implement a check that all args are child types of AbstractBlock, when that is properly implemented
    return CombinedBlock(*args, name=name, helper_indices=helper_indices, model_alias=model_alias)


class CombinedBlock(Block):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, *blocks, name="", helper_indices=None, model_alias=False):

        # Store the actual blocks in ._blocks_unsorted, and use .blocks_w_helpers and .blocks to index from there.
        self._blocks_unsorted = blocks

        # Upon instantiation, we only have enough information to conduct a sort ignoring helper blocks
        # since we need a `calibration` to resolve cyclic dependencies when including helper blocks in a topological sort
        # Hence, we will cache that info upon first invocation of the steady_state
        self.helper_indices = helper_indices if helper_indices is not None else []
        self._sorted_indices_w_o_helpers = utils.graph.block_sort([b for i, b in enumerate(blocks) if i not in self.helper_indices])
        self._sorted_indices_w_helpers = None  # These indices are cached the first time steady state is evaluated
        self._required = utils.graph.find_outputs_that_are_intermediate_inputs([b for i, b in enumerate(blocks) if i not in self.helper_indices])

        # User-facing attributes for accessing blocks
        # .blocks_w_helpers meant to only interface with steady_state functionality
        # .blocks meant to interface with dynamic functionality (impulses and jacobian calculations)
        self.blocks_w_helpers = None
        self.blocks = [self._blocks_unsorted[i] for i in self._sorted_indices_w_o_helpers]

        if not name:
            self.name = f"{self.blocks[0].name}_to_{self.blocks[-1].name}_combined"
        else:
            self.name = name

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

    def steady_state(self, calibration, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for helper blocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = utils.graph.block_sort(self._blocks_unsorted, ignore_helpers=False,
                                                                    helper_indices=self.helper_indices,
                                                                    calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        ss_partial_eq = deepcopy(calibration)
        for block in self.blocks_w_helpers:
            ss_partial_eq.update(eval_block_ss(block, ss_partial_eq, **kwargs))
        return ss_partial_eq

    def impulse_nonlinear(self, ss, exogenous, **kwargs):
        """Calculate a partial equilibrium, non-linear impulse response to a set of `exogenous` shocks from
        a steady state, `ss`"""
        irf_nonlin_partial_eq = deepcopy(exogenous)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_nonlin_partial_eq.update({k: v - ss[k] for k, v in block.impulse_nonlinear(ss, input_args, **kwargs)})

        return ImpulseDict(irf_nonlin_partial_eq, ss).levels()

    def impulse_linear(self, ss, exogenous, T=None):
        """Calculate a partial equilibrium, linear impulse response to a set of `exogenous` shocks from
        a steady_state, `ss`"""
        irf_lin_partial_eq = deepcopy(exogenous)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update({k: v for k, v in block.impulse_linear(ss, input_args, T=T)})

        return ImpulseDict(irf_lin_partial_eq, ss)

    def jacobian(self, ss, exogenous=None, T=None, outputs=None, save=False, use_saved=False):
        """Calculate a partial equilibrium Jacobian with respect to a set of `exogenous` shocks at
        a steady state, `ss`"""
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None:
            outputs = self.outputs
        kwargs = {"exogenous": exogenous, "T": T, "outputs": outputs, "save": save, "use_saved": use_saved}

        for i, block in enumerate(self.blocks):
            curlyJ = block.jacobian(ss, **{k: kwargs[k] for k in utils.misc.input_kwarg_list(block.jacobian) if k in kwargs}).complete()

            # If we want specific list of outputs, restrict curlyJ to that before continuing
            curlyJ = curlyJ[[k for k in curlyJ.outputs if k in outputs or k in self._required]]
            if i == 0:
                J_partial_eq = curlyJ.compose(JacobianDict.identity(exogenous))
            else:
                J_partial_eq.update(curlyJ.compose(J_partial_eq))

        return J_partial_eq

    def solve_steady_state(self, calibration, unknowns, targets, solver=None, **kwargs):
        """Evaluate a general equilibrium steady state of the CombinedBlock given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for helper blocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = utils.graph.block_sort(self._blocks_unsorted, ignore_helpers=False,
                                                                    helper_indices=self.helper_indices,
                                                                    calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        if solver is None:
            solver = provide_solver_default(unknowns)

        return super().solve_steady_state(calibration, unknowns, targets, solver=solver, sort_blocks=False, **kwargs)
