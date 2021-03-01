"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy
import numpy as np

from ..primitives import Block
from .. import utilities as utils
from ..steady_state.drivers import eval_block_ss
from ..steady_state.support import provide_solver_default
from ..nonlinear import td_solve
from ..jacobian.drivers import get_G
from ..jacobian.classes import JacobianDict
from ..blocks.het_block import HetBlock


def combine(*args, name="", model_alias=False):
    # TODO: Implement a check that all args are child types of AbstractBlock, when that is properly implemented
    return CombinedBlock(*args, name=name, model_alias=model_alias)


class CombinedBlock(Block):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, *blocks, name="", model_alias=False):
        # Store the actual blocks in ._blocks_unsorted, and use .blocks_w_helpers and .blocks to index from there.
        self._blocks_unsorted = blocks

        # Upon instantiation, we only have enough information to conduct a sort ignoring HelperBlocks
        # since we need a `calibration` to resolve cyclic dependencies when including HelperBlocks in a topological sort
        # Hence, we will cache that info upon first invocation of the steady_state
        self._sorted_indices_w_o_helpers = utils.graph.block_sort(blocks, ignore_helpers=True)
        self._sorted_indices_w_helpers = None  # These indices are cached the first time steady state is evaluated
        self._required = utils.graph.find_outputs_that_are_intermediate_inputs(blocks, ignore_helpers=True)

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

    def steady_state(self, calibration):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for HelperBlocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = utils.graph.block_sort(self._blocks_unsorted, ignore_helpers=False,
                                                                    calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        ss_partial_eq = deepcopy(calibration)
        for block in self.blocks_w_helpers:
            ss_partial_eq.update(eval_block_ss(block, ss_partial_eq))
        return ss_partial_eq

    def impulse_nonlinear(self, ss, exogenous, in_deviations=True, **kwargs):
        """Calculate a partial equilibrium, non-linear impulse response to a set of `exogenous` shocks from
        a steady state, `ss`"""
        irf_nonlin_partial_eq = {k: ss[k] + v for k, v in exogenous.items()}
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_nonlin_partial_eq.update({k: v for k, v in block.impulse_nonlinear(ss, input_args, **kwargs).items()})

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] - 1 if not np.isclose(ss[k], 0) else v for k, v in irf_nonlin_partial_eq.items()}
        else:
            return irf_nonlin_partial_eq

    def impulse_linear(self, ss, exogenous, T=None, in_deviations=True):
        """Calculate a partial equilibrium, linear impulse response to a set of `exogenous` shocks from
        a steady_state, `ss`"""
        irf_lin_partial_eq = deepcopy(exogenous)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update({k: v for k, v in block.impulse_linear(ss, input_args, T=T).items()})

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] if not np.isclose(ss[k], 0) else v for k, v in irf_lin_partial_eq.items()}
        else:
            return irf_lin_partial_eq

    def jacobian(self, ss, exogenous=None, T=None, outputs=None, save=False, use_saved=False):
        """Calculate a partial equilibrium Jacobian with respect to a set of `exogenous` shocks at
        a steady state, `ss`"""
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None:
            outputs = self.outputs

        J_partial_eq = JacobianDict.identity(exogenous)
        for block in self.blocks:
            if isinstance(block, HetBlock):
                curlyJ = block.jacobian(ss, exogenous, T, save=save, use_saved=use_saved).complete()
            else:
                curlyJ = block.jacobian(ss, exogenous, T).complete()

            # If we want specific list of outputs, restrict curlyJ to that before continuing
            curlyJ = curlyJ[[k for k in curlyJ.outputs if k in outputs or k in self._required]]
            J_partial_eq.update(curlyJ.compose(J_partial_eq))

        return J_partial_eq

    def solve_steady_state(self, calibration, unknowns, targets, solver=None, **kwargs):
        """Evaluate a general equilibrium steady state of the CombinedBlock given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for HelperBlocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = utils.graph.block_sort(self._blocks_unsorted, ignore_helpers=False,
                                                                    calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        if solver is None:
            solver = provide_solver_default(unknowns)

        return super().solve_steady_state(calibration, unknowns, targets, solver=solver, **kwargs)

    def solve_impulse_nonlinear(self, ss, exogenous, unknowns, targets, in_deviations=True, **kwargs):
        """Calculate a general equilibrium, non-linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        irf_nonlin_gen_eq = td_solve(self.blocks, ss, unknowns, targets,
                                     exogenous={k: ss[k] + v for k, v in exogenous.items()}, **kwargs)

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] - 1 if not np.isclose(ss[k], 0) else v for k, v in irf_nonlin_gen_eq.items()}
        else:
            return irf_nonlin_gen_eq

    def solve_impulse_linear(self, ss, exogenous, unknowns, targets, T=None, in_deviations=True, **kwargs):
        """Calculate a general equilibrium, linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        if T is None:
            # infer T from exogenous, check that all shocks have same length
            shock_lengths = [x.shape[0] for x in exogenous.values()]
            if shock_lengths[1:] != shock_lengths[:-1]:
                raise ValueError('Not all shocks in kwargs (exogenous) are same length!')
            T = shock_lengths[0]

        J_gen_eq = get_G(self.blocks, list(exogenous.keys()), unknowns, targets, T=T, ss=ss, **kwargs)
        irf_lin_gen_eq = J_gen_eq.apply(exogenous)

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] if not np.isclose(ss[k], 0) else v for k, v in irf_lin_gen_eq.items()}
        else:
            return irf_lin_gen_eq

    def solve_jacobian(self, ss, exogenous, unknowns, targets, T=None, **kwargs):
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        J_gen_eq = get_G(self.blocks, exogenous, unknowns, targets, T=T, ss=ss, **kwargs)
        return J_gen_eq

