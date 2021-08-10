"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy
import numpy as np

from .support.impulse import ImpulseDict
from ..primitives import Block
from .. import utilities as utils
from ..blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from ..blocks.parent import Parent
from ..steady_state.support import provide_solver_default
from ..jacobian.classes import JacobianDict
from ..steady_state.classes import SteadyStateDict


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
    def __init__(self, blocks, name="", model_alias=False):
        super().__init__()

        self._blocks_unsorted = [b if isinstance(b, Block) else JacobianDictBlock(b) for b in blocks]
        self._sorted_indices = utils.graph.block_sort(blocks)
        self._required = utils.graph.find_outputs_that_are_intermediate_inputs(blocks)
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

    def _steady_state(self, calibration, helper_blocks=None, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""
        if helper_blocks is None:
            helper_blocks = []

        topsorted = utils.graph.block_sort(self.blocks, calibration=calibration, helper_blocks=helper_blocks)
        blocks_all = self.blocks + helper_blocks

        ss_partial_eq_toplevel = deepcopy(calibration)
        ss_partial_eq_internal = {}
        for i in topsorted:
            outputs = blocks_all[i].steady_state(ss_partial_eq_toplevel, **kwargs)
            ss_partial_eq_toplevel.update(outputs.toplevel)
            if outputs.internal:
                ss_partial_eq_internal.update(outputs.internal)
        return SteadyStateDict(ss_partial_eq_toplevel, internal=ss_partial_eq_internal)

    def _impulse_nonlinear(self, ss, exogenous, **kwargs):
        """Calculate a partial equilibrium, non-linear impulse response to a set of `exogenous` shocks from
        a steady state, `ss`"""
        irf_nonlin_partial_eq = deepcopy(exogenous)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_nonlin_partial_eq.update({k: v for k, v in block.impulse_nonlinear(ss, input_args, **kwargs)})

        return ImpulseDict(irf_nonlin_partial_eq)

    def _impulse_linear(self, ss, exogenous, T=None, Js=None):
        """Calculate a partial equilibrium, linear impulse response to a set of `exogenous` shocks from
        a steady_state, `ss`"""
        irf_lin_partial_eq = deepcopy(exogenous)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update({k: v for k, v in block.impulse_linear(ss, input_args, T=T, Js=Js)})

        return ImpulseDict(irf_lin_partial_eq)

    def partial_jacobians(self, ss, inputs=None, T=None, Js=None):
        """Calculate partial Jacobians (i.e. without forward accumulation) wrt `inputs` and outputs of other blocks."""
        if inputs is None:
            inputs = self.inputs

        # Add intermediate inputs; remove vector-valued inputs
        shocks = set(inputs) | self._required
        shocks -= set([k for k, v in ss.items() if np.size(v) > 1])

        # Compute Jacobians along the DAG
        curlyJs = {}
        kwargs = {"exogenous": shocks, "T": T, "Js": Js}
        for block in self.blocks:
            # Don't remap here
            curlyJ = block.jacobian(ss, **{k: kwargs[k] for k in utils.misc.input_kwarg_list(block._jacobian)
                                           if k in kwargs})

            # Don't return empty Jacobians
            if curlyJ.outputs:
                curlyJs[block.name] = curlyJ

        return curlyJs

    def _jacobian(self, ss, exogenous=None, T=None, outputs=None, Js=None):
        """Calculate a partial equilibrium Jacobian with respect to a set of `exogenous` shocks at
        a steady state, `ss`"""
        if outputs is not None:
            # if list of outputs provided, we need to obtain these and 'required' along the way
            alloutputs = set(outputs) | self._required
        else:
            # otherwise, set to None, implies default behavior of obtaining all outputs in curlyJs
            alloutputs = None

        # Compute all partial Jacobians
        curlyJs = self.partial_jacobians(ss, inputs=exogenous, T=T, Js=Js)

        # Forward accumulate partial Jacobians
        out = JacobianDict.identity(exogenous)
        for curlyJ in curlyJs.values():
            if alloutputs is not None:
                # don't accumulate derivatives we don't need or care about
                curlyJ = curlyJ[[k for k in alloutputs if k in curlyJ.outputs]]
            out.update(curlyJ.compose(out))

        if outputs is not None:
            # don't return derivatives that we don't care about (even if required them above)
            return out[[k for k in outputs if k in out.outputs]]
        else:
            return out


    def solve_steady_state(self, calibration, unknowns, targets, solver=None, helper_blocks=None,
                           sort_blocks=False, **kwargs):
        """Evaluate a general equilibrium steady state of the CombinedBlock given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        if solver is None:
            solver = provide_solver_default(unknowns)
        if helper_blocks and sort_blocks is False:
            sort_blocks = True

        return super().solve_steady_state(calibration, unknowns, targets, solver=solver,
                                          helper_blocks=helper_blocks, sort_blocks=sort_blocks, **kwargs)


# Useful type aliases
Model = CombinedBlock
