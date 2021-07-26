"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from .support.impulse import ImpulseDict
from ..primitives import Block
from .. import utilities as utils
from ..blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from ..steady_state.drivers import eval_block_ss
from ..steady_state.support import provide_solver_default
from ..jacobian.classes import JacobianDict
from ..steady_state.classes import SteadyStateDict
from .support.bijection import Bijection


def combine(blocks, name="", model_alias=False):
    return CombinedBlock(blocks, name=name, model_alias=model_alias)


# Useful functional alias
def create_model(blocks, **kwargs):
    return combine(blocks, model_alias=True, **kwargs)


class CombinedBlock(Block):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, blocks, name="", model_alias=False):

        self._blocks_unsorted = [b if isinstance(b, Block) else JacobianDictBlock(b) for b in blocks]
        self._sorted_indices = utils.graph.block_sort(blocks)
        self._required = utils.graph.find_outputs_that_are_intermediate_inputs(blocks)
        self.blocks = [self._blocks_unsorted[i] for i in self._sorted_indices]
        self.M = Bijection({})

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

    def _steady_state(self, calibration, helper_blocks=None, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""
        if helper_blocks is None:
            helper_blocks = []

        topsorted = utils.graph.block_sort(self.blocks, calibration=calibration, helper_blocks=helper_blocks)
        blocks_all = self.blocks + helper_blocks

        ss_partial_eq_toplevel = deepcopy(calibration)
        ss_partial_eq_internal = {}
        for i in topsorted:
            outputs = eval_block_ss(blocks_all[i], ss_partial_eq_toplevel, **kwargs)
            ss_partial_eq_toplevel.update(outputs.toplevel)
            if outputs.internal:
                ss_partial_eq_internal.update(outputs.internal)
        ss_partial_eq_internal = {self.name: ss_partial_eq_internal} if ss_partial_eq_internal else {}
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

    def _jacobian(self, ss, exogenous=None, T=None, outputs=None, Js=None):
        """Calculate a partial equilibrium Jacobian with respect to a set of `exogenous` shocks at
        a steady state, `ss`"""
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None:
            outputs = self.outputs
        kwargs = {"exogenous": exogenous, "T": T, "outputs": outputs, "Js": Js}

        for i, block in enumerate(self.blocks):
            curlyJ = block._jacobian(ss, **{k: kwargs[k] for k in utils.misc.input_kwarg_list(block._jacobian) if k in kwargs}).complete()

            # If we want specific list of outputs, restrict curlyJ to that before continuing
            curlyJ = curlyJ[[k for k in curlyJ.outputs if k in outputs or k in self._required]]
            if i == 0:
                J_partial_eq = curlyJ.compose(JacobianDict.identity(exogenous))
            else:
                J_partial_eq.update(curlyJ.compose(J_partial_eq))

        return J_partial_eq

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
