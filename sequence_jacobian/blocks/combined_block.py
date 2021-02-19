"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy
import numpy as np

from .. import utilities as utils
from ..steady_state import eval_block_ss, steady_state, provide_solver_default
from ..nonlinear import td_solve
from ..jacobian.drivers import get_G, forward_accumulate, curlyJ_sorted
from ..jacobian.classes import JacobianDict


def combine(*args, name="", model_alias=False):
    # TODO: Implement a check that all args are child types of AbstractBlock, when that is properly implemented
    return CombinedBlock(*args, name=name, model_alias=model_alias)


class CombinedBlock:
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, *blocks, name="", model_alias=False):
        self._blocks_unsorted = blocks

        self._sorted_indices_w_helpers = None  # These indices are cached the first time steady state is evaluated
        self.blocks_w_helpers = None

        self._sorted_indices_w_o_helpers = utils.graph.block_sort(blocks, ignore_helpers=True)
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

        self._model_alias = model_alias

    def __repr__(self):
        if self._model_alias:
            return f"<Model '{self.name}'>"
        else:
            return f"<CombinedBlock '{self.name}'>"

    def steady_state(self, calibration):
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

    def impulse_nonlinear(self, ss, shocked_paths, in_deviations=True, **kwargs):
        irf_nonlin_partial_eq = {k: ss[k] + v for k, v in shocked_paths.items()}
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}
            # To only return dynamic paths of variables that do not remain at their steady state value
            irf_nonlin_partial_eq.update({k: v for k, v in block.impulse_nonlinear(ss, input_args, **kwargs).items()
                                          if not np.all(v == ss[k])})

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] - 1 if not np.isclose(ss[k], 0) else v for k, v in irf_nonlin_partial_eq.items()}
        else:
            return irf_nonlin_partial_eq

    def impulse_linear(self, ss, exogenous_paths, T=None, in_deviations=True):
        exogenous = list(exogenous_paths.keys())
        if T is None:
            T = len(list(exogenous_paths.values())[0])
        J_partial_eq = self.jacobian(ss, exogenous=exogenous, T=T)
        irf_lin_partial_eq = J_partial_eq.apply(exogenous_paths)

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] if not np.isclose(ss[k], 0) else v for k, v in irf_lin_partial_eq.items()}
        else:
            return irf_lin_partial_eq

    def jacobian(self, ss, exogenous=None, T=None, outputs=None, save=False, use_saved=False):
        if exogenous is None:
            exogenous = list(self.inputs)

        curlyJs, required = curlyJ_sorted(self.blocks, exogenous, ss, T=T, save=save, use_saved=use_saved)
        J_partial_eq = forward_accumulate(curlyJs, exogenous, outputs=outputs, required=required)
        return J_partial_eq

    def solve_steady_state(self, calibration, unknowns, targets, solver=None, **kwargs):
        # If this is the first time invoking steady_state/solve_steady_state, cache the sorted indices
        # accounting for HelperBlocks
        if self._sorted_indices_w_helpers is None:
            self._sorted_indices_w_helpers = utils.graph.block_sort(self._blocks_unsorted, ignore_helpers=False,
                                                                    calibration=calibration)
            self.blocks_w_helpers = [self._blocks_unsorted[i] for i in self._sorted_indices_w_helpers]

        if solver is None:
            solver = provide_solver_default(unknowns)
        ss_gen_eq = steady_state(self.blocks_w_helpers, calibration, unknowns, targets, solver=solver, **kwargs)
        return ss_gen_eq

    def solve_impulse_nonlinear(self, ss, exogenous, unknowns, targets, in_deviations=True, **kwargs):
        irf_nonlin_gen_eq = td_solve(ss, self.blocks, unknowns, targets,
                                     shocked_paths={k: ss[k] + v for k, v in exogenous.items()}, **kwargs)

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] - 1 if not np.isclose(ss[k], 0) else v for k, v in irf_nonlin_gen_eq.items()}
        else:
            return irf_nonlin_gen_eq

    def solve_impulse_linear(self, ss, exogenous, unknowns, targets, T=None, in_deviations=True, **kwargs):
        if T is None:
            T = len(list(exogenous.values())[0])
        J_gen_eq = get_G(self.blocks, list(exogenous.keys()), unknowns, targets, T=T, ss=ss, **kwargs)
        irf_lin_gen_eq = J_gen_eq.apply(exogenous)

        # Default to percentage deviations from steady state. If the steady state value is zero, then just return
        # the level deviations from zero.
        if in_deviations:
            return {k: v/ss[k] if not np.isclose(ss[k], 0) else v for k, v in irf_lin_gen_eq.items()}
        else:
            return irf_lin_gen_eq

    def solve_jacobian(self, ss, exogenous, unknowns, targets, T=None, **kwargs):
        J_gen_eq = get_G(self.blocks, exogenous, unknowns, targets, T=T, ss=ss, **kwargs)
        return J_gen_eq

