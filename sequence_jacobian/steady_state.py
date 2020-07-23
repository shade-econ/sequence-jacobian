"""A general function for computing a model's steady state variable and parameters values"""

import numpy as np
import scipy.optimize as opt
from copy import deepcopy

import sequence_jacobian as sj
from sequence_jacobian import HelperBlock
from sequence_jacobian.utils import make_tuple, broyden_solver


# Find the steady state solution
def steady_state(blocks, calibration, unknowns, targets,
                 consistency_check=True, ttol=1e-9, ctol=1e-9,
                 solver=None, **solver_kwargs):

    ss_values = deepcopy(calibration)
    topsorted = sj.utils.block_sort(blocks, calibration=calibration)

    def residual(unknown_values, include_helpers=True, update_unknowns_inplace=False):
        ss_values.update(smart_zip(unknowns.keys(), unknown_values))

        helper_outputs = {}

        # Progress through the DAG computing the resulting steady state values based on the unknown_values
        # provided to the residual function
        for i in topsorted:
            if not include_helpers and isinstance(blocks[i], HelperBlock):
                continue
            else:
                outputs = eval_block_ss(blocks[i], ss_values)
                if include_helpers and isinstance(blocks[i], HelperBlock):
                    helper_outputs.update(outputs)
                    ss_values.update(outputs)
                else:
                    # Don't overwrite entries in ss_values corresponding to what has already
                    # been solved for in helper_blocks so we can check for consistency after-the-fact
                    ss_values.update(dict_diff(outputs, helper_outputs))

        # Update the "unknowns" dictionary *in place* with its steady state values.
        # i.e. the "unknowns" in the namespace in which this function is invoked will change!
        # Useful for a) if the unknown values are updated while iterating each blocks' ss computation within the DAG,
        # and/or b) if the user wants to update "unknowns" in place for use in other computations.
        if update_unknowns_inplace:
            unknowns.update(smart_zip(unknowns.keys(), [ss_values[key] for key in unknowns.keys()]))

        # Because in solve_for_unknowns, models that are fully "solved" (i.e. RBC) require the
        # dict of ss_values to compute the "unknown_solutions"
        return compute_target_values(targets, ss_values)

    unknown_solutions = solve_for_unknowns(residual, unknowns, tol=ttol, solver=solver, **solver_kwargs)

    # Check that the solution is consistent with what would come out of the DAG without
    # the helper blocks
    if consistency_check:
        assert abs(np.max(residual(unknown_solutions, include_helpers=False))) < ctol

    # Update to set the solutions for the steady state values of the unknowns
    ss_values.update(zip(unknowns, make_tuple(unknown_solutions)))

    # TODO: Check Walras' Law

    return ss_values


def find_target_block(blocks, target):
    for block in blocks:
        if target in blocks.output:
            return block


# Allow targets to be specified in the following formats
# 1) target = {"asset_mkt": 0} (the standard case, where the target = 0)
# 2) target = {"r": 0.01} (allowing for the target to be non-zero)
# 3) target = {"K": "A"} (allowing the target to be another variable in potential_args)
def compute_target_values(targets, potential_args):
    target_values = np.empty(len(targets))
    for (i, t) in enumerate(targets):
        v = targets[t]
        if type(v) == str:
            target_values[i] = potential_args[t] - potential_args[v]
        else:
            target_values[i] = potential_args[t] - v

    # Univariate solvers require float return values (and not lists)
    if len(targets) == 1:
        return target_values[0]
    else:
        return target_values


# Analogous to the SHADE workflow of having blocks call utils.apply(self._fss, inputs) but not as general.
def eval_block_ss(block, potential_args):
    input_args = {unprime(arg_name): potential_args[unprime(arg_name)] for arg_name in block.inputs}

    # Simple and HetBlocks require different handling of block.ss() output since
    # SimpleBlocks return a tuple of un-labeled arguments, whereas HetBlocks return dictionaries
    if isinstance(block, sj.SimpleBlock) or isinstance(block, sj.HelperBlock):
        output_args = make_tuple(block.ss(**input_args))
        outputs = {o: output_args[i] for i, o in enumerate(block.output_list)}
    else:  # assume it's a HetBlock. Figure out a nicer way to handle SolvedBlocks/CombinedBlocks later on
        outputs = block.ss(**input_args)

    return outputs


def solve_for_unknowns(residual, unknowns, tol=1e-9, solver=None, **solver_kwargs):
    if solver is None:
        raise RuntimeError("Must provide a numerical solver from the following set: brentq, broyden, solved")
    elif solver == "brentq":
        lb, ub = list(unknowns.values())[0]  # Since brentq is a univariate solver can assume unknowns is a size 1 dict
        unknown_solutions, sol = opt.brentq(residual, lb, ub, xtol=tol, **solver_kwargs)
        if not sol.converged:
            raise ValueError("Steady-state solver did not converge.")
    elif solver == "broyden":
        init_values = np.array(list(unknowns.values()))
        unknown_solutions, _ = broyden_solver(residual, init_values, tol=tol, **solver_kwargs)
    elif solver is "solved":
        # If the entire solution is provided by the helper blocks
        # Call residual() once to update ss_values and to check the targets match the provided solution.
        # The initial value passed into residual (np.zeros()) in this case is irrelevant, but something
        # must still be passed in since the residual function requires an argument.
        assert abs(np.max(residual(smart_zeros(len(unknowns)), update_unknowns_inplace=True))) < tol
        unknown_solutions = list(unknowns.values())
    else:
        raise RuntimeError(f"steady_state is not yet compatible with {solver}.")

    return unknown_solutions


# Expects the notation for a variable primed to be of the standard format "var_p"
# which will then return "var." If the variable is not primed, "ovar" then simply return it
def unprime(s):
    if s[-2:] == "_p":
        return s[:-2]
    else:
        return s


# Returns the dictionary that is the "set difference" between d1 and d2 (based on keys, not key-value pairs)
# E.g. d1 = {"a": 1, "b": 2}, d2 = {"b": 5}, then dict_diff(d1, d2) = {"a": 1}
def dict_diff(d1, d2):
    o_dict = {}
    for k in set(d1.keys()).difference(set(d2.keys())):
        o_dict[k] = d1[k]

    return o_dict


# For handling the case where keys and values may be scalars
def smart_zip(keys, values):
    if isinstance(values, float):
        return zip(keys, [values])
    else:
        return zip(keys, values)


# Return either the float 0. or a np.ndarray of length 0 depending on whether n > 1
def smart_zeros(n):
    if n > 1:
        return np.zeros(n)
    else:
        return 0.