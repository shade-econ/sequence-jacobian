"""A general function for computing a model's steady state variables and parameters values"""

import numpy as np
import scipy.optimize as opt
from copy import deepcopy
from functools import partial

from .support import compute_target_values, extract_multivariate_initial_values_and_bounds,\
    extract_univariate_initial_values_or_bounds, constrained_multivariate_residual, run_consistency_check,\
    subset_helper_block_unknowns, instantiate_steady_state_mutable_kwargs, find_excludable_helper_blocks
from ..utilities import solvers, graph, misc


# Find the steady state solution
def steady_state(blocks, calibration, unknowns, targets, sort_blocks=True,
                 helper_blocks=None, helper_targets=None,
                 consistency_check=True, ttol=2e-12, ctol=1e-9, fragile=False,
                 block_kwargs=None, verbose=False, solver=None, solver_kwargs=None,
                 constrained_method="linear_continuation", constrained_kwargs=None):
    """
    For a given model (blocks), calibration, unknowns, and targets, solve for the steady state values.

    blocks: `list`
        A list of blocks, which include the types: SimpleBlock, HetBlock, SolvedBlock, CombinedBlock
    calibration: `dict`
        The pre-specified values of variables/parameters provided to the steady state computation
    unknowns: `dict`
        A dictionary mapping unknown variables to either initial values or bounds to be provided to the numerical solver
    targets: `dict`
        A dictionary mapping target variables to desired numerical values, other variables solved for along the DAG
    sort_blocks: `bool`
        Whether the blocks need to be topologically sorted (only False when this function is called from within a
        Block object, like CombinedBlock, that has already pre-sorted the blocks)
    helper_blocks: `list`
        A list of blocks that replace some of the equations in the DAG to aid steady state calculation
    helper_targets: `list/dict`
        A list/dict of target names (and optionally their values) that are handled by the helper blocks
    consistency_check: `bool`
        If helper blocks are a portion of the argument blocks, re-run the DAG with the computed steady state values
        without the assistance of helper blocks and see if the targets are still hit
    ttol: `float`
        The tolerance for the targets---how close the user wants the computed target values to equal the desired values
    ctol: `float`
        The tolerance for the consistency check---how close the user wants the computed target values, without the
        use of helper blocks, to equal the desired values
    fragile: `bool`
        Throw errors instead of warnings when certain criteria are not met, i.e if the consistency_check fails
    block_kwargs: `dict`
        A dict of any kwargs that specify additional settings in order to evaluate block.steady_state for any
        potential Block object, e.g. HetBlocks have backward_tol and forward_tol settings that are specific to that
        Block sub-class.
    verbose: `bool`
        Display the content of optional print statements within the solver for more responsive feedback
    solver: `string`
        The name of the numerical solver that the user would like to user. Can either be a custom solver the user
        implemented, or one of the standard root-finding methods in scipy.optim.root_scalar or scipy.optim.root
    solver_kwargs: `dict`
        The keyword arguments that the user's chosen solver requires to run
    constrained_method: `str`
        When using solvers that typically only take an initial value, x0, we provide a few options for manipulating
        the solver to account for bounds when finding a solution. These methods are described in the
        constrained_multivariate_residual function.
    constrained_kwargs:
        The keyword arguments that the user's chosen constrained method requires to run

    return: ss_values: `dict`
        A dictionary containing all of the pre-specified values and computed values from the steady state computation
    """

    helper_blocks, helper_targets, block_kwargs, solver_kwargs, constrained_kwargs =\
        instantiate_steady_state_mutable_kwargs(helper_blocks, helper_targets, block_kwargs,
                                                solver_kwargs, constrained_kwargs)

    # Initial setup of blocks, targets, and dictionary of steady state values to be returned
    blocks_all = blocks + helper_blocks
    targets = {t: 0. for t in targets} if isinstance(targets, list) else targets
    helper_targets = {t: targets[t] for t in targets if t in helper_targets} if isinstance(helper_targets, list) else helper_targets

    ss_values = deepcopy(calibration)
    ss_values.update(helper_targets)

    helper_unknowns = subset_helper_block_unknowns(unknowns, helper_blocks, helper_targets)
    helper_indices = np.arange(len(blocks), len(blocks_all))

    if sort_blocks:
        dep = graph.construct_dependency_graph(blocks, graph.construct_output_map(blocks, helper_blocks=helper_blocks),
                                               calibration=ss_values, helper_blocks=helper_blocks)
        topsorted = graph.topological_sort(dep)
    else:
        topsorted = range(len(blocks + helper_blocks))

    def residual(targets_dict, unknown_keys, unknown_values, include_helpers=True):
        ss_values.update(misc.smart_zip(unknown_keys, unknown_values))

        helper_outputs = {}

        # TODO: Later on optimize to not evaluating blocks in residual that are no longer needed due to helper
        #   block subsetting
        # Progress through the DAG computing the resulting steady state values based on the unknown_values
        # provided to the residual function
        for i in topsorted:
            if not include_helpers and blocks_all[i] in helper_blocks:
                continue
            # Want to see hetoutputs
            elif hasattr(blocks_all[i], 'hetoutput') and blocks_all[i].hetoutput is not None:
                outputs = eval_block_ss(blocks_all[i], ss_values, hetoutput=True, verbose=verbose, **block_kwargs)
                ss_values.update(misc.dict_diff(outputs, helper_outputs))
            else:
                outputs = eval_block_ss(blocks_all[i], ss_values, consistency_check=consistency_check,
                                        ttol=ttol, ctol=ctol, verbose=verbose, **block_kwargs)
                if include_helpers and blocks_all[i] in helper_blocks:
                    helper_outputs.update(outputs)
                    ss_values.update(outputs)
                else:
                    # Don't overwrite entries in ss_values corresponding to what has already
                    # been solved for in helper_blocks so we can check for consistency after-the-fact
                    ss_values.update(misc.dict_diff(outputs, helper_outputs))

        # Because in solve_for_unknowns, models that are fully "solved" (i.e. RBC) require the
        # dict of ss_values to compute the "unknown_solutions"
        return compute_target_values(targets_dict, ss_values)

    if helper_blocks:
        unknowns_solved = _solve_for_unknowns_w_helper_blocks(residual, unknowns, targets, helper_unknowns,
                                                              helper_targets, helper_indices, blocks_all, dep,
                                                              solver, solver_kwargs,
                                                              constrained_method=constrained_method,
                                                              constrained_kwargs=constrained_kwargs,
                                                              tol=ttol, verbose=verbose, fragile=fragile)
    else:
        unknowns_solved = _solve_for_unknowns(residual, unknowns, targets, solver, solver_kwargs,
                                              constrained_method=constrained_method,
                                              constrained_kwargs=constrained_kwargs,
                                              tol=ttol, verbose=verbose, fragile=fragile)

    # Check that the solution is consistent with what would come out of the DAG without the helper blocks
    if consistency_check:
        # Add the unknowns not handled by helpers into the DAG to be checked.
        unknowns_solved.update({k: ss_values[k] for k in unknowns if k not in unknowns_solved})

        cresid = abs(np.max(residual(targets, unknowns_solved.keys(), unknowns_solved.values(),
                                     include_helpers=False)))
        run_consistency_check(cresid, ctol=ctol, fragile=fragile)

    # Update to set the solutions for the steady state values of the unknowns
    ss_values.update(unknowns_solved)

    return ss_values


def eval_block_ss(block, calibration, **kwargs):
    """Evaluate the .ss method of a block, given a dictionary of potential arguments"""
    return block.steady_state({k: v for k, v in calibration.items() if k in block.inputs},
                              **{k: v for k, v in kwargs.items() if k in misc.input_kwarg_list(block.steady_state)})


def _solve_for_unknowns(residual, unknowns, targets, solver, solver_kwargs, residual_kwargs=None,
                        constrained_method="linear_continuation", constrained_kwargs=None,
                        tol=2e-12, verbose=False, fragile=False):
    """
    Given a residual function (constructed within steady_state) and a set of bounds or initial values for
    the set of unknowns, solve for the root.
    TODO: Implemented as a hidden method as of now because this function relies on the structure of steady_state
        specifically and will not work with a generic residual function, due to the way it currently expects residual
        to call variables not provided as arguments explicitly but that exist in its enclosing scope.

    residual: `function`
        A function to be supplied to a numerical solver that takes unknown values as arguments
        and returns computed targets.
    unknowns: `dict`
        Refer to the `steady_state` function docstring for the "unknowns" variable
    targets: `dict`
        Refer to the `steady_state` function docstring for the "targets" variable
    tol: `float`
        The absolute convergence tolerance of the computed target to the desired target value in the numerical solver
    solver: `str`
        Refer to the `steady_state` function docstring for the "solver" variable
    solver_kwargs:
        Refer to the `steady_state` function docstring for the "solver_kwargs" variable

    return: The root[s] of the residual function as either a scalar (float) or a list of floats
    """
    if residual_kwargs is None:
        residual_kwargs = {}

    scipy_optimize_uni_solvers = ["bisect", "brentq", "brenth", "ridder", "toms748", "newton", "secant", "halley"]
    scipy_optimize_multi_solvers = ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing", "diagbroyden",
                                    "excitingmixing", "krylov", "df-sane"]

    # Construct a reduced residual function, which contains addl context of unknowns, targets, and keyword arguments.
    # This is to bypass issues with passing a residual function that requires contextual, positional arguments
    # separate from the unknown values that need to be solved for into the multivariate solvers
    residual_f = partial(residual, targets, unknowns.keys(), **residual_kwargs)

    if solver is None:
        raise RuntimeError("Must provide a numerical solver from the following set: brentq, broyden, solved")
    elif solver in scipy_optimize_uni_solvers:
        initial_values_or_bounds = extract_univariate_initial_values_or_bounds(unknowns)
        result = opt.root_scalar(residual_f, method=solver, xtol=tol,
                                 **initial_values_or_bounds, **solver_kwargs)
        if not result.converged:
            raise ValueError(f"Steady-state solver, {solver}, did not converge.")
        unknown_solutions = result.root
    elif solver in scipy_optimize_multi_solvers:
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns, fragile=fragile)
        # If no bounds were provided
        if not bounds:
            result = opt.root(residual_f, initial_values,
                              method=solver, tol=tol, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual_f, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            result = opt.root(constrained_residual, initial_values,
                              method=solver, tol=tol, **solver_kwargs)
        if not result.success:
            raise ValueError(f"Steady-state solver, {solver}, did not converge."
                             f" The termination status is {result.status}.")
        unknown_solutions = list(result.x)
    # TODO: Implement a more general interface for custom solvers, so we don't need to add new elifs at this level
    #  everytime a new custom solver is implemented.
    elif solver == "broyden_custom":
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        # If no bounds were provided
        if not bounds:
            unknown_solutions, _ = solvers.broyden_solver(residual_f, initial_values,
                                                          tol=tol, verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual_f, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = solvers.broyden_solver(constrained_residual, initial_values,
                                                          verbose=verbose, tol=tol, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "newton_custom":
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        # If no bounds were provided
        if not bounds:
            unknown_solutions, _ = solvers.newton_solver(residual_f, initial_values,
                                                         tol=tol, verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual_f, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = solvers.newton_solver(constrained_residual, initial_values,
                                                         tol=tol, verbose=verbose, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "solved":
        # If the model does not require a numerical solution then return an empty tuple for the unknowns
        # that require a numerical solution
        unknown_solutions = ()
    else:
        raise RuntimeError(f"steady_state is not yet compatible with {solver}.")

    return dict(misc.smart_zip(unknowns.keys(), unknown_solutions))


def _solve_for_unknowns_w_helper_blocks(residual, unknowns, targets, helper_unknowns, helper_targets,
                                        helper_indices, blocks_all, block_dependencies,
                                        solver, solver_kwargs, constrained_method="linear_continuation",
                                        constrained_kwargs=None, tol=2e-12, verbose=False, fragile=False):
    """Enhance the solver executed in _solve_for_unknowns by handling a subset of unknowns and targets
    with helper blocks, reducing the number of unknowns that need to be numerically solved for."""
    # Initial evaluation of the DAG at the initial values of the unknowns, including the helper blocks,
    # to populate the `ss_values` dict with the unknown values that:
    # a) are handled by helper blocks and b) are excludable from the main DAG.
    unknowns_init_vals = [v if not isinstance(v, tuple) else (v[0] + v[1]) / 2 for v in unknowns.values()]
    targets_init_vals = dict(misc.smart_zip(targets.keys(), residual(targets, unknowns.keys(), unknowns_init_vals)))

    # Find the unknowns and targets that are both handled by helper blocks and are excludable from the main DAG
    # evaluation by checking block dependencies
    unknowns_excl, targets_excl = find_excludable_helper_blocks(blocks_all, block_dependencies,
                                                                helper_indices, helper_unknowns, helper_targets)

    # Subset out the unknowns and targets that are not excludable from the main DAG loop
    unknowns_non_excl = {k: unknowns[k] for k in misc.list_diff(list(unknowns.keys()), unknowns_excl)}
    targets_non_excl = misc.dict_diff(targets, targets_excl)

    # If the `targets` that are handled by helpers and excludable from the main DAG evaluate to 0. at the set of
    # `unknowns` initial values and the initial `calibration`, then those `targets` have been hit analytically and
    # we can omit them and their corresponding `unknowns` in the main DAG.
    if np.all(np.isclose([targets_init_vals[t] for t in targets_excl.keys()], 0.)):
        unknown_solutions = _solve_for_unknowns(residual, unknowns_non_excl, targets_non_excl,
                                                solver, solver_kwargs,
                                                residual_kwargs={"include_helpers": False},
                                                constrained_method=constrained_method,
                                                constrained_kwargs=constrained_kwargs,
                                                tol=tol, verbose=verbose, fragile=fragile)
    # If targets handled by helpers and excludable from the main DAG are not satisfied then
    # it is assumed that helper blocks merely aid in providing more accurate guesses for the DAG solution,
    # and they remain a part of the main DAG when solving.
    else:
        unknown_solutions = _solve_for_unknowns(residual, unknowns, targets, solver, solver_kwargs,
                                                constrained_method=constrained_method,
                                                constrained_kwargs=constrained_kwargs,
                                                tol=tol, verbose=verbose, fragile=fragile)
    return unknown_solutions
