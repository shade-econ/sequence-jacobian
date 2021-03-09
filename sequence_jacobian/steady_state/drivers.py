"""A general function for computing a model's steady state variables and parameters values"""

import numpy as np
import scipy.optimize as opt
from copy import deepcopy

from .support import compute_target_values, extract_multivariate_initial_values_and_bounds,\
    extract_univariate_initial_values_or_bounds, constrained_multivariate_residual, run_consistency_check
from ..utilities import solvers, graph, misc
from ..blocks.helper_block import HelperBlock


# Find the steady state solution
def steady_state(blocks, calibration, unknowns, targets,
                 consistency_check=True, ttol=2e-12, ctol=1e-9,
                 block_kwargs=None, verbose=False, fragile=False, solver=None, solver_kwargs=None,
                 constrained_method="linear_continuation", constrained_kwargs=None):
    """
    For a given model (blocks), calibration, unknowns, and targets, solve for the steady state values.

    blocks: `list`
        A list of blocks, which include the types: SimpleBlock, HetBlock, HelperBlock, SolvedBlock, CombinedBlock
    calibration: `dict`
        The pre-specified values of variables/parameters provided to the steady state computation
    unknowns: `dict`
        A dictionary mapping unknown variables to either initial values or bounds to be provided to the numerical solver
    targets: `dict`
        A dictionary mapping target variables to desired numerical values, other variables solved for along the DAG
    consistency_check: `bool`
        If HelperBlocks are a portion of the argument blocks, re-run the DAG with the computed steady state values
        without the assistance of HelperBlocks and see if the targets are still hit
    ttol: `float`
        The tolerance for the targets---how close the user wants the computed target values to equal the desired values
    ctol: `float`
        The tolerance for the consistency check---how close the user wants the computed target values, without the
        use of HelperBlocks, to equal the desired values
    block_kwargs: `dict`
        A dict of any kwargs that specify additional settings in order to evaluate block.steady_state for any
        potential Block object, e.g. HetBlocks have backward_tol and forward_tol settings that are specific to that
        Block sub-class.
    verbose: `bool`
        Display the content of optional print statements within the solver for more responsive feedback
    fragile: `bool`
        Throw errors instead of warnings when certain criteria are not met, i.e if the consistency_check fails
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

    # Populate otherwise mutable default arguments
    if block_kwargs is None:
        block_kwargs = {}
    if solver_kwargs is None:
        solver_kwargs = {}
    if constrained_kwargs is None:
        constrained_kwargs = {}

    ss_values = deepcopy(calibration)
    topsorted = graph.block_sort(blocks, calibration=calibration)

    def residual(unknown_values, include_helpers=True, update_unknowns_inplace=False):
        ss_values.update(misc.smart_zip(unknowns.keys(), unknown_values))

        helper_outputs = {}

        # Progress through the DAG computing the resulting steady state values based on the unknown_values
        # provided to the residual function
        for i in topsorted:
            if not include_helpers and isinstance(blocks[i], HelperBlock):
                continue
            else:
                outputs = eval_block_ss(blocks[i], ss_values, consistency_check=consistency_check,
                                        ttol=ttol, ctol=ctol, verbose=verbose, **block_kwargs)
                if include_helpers and isinstance(blocks[i], HelperBlock):
                    helper_outputs.update(outputs)
                    ss_values.update(outputs)
                else:
                    # Don't overwrite entries in ss_values corresponding to what has already
                    # been solved for in helper_blocks so we can check for consistency after-the-fact
                    ss_values.update(misc.dict_diff(outputs, helper_outputs))

        # Update the "unknowns" dictionary *in place* with its steady state values.
        # i.e. the "unknowns" in the namespace in which this function is invoked will change!
        # Useful for a) if the unknown values are updated while iterating each blocks' ss computation within the DAG,
        # and/or b) if the user wants to update "unknowns" in place for use in other computations.
        if update_unknowns_inplace:
            unknowns.update(misc.smart_zip(unknowns.keys(), [ss_values[key] for key in unknowns.keys()]))

        # Because in solve_for_unknowns, models that are fully "solved" (i.e. RBC) require the
        # dict of ss_values to compute the "unknown_solutions"
        return compute_target_values(targets, ss_values)

    unknown_solutions = _solve_for_unknowns(residual, unknowns, solver, solver_kwargs,
                                            constrained_method, constrained_kwargs,
                                            tol=ttol, verbose=verbose)

    # Check that the solution is consistent with what would come out of the DAG without the helper blocks
    if consistency_check:
        cresid = abs(np.max(residual(unknown_solutions, include_helpers=False)))
        run_consistency_check(cresid, ctol=ctol, fragile=fragile)

    # Update to set the solutions for the steady state values of the unknowns
    ss_values.update(zip(unknowns, misc.make_tuple(unknown_solutions)))

    # Find the hetoutputs of the Hetblocks that have hetoutputs
    for i in misc.find_blocks_with_hetoutputs(blocks):
        ss_values.update(eval_block_ss(blocks[i], ss_values, hetoutput=True, **block_kwargs))

    return ss_values


def eval_block_ss(block, calibration, **kwargs):
    """Evaluate the .ss method of a block, given a dictionary of potential arguments"""
    return block.steady_state({k: v for k, v in calibration.items() if k in block.inputs},
                              **{k: v for k, v in kwargs.items() if k in misc.input_kwarg_list(block.steady_state)})


def _solve_for_unknowns(residual, unknowns, solver, solver_kwargs,
                        constrained_method, constrained_kwargs,
                        tol=2e-12, verbose=False):
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
    tol: `float`
        The absolute convergence tolerance of the computed target to the desired target value in the numerical solver
    solver: `str`
        Refer to the `steady_state` function docstring for the "solver" variable
    solver_kwargs:
        Refer to the `steady_state` function docstring for the "solver_kwargs" variable

    return: The root[s] of the residual function as either a scalar (float) or a list of floats
    """
    scipy_optimize_uni_solvers = ["bisect", "brentq", "brenth", "ridder", "toms748", "newton", "secant", "halley"]
    scipy_optimize_multi_solvers = ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing", "diagbroyden",
                                    "excitingmixing", "krylov", "df-sane"]

    if solver is None:
        raise RuntimeError("Must provide a numerical solver from the following set: brentq, broyden, solved")
    elif solver in scipy_optimize_uni_solvers:
        initial_values_or_bounds = extract_univariate_initial_values_or_bounds(unknowns)
        result = opt.root_scalar(residual, method=solver, xtol=tol, **initial_values_or_bounds, **solver_kwargs)
        if not result.converged:
            raise ValueError(f"Steady-state solver, {solver}, did not converge.")
        unknown_solutions = result.root
    elif solver in scipy_optimize_multi_solvers:
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        # If no bounds were provided
        if not bounds:
            result = opt.root(residual, initial_values, method=solver, tol=tol, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            result = opt.root(constrained_residual, initial_values, method=solver, tol=tol, **solver_kwargs)
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
            unknown_solutions, _ = solvers.broyden_solver(residual, initial_values, tol=tol,
                                                          verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = solvers.broyden_solver(constrained_residual, initial_values,
                                                          verbose=verbose, tol=tol, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "newton_custom":
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        # If no bounds were provided
        if not bounds:
            unknown_solutions, _ = solvers.newton_solver(residual, initial_values, tol=tol,
                                                         verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = solvers.newton_solver(constrained_residual, initial_values,
                                                         verbose=verbose, tol=tol, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "solved":
        # If the entire solution is provided by the helper blocks
        # Call residual() once to update ss_values and to check the targets match the provided solution.
        # The initial value passed into residual (np.zeros()) in this case is irrelevant, but something
        # must still be passed in since the residual function requires an argument.
        assert abs(np.max(residual(misc.smart_zeros(len(unknowns)), update_unknowns_inplace=True))) < tol
        unknown_solutions = list(unknowns.values())
    else:
        raise RuntimeError(f"steady_state is not yet compatible with {solver}.")

    return unknown_solutions
