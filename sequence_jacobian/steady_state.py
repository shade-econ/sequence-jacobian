"""A general function for computing a model's steady state variables and parameters values"""

import numpy as np
import scipy.optimize as opt
from copy import deepcopy

from . import utilities as utils
from .utilities.misc import unprime, dict_diff, smart_zip, smart_zeros, find_blocks_with_hetoutputs
from .blocks.simple_block import SimpleBlock
from .blocks.helper_block import HelperBlock
from .blocks.het_block import HetBlock


# Find the steady state solution
def steady_state(blocks, calibration, unknowns, targets,
                 consistency_check=True, ttol=1e-9, ctol=1e-9,
                 verbose=False, solver=None, solver_kwargs=None,
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
    if solver_kwargs is None:
        solver_kwargs = {}
    if constrained_kwargs is None:
        constrained_kwargs = {}

    ss_values = deepcopy(calibration)
    topsorted = utils.graph.block_sort(blocks, calibration=calibration)

    def residual(unknown_values, include_helpers=True, update_unknowns_inplace=False):
        ss_values.update(smart_zip(unknowns.keys(), unknown_values))

        helper_outputs = {}

        # Progress through the DAG computing the resulting steady state values based on the unknown_values
        # provided to the residual function
        for i in topsorted:
            if not include_helpers and isinstance(blocks[i], HelperBlock):
                continue
            else:
                outputs = eval_block_ss(blocks[i], ss_values, consistency_check=consistency_check,
                                        ttol=ttol, ctol=ctol, verbose=verbose)
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

    unknown_solutions = _solve_for_unknowns(residual, unknowns, solver, solver_kwargs,
                                            constrained_method, constrained_kwargs,
                                            tol=ttol, verbose=verbose)

    # Check that the solution is consistent with what would come out of the DAG without the helper blocks
    if consistency_check:
        assert abs(np.max(residual(unknown_solutions, include_helpers=False))) < ctol

    # Update to set the solutions for the steady state values of the unknowns
    ss_values.update(zip(unknowns, utils.misc.make_tuple(unknown_solutions)))

    # Find the hetoutputs of the Hetblocks that have hetoutputs
    for i in find_blocks_with_hetoutputs(blocks):
        ss_values.update(eval_block_ss(blocks[i], ss_values, hetoutput=True))

    return ss_values


def find_target_block(blocks, target):
    for block in blocks:
        if target in blocks.output:
            return block


# Allow targets to be specified in the following formats
# 1) target = {"asset_mkt": 0} or ["asset_mkt"] (the standard case, where the target = 0)
# 2) target = {"r": 0.01} (allowing for the target to be non-zero)
# 3) target = {"K": "A"} (allowing the target to be another variable in potential_args)
def compute_target_values(targets, potential_args):
    """
    For a given set of target specifications and potential arguments available, compute the targets.
    Called as the return value for the residual function when utilizing the numerical solver.

    targets: Refer to `steady_state` function docstring
    potential_args: Refer to the `steady_state` function docstring for the "calibration" variable

    return: A `float` (if computing a univariate target) or an `np.ndarray` (if using a multivariate target)
    """
    target_values = np.empty(len(targets))
    for (i, t) in enumerate(targets):
        v = targets[t] if isinstance(targets, dict) else 0
        if type(v) == str:
            target_values[i] = potential_args[t] - potential_args[v]
        else:
            target_values[i] = potential_args[t] - v
        # TODO: Implement feature to allow for an arbitrary explicit function expression as a potential target value
        #   e.g. targets = {"goods_mkt": "Y - C - I"}, so long as the expression is only comprise of generic numerical
        #   operators and variables solved for along the DAG prior to reaching the target.

    # Univariate solvers require float return values (and not lists)
    if len(targets) == 1:
        return target_values[0]
    else:
        return target_values

# Analogous to the SHADE workflow of having blocks call utils.apply(self._fss, inputs) but not as general.
def eval_block_ss(block, potential_args, consistency_check=True, ttol=1e-9, ctol=1e-9, verbose=False, **kwargs):
    """
    Evaluate the .ss method of a block, given a dictionary of potential arguments.

    Refer to the `steady_state` function docstring for information on args/kwargs

    return: A `dict` of output names (as `str`) and output values from evaluating the .ss method of a block
    """
    input_args = {unprime(arg_name): potential_args[unprime(arg_name)] for arg_name in block.inputs}

    # Simple and HetBlocks require different handling of block.ss() output since
    # SimpleBlocks return a tuple of un-labeled arguments, whereas HetBlocks return dictionaries
    if isinstance(block, SimpleBlock) or isinstance(block, HelperBlock):
        output_args = utils.misc.make_tuple(block.ss(**input_args, **kwargs))
        outputs = {o: output_args[i] for i, o in enumerate(block.output_list)}
    else:  # assume it's a HetBlock or a SolvedBlock
        if isinstance(block, HetBlock):  # since .ss for SolvedBlocks calls the steady_state driver function
            outputs = block.ss(**input_args, **kwargs)
        else:  # since .ss for SolvedBlocks calls the steady_state driver function
            outputs = block.ss(**input_args, consistency_check=consistency_check,
                               ttol=ttol, ctol=ctol, verbose=verbose, **kwargs)

    return outputs


def _solve_for_unknowns(residual, unknowns, solver, solver_kwargs,
                        constrained_method, constrained_kwargs,
                        tol=1e-9, verbose=False):
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
            unknown_solutions, _ = utils.solvers.broyden_solver(residual, initial_values, tol=tol,
                                                                verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = utils.solvers.broyden_solver(constrained_residual, initial_values,
                                                                verbose=verbose, tol=tol, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "newton_custom":
        initial_values, bounds = extract_multivariate_initial_values_and_bounds(unknowns)
        # If no bounds were provided
        if not bounds:
            unknown_solutions, _ = utils.solvers.newton_solver(residual, initial_values, tol=tol,
                                                               verbose=verbose, **solver_kwargs)
        else:
            constrained_residual = constrained_multivariate_residual(residual, bounds, verbose=verbose,
                                                                     method=constrained_method,
                                                                     **constrained_kwargs)
            unknown_solutions, _ = utils.solvers.newton_solver(constrained_residual, initial_values,
                                                               verbose=verbose, tol=tol, **solver_kwargs)
        unknown_solutions = list(unknown_solutions)
    elif solver == "solved":
        # If the entire solution is provided by the helper blocks
        # Call residual() once to update ss_values and to check the targets match the provided solution.
        # The initial value passed into residual (np.zeros()) in this case is irrelevant, but something
        # must still be passed in since the residual function requires an argument.
        assert abs(np.max(residual(smart_zeros(len(unknowns)), update_unknowns_inplace=True))) < tol
        unknown_solutions = list(unknowns.values())
    else:
        raise RuntimeError(f"steady_state is not yet compatible with {solver}.")

    return unknown_solutions


def extract_univariate_initial_values_or_bounds(unknowns):
    val = next(iter(unknowns.values()))
    if np.isscalar(val):
        return {"x0": val}
    else:
        return {"bracket": (val[0], val[1])}


def extract_multivariate_initial_values_and_bounds(unknowns):
    """Provided a dict mapping names of unknowns to initial values/bounds, return separate dicts of
    the initial values and bounds.
    Note: For one-sided bounds, simply put np.inf/-np.inf as the other side of the bounds, so there is
    no ambiguity about which is the unconstrained side.
"""
    initial_values = []
    multi_bounds = {}
    for k, v in unknowns.items():
        if np.isscalar(v):
            initial_values.append(v)
        elif len(v) == 3:
            lb, iv, ub = v
            assert lb < iv < ub
            initial_values.append(iv)
            multi_bounds[k] = (lb, ub)
        else:
            raise ValueError(f"{len(v)} is an invalid size for the value of an unknown."
                             f" the values of `unknowns` must either be a scalar, pertaining to a"
                             f" single initial value for the root solver to begin from,"
                             f" or a length 3 tuple, pertaining to a lower bound, initial value, and upper bound.")

    return np.asarray(initial_values), multi_bounds


def residual_with_linear_continuation(residual, bounds, eval_at_boundary=False,
                                      boundary_epsilon=1e-4, penalty_scale=1e1,
                                      verbose=False):
    """Modify a residual function to implement bounds by an additive penalty for exceeding the boundaries
    provided, scaled by the amount the guess exceeds the boundary.

    e.g. For residual function f(x), desiring x in (0, 1) (so assuming eval_at_boundary = False)
         If the guess for x is 1.1 then we will censor to x_censored = 1 - boundary_epsilon, and return
         f(x_censored) + penalty (where the penalty does not require re-evaluating f() which may be costly)

    residual: `function`
        The function whose roots we want to solve for
    bounds: `dict`
        A dict mapping the names of the unknowns (`str`) to length two tuples corresponding to the lower and upper
        bounds.
    eval_at_boundary: `bool`
        Whether to allow the residual function to be evaluated at exactly the boundary values or not.
        Think of it as whether the solver will treat the bounds as creating a closed or open set for the search space.
    boundary_epsilon: `float`
        The amount to adjust the proposed guess, x, by to calculate the censored value of the residual function,
        when the proposed guess exceeds the boundaries.
    penalty_scale: `float`
        The linear scaling factor for adjusting the penalty for the proposed unknown values exceeding the boundary.
    verbose: `bool`
        Whether to print out additional information for how the constrained residual function is behaving during
        optimization. Useful for tuning the solver.
    """
    lbs = np.asarray([v[0] for v in bounds.values()])
    ubs = np.asarray([v[1] for v in bounds.values()])

    def constr_residual(x, residual_cache=[]):
        """Implements a constrained residual function, where any attempts to evaluate x outside of the
        bounds provided will result in a linear penalty function scaled by `penalty_scale`.

        Note: We are purposefully using residual_cache as a mutable default argument to cache the most recent
        valid evaluation (maintain state between function calls) of the residual function to induce solvers
        to backstep if they encounter a region of the search space that returns nan values.
        See Hitchhiker's Guide to Python post on Mutable Default Arguments: "When the Gotcha Isn't a Gotcha"
        """
        if eval_at_boundary:
            x_censored = np.where(x < lbs, lbs, x)
            x_censored = np.where(x > ubs, ubs, x_censored)
        else:
            x_censored = np.where(x < lbs, lbs + boundary_epsilon, x)
            x_censored = np.where(x > ubs, ubs - boundary_epsilon, x_censored)

        residual_censored = residual(x_censored)

        if verbose:
            print(f"Attempted x is {x}")
            print(f"Censored x is {x_censored}")
            print(f"The residual_censored is {residual_censored}")

        if np.any(np.isnan(residual_censored)):
            # Provide a scaled penalty to the solver when trying to evaluate residual() in an undefined region
            residual_censored = residual_cache[0] * penalty_scale

            if verbose:
                print(f"The new residual_censored is {residual_censored}")
        else:
            if not residual_cache:
                residual_cache.append(residual_censored)
            else:
                residual_cache[0] = residual_censored

        if verbose:
            print(f"The residual_cache is {residual_cache[0]}")

        # Provide an additive, scaled penalty to the solver when trying to evaluate residual() outside of the boundary
        residual_with_boundary_penalty = residual_censored + \
                                         (x - x_censored) * penalty_scale * residual_censored
        return residual_with_boundary_penalty

    return constr_residual


def constrained_multivariate_residual(residual, bounds, method="linear_continuation", verbose=False,
                                      **constrained_kwargs):
    """Return a constrained version of the residual function, which accounts for bounds, using the specified method.
    See the docstring of the specific method of interest for further details."""
    if method == "linear_continuation":
        return residual_with_linear_continuation(residual, bounds, verbose=verbose, **constrained_kwargs)
    # TODO: Implement logistic transform as another option for constrained multivariate residual
    else:
        raise ValueError(f"Method {method} for constrained multivariate root-finding has not yet been implemented.")
