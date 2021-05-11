"""Various lower-level functions to support the computation of steady states"""

import warnings
from numbers import Real
import numpy as np


def instantiate_steady_state_mutable_kwargs(dissolve, helper_blocks, helper_targets,
                                            block_kwargs, solver_kwargs, constrained_kwargs):
    """Instantiate mutable types from `None` default values in the steady_state function"""
    if dissolve is None:
        dissolve = []
    if helper_blocks is None and helper_targets is None:
        helper_blocks = []
        helper_targets = []
    elif helper_blocks is not None and helper_targets is None:
        raise ValueError("If the user has provided `helper_blocks`, the kwarg `helper_targets` must be specified"
                         " indicating which target variables are handled by the `helper_blocks`.")
    elif helper_blocks is None and helper_targets is not None:
        raise ValueError("If the user has provided `helper_targets`, the kwarg `helper_blocks` must be specified"
                         " indicating which helper blocks handle the `helper_targets`")
    if block_kwargs is None:
        block_kwargs = {}
    if solver_kwargs is None:
        solver_kwargs = {}
    if constrained_kwargs is None:
        constrained_kwargs = {}

    return dissolve, helper_blocks, helper_targets, block_kwargs, solver_kwargs, constrained_kwargs


def provide_solver_default(unknowns):
    if len(unknowns) == 1:
        bounds = list(unknowns.values())[0]
        if not isinstance(bounds, tuple) or bounds[0] > bounds[1]:
            raise ValueError("Unable to find a compatible one-dimensional solver with provided `unknowns`.\n"
                             " Please provide valid lower/upper bounds, e.g. unknowns = {`a`: (0, 1)}")
        else:
            return "brentq"
    elif len(unknowns) > 1:
        init_values = list(unknowns.values())
        if not np.all([isinstance(v, Real) for v in init_values]):
            raise ValueError("Unable to find a compatible multi-dimensional solver with provided `unknowns`.\n"
                             " Please provide valid initial values, e.g. unknowns = {`a`: 1, `b`: 2}")
        else:
            return "broyden_custom"
    else:
        raise ValueError("`unknowns` is empty! Please provide a dict of keys/values equal to the number of unknowns"
                         " that need to be solved for.")


def run_consistency_check(cresid, ctol=1e-9, fragile=False):
    if cresid > ctol:
        if fragile:
            raise RuntimeError(f"The target values evaluated for the proposed set of unknowns produce a "
                               f"maximum residual value of {cresid}, which is greater than the ctol {ctol}.\n"
                               f" If used, check if HelperBlocks are indeed compatible with the DAG.\n"
                               f" If this is not an issue, adjust ctol accordingly.")
        else:
            warnings.warn(f"The target values evaluated for the proposed set of unknowns produce a "
                          f"maximum residual value of {cresid}, which is greater than the ctol {ctol}.\n"
                          f" If used, check if HelperBlocks are indeed compatible with the DAG.\n"
                          f" If this is not an issue, adjust ctol accordingly.")


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


def subset_helper_block_unknowns(unknowns_all, helper_blocks, helper_targets):
    """Find the set of unknowns that the `helper_blocks` solve for"""
    unknowns_handled_by_helpers = {}
    for block in helper_blocks:
        unknowns_handled_by_helpers.update({u: unknowns_all[u] for u in block.outputs if u in unknowns_all})

    n_unknowns = len(unknowns_handled_by_helpers)
    n_targets = len(helper_targets)
    if n_unknowns != n_targets:
        raise ValueError(f"The provided helper_blocks handle {n_unknowns} unknowns != {n_targets} targets."
                         f" User must specify an equal number of unknowns/targets solved for by helper blocks.")

    return unknowns_handled_by_helpers


def find_excludable_helper_blocks(blocks_all, helper_indices, helper_unknowns, helper_targets):
    """Of the set of helper_unknowns and helper_targets, find the ones that can be excluded from the main DAG
    for the purposes of numerically solving unknowns."""
    excludable_helper_unknowns = {}
    excludable_helper_targets = {}
    for i in helper_indices:
        excludable_helper_unknowns.update({h: helper_unknowns[h] for h in blocks_all[i].outputs if h in helper_unknowns})
        excludable_helper_targets.update({h: helper_targets[h] for h in blocks_all[i].outputs | blocks_all[i].inputs if h in helper_targets})
    return excludable_helper_unknowns, excludable_helper_targets


def extract_univariate_initial_values_or_bounds(unknowns):
    val = next(iter(unknowns.values()))
    if np.isscalar(val):
        return {"x0": val}
    else:
        return {"bracket": (val[0], val[1])}


def extract_multivariate_initial_values_and_bounds(unknowns, fragile=False):
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
        elif len(v) == 2:
            if fragile:
                raise ValueError(f"{len(v)} is an invalid size for the value of an unknown."
                                 f" the values of `unknowns` must either be a scalar, pertaining to a"
                                 f" single initial value for the root solver to begin from,"
                                 f" a length 2 tuple, pertaining to a lower bound and an upper bound,"
                                 f" or a length 3 tuple, pertaining to a lower bound, initial value, and upper bound.")
            else:
                warnings.warn("Interpreting values of `unknowns` from length 2 tuple as lower and upper bounds"
                              " and averaging them to get a scalar initial value to provide to the solver.")
                initial_values.append((v[0] + v[1])/2)
        elif len(v) == 3:
            lb, iv, ub = v
            assert lb < iv < ub
            initial_values.append(iv)
            multi_bounds[k] = (lb, ub)
        else:
            raise ValueError(f"{len(v)} is an invalid size for the value of an unknown."
                             f" the values of `unknowns` must either be a scalar, pertaining to a"
                             f" single initial value for the root solver to begin from,"
                             f" a length 2 tuple, pertaining to a lower bound and an upper bound,"
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
