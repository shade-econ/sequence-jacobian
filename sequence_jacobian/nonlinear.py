"""Functions for solving for the non-linear transition dynamics provided a given shock path (e.g. solving MIT shocks)"""

import numpy as np

from .utilities import misc, graph
from .jacobian.drivers import get_H_U
from .jacobian.support import pack_vectors, unpack_vectors


def td_solve(block_list, ss, exogenous, unknowns, targets, H_U=None, H_U_factored=None, monotonic=False,
             returnindividual=False, tol=1E-8, maxit=30, verbose=True, save=False, use_saved=False,
             grid_paths=None):
    """Solves for GE nonlinear perfect foresight paths for SHADE model, given shocks in kwargs.

    Use a quasi-Newton method with the Jacobian H_U mapping unknowns to targets around steady state.
    
    Parameters
    ----------
    block_list      : list, blocks in model (SimpleBlocks or HetBlocks)
    ss              : dict, all steady-state information
    exogenous       : dict, all shocked Z go here, must all have same length T
    unknowns        : list, unknowns of SHADE DAG, the 'U' in H(U, Z)
    targets         : list, targets of SHADE DAG, the 'H' in H(U, Z)
    H_U             : [optional] array (nU*nU), Jacobian of targets with respect to unknowns
    H_U_factored    : [optional] tuple, LU decomposition of H_U, save time by supplying this from utils.misc.factor()
    monotonic       : [optional] bool, flag indicating HetBlock policy for some k' is monotonic in state k
                                                                        (allows more efficient interpolation)
    returnindividual: [optional] bool, flag to return individual outcomes from HetBlock.td
    tol             : [optional] scalar, for convergence of Newton's method we require |H|<tol
    maxit           : [optional] int, maximum number of iterations of Newton's method
    verbose           : [optional] bool, flag to print largest absolute error for each target
    save            : [optional] bool, flag for saving Jacobians inside HetBlocks during calc of H_U
    use_saved       : [optional] bool, flag for using saved Jacobians inside HetBlocks during calc of H_U

    Returns
    ----------
    results : dict, return paths for all aggregate variables, plus individual outcomes of HetBlock if returnindividual
    """

    # check to make sure that exogenous are valid shocks
    for x in unknowns + targets:
        if x in exogenous:
            raise ValueError(f'Shock {x} in td_solve cannot also be an unknown or target!')

    # infer T from a single shocked Z in exogenous
    for v in exogenous.values():
        T = v.shape[0]
        break
    
    # initialize guess for unknowns to steady state length T
    unknown_paths = {k: np.full(T, ss[k]) for k in unknowns}
    Uvec = pack_vectors(unknown_paths, unknowns, T)

    # obtain H_U_factored if we don't have it already 
    if H_U_factored is None:
        if H_U is None:
            # not even H_U is supplied, get it (costly if there are HetBlocks)
            H_U = get_H_U(block_list, unknowns, targets, T, ss, save=save, use_saved=use_saved)
        H_U_factored = misc.factor(H_U)

    # do a topological sort once to avoid some redundancy
    sort = graph.block_sort(block_list)

    # iterate until convergence
    for it in range(maxit):
        results = td_map(block_list, ss, exogenous, unknown_paths, sort=sort,
                         monotonic=monotonic, returnindividual=returnindividual,
                         grid_paths=grid_paths)
        errors = {k: np.max(np.abs(results[k])) for k in targets}
        if verbose:
            print(f'On iteration {it}')
            for k in errors:
                print(f'   max error for {k} is {errors[k]:.2E}')
        if all(v < tol for v in errors.values()):
            break
        else:
            # update guess U by -H_U^(-1) times errors H
            Hvec = pack_vectors(results, targets, T)
            Uvec -= misc.factored_solve(H_U_factored, Hvec)
            unknown_paths = unpack_vectors(Uvec, unknowns, T)
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')
    
    return results


def td_map(block_list, ss, exogenous, unknowns=None, sort=None,
           monotonic=False, returnindividual=False, grid_paths=None):
    """Helper for td_solve, calculates H(U, Z), where U and Z are in kwargs.
    
    Goes through block_list, topologically sorts the implied DAG, calculates H(U, Z),
    with missing paths always being interpreted as remaining at the steady state for a particular variable"""

    if unknowns is None:
        unknowns = {}

    hetoptions = {'monotonic': monotonic, 'returnindividual': returnindividual, 'grid_paths': grid_paths}

    # first get topological sort if none already provided
    if sort is None:
        sort = graph.block_sort(block_list)

    # initialize results
    results = {**exogenous, **unknowns}
    for n in sort:
        block = block_list[n]

        # if this block is supposed to output something already there, that's bad (should only be because of kwargs)
        if not block.outputs.isdisjoint(results):
            raise ValueError(f'Block {block} outputting already-present outputs {block.outputs & results.keys()}')

        # if any input to the block has changed, run the block
        if not block.inputs.isdisjoint(results):
            results.update(block.impulse_nonlinear(ss, exogenous={k: results[k] - ss[k] for k in block.inputs if k in results},
                                                   **{k: v for k, v in hetoptions.items()
                                                      if k in misc.input_kwarg_list(block.impulse_nonlinear)}))

    return results
