import numpy as np
import utils
import jacobian as jac
import het_block as het

def td_map(ss, block_list, sort=None, monotonic=False, returnindividual=False, **kwargs):
    """Goes through block_list, topologically sorts the implied DAG, and evaluates non-SS paths in kwargs on it,
    with missing paths always being interpreted as remaining at the steady state for a particular variable"""

    hetoptions = {'monotonic' : monotonic, 'returnindividual' : returnindividual}

    # first get topological sort if none already provided
    if sort is None:
        sort = utils.block_sort(block_list)

    # initialize results
    results = kwargs
    for n in sort:
        block = block_list[n]

        # if this block is supposed to output something already there, that's bad (should only be because of kwargs)
        if not block.outputs.isdisjoint(results):
            raise ValueError(f'Block {block} outputting already-present outputs {block.outputs & results.keys()}')

        # if any input to the block has changed, run the block
        blockoptions = hetoptions if isinstance(block, het.HetBlock) else {}
        if not block.inputs.isdisjoint(results):
            results.update(block.td(ss, **blockoptions, **{k: results[k] for k in block.inputs if k in results}))

    return results


def td_solve(ss, block_list, unknowns, targets, H_U=None, H_U_factored=None, monotonic=False, returnindividual=False, tol=1E-8, maxit=30, noisy=True, **kwargs):
    # check to make sure that kwargs are valid shocks
    for x in unknowns + targets:
        if x in kwargs:
            raise ValueError(f'Shock {x} in td_solve cannot also be an unknown or target!')

    # for now, assume we have H_U, will deal with the rest later!
    if H_U_factored is None:
        H_U_factored = utils.factor(H_U)

    # initialize guess for unknowns at ss after inferring T
    T = H_U_factored[0].shape[0] // len(unknowns)
    Us = {k: np.full(T, ss[k]) for k in unknowns}
    Uvec = jac.pack_vectors(Us, unknowns, T)

    # do a topological sort once to avoid some redundancy
    sort = utils.block_sort(block_list)

    # iterate until convergence
    for it in range(maxit):
        results = td_map(ss, block_list, sort, monotonic, returnindividual, **kwargs, **Us)
        errors = {k: np.max(np.abs(results[k])) for k in targets}
        if noisy:
            print(f'On iteration {it}')
            for k in errors:
                print(f'   max error for {k} is {errors[k]:.2E}')
        if all(v < tol for v in errors.values()):
            break
        else:
            Hvec = jac.pack_vectors(results, targets, T)
            Uvec -= utils.factored_solve(H_U_factored, Hvec)
            Us = jac.unpack_vectors(Uvec, unknowns, T)
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')
    
    return results