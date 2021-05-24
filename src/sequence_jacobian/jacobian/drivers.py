"""Main methods (drivers) for computing and manipulating both block-level and model-level Jacobians"""

import numpy as np

from .classes import JacobianDict
from .support import pack_vectors, unpack_vectors
from ..utilities import misc, graph

'''Drivers:
    - get_H_U               : get H_U matrix mapping all unknowns to all targets
    - get_impulse           : get single GE impulse response
    - get_G                 : get G matrices characterizing all GE impulse responses

    - curlyJs_sorted        : get block Jacobians curlyJ and return them topologically sorted
    - forward_accumulate    : forward accumulation on DAG, taking in topologically sorted Jacobians
'''


def get_H_U(blocks, unknowns, targets, T, ss=None, Js=None):
    """Get T*n_u by T*n_u matrix H_U, Jacobian mapping all unknowns to all targets.

    Parameters
    ----------
    blocks     : list, simple blocks, het blocks, or jacdicts
    unknowns   : list of str, names of unknowns in DAG
    targets    : list of str, names of targets in DAG
    T          : int, truncation horizon
                    (if asymptotic, truncation horizon for backward iteration in HetBlocks)
    ss         : [optional] dict, steady state required if blocks contains any non-jacdicts
    Js         : [optional] dict of {str: JacobianDict}}, supply saved Jacobians

    Returns
    -------
    H_U :
        if asymptotic=False:
            array(T*n_u*T*n_u) H_U, Jacobian mapping all unknowns to all targets
        is asymptotic=True:
            array((2*Tpost-1)*n_u*n_u), representation of asymptotic columns of H_U
    """

    # do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(blocks, unknowns, ss, T, Js)

    # do matrix forward accumulation to get H_U = J^(curlyH, curlyU)
    H_U_unpacked = forward_accumulate(curlyJs, unknowns, targets, required)

    # pack these n_u^2 matrices, each T*T, into a single matrix
    return H_U_unpacked[targets, unknowns].pack(T)


def get_impulse(blocks, dZ, unknowns, targets, T=None, ss=None, outputs=None, Js=None):
    """Get a single general equilibrium impulse response.

    Extremely fast when H_U_factored = utils.misc.factor(get_HU(...)) has already been computed
    and supplied to this function. Less so but still faster when H_U already computed.

    Parameters
    ----------
    blocks   : list, simple blocks or jacdicts
    dZ           : dict, path of an exogenous variable
    unknowns     : list of str, names of unknowns in DAG
    targets      : list of str, names of targets in DAG
    T            : [optional] int, truncation horizon
    ss           : [optional] dict, steady state required if blocks contains non-jacdicts
    outputs      : [optional] list of str, variables we want impulse responses for
    Js           : [optional] dict of {str: JacobianDict}}, supply saved Jacobians

    Returns
    -------
    out : dict, impulse responses to shock dZ
    """
    # step 0 (preliminaries): infer T, do topological sort and get curlyJs
    if T is None:
        for x in dZ.values():
            T = len(x)
            break

    curlyJs, required = curlyJ_sorted(blocks, unknowns + list(dZ.keys()), ss, T, Js)

    # step 1: do (matrix) forward accumulation to get H_U = J^(curlyH, curlyU)
    H_U_unpacked = forward_accumulate(curlyJs, unknowns, targets, required)

    # step 2: do (vector) forward accumulation to get J^(o, curlyZ)dZ for all o in
    # 'alloutputs', the combination of outputs (if specified) and targets
    alloutputs = None
    if outputs is not None:
        alloutputs = set(outputs) | set(targets)

    J_curlyZ_dZ = forward_accumulate(curlyJs, dZ, alloutputs, required)

    # step 3: solve H_UdU = -H_ZdZ for dU
    H_U = H_U_unpacked[targets, unknowns].pack(T)
    H_ZdZ_packed = pack_vectors(J_curlyZ_dZ, targets, T)
    dU_packed = -np.linalg.solve(H_U, H_ZdZ_packed)
    dU = unpack_vectors(dU_packed, unknowns, T)

    # step 4: do (vector) forward accumulation to get J^(o, curlyU)dU
    # then sum together with J^(o, curlyZ)dZ to get all output impulse responses
    J_curlyU_dU = forward_accumulate(curlyJs, dU, outputs, required)
    if outputs is None:
        outputs = J_curlyZ_dZ.keys() | J_curlyU_dU.keys()
    return {**dZ, **{o: J_curlyZ_dZ.get(o, np.zeros(T)) + J_curlyU_dU.get(o, np.zeros(T)) for o in outputs}}


def get_G(blocks, exogenous, unknowns, targets, T=300, ss=None, outputs=None, Js=None):
    """Compute Jacobians G that fully characterize general equilibrium outputs in response
    to all exogenous shocks in 'exogenous'

    Faster when H_U_factored = utils.misc.factor(get_HU(...)) has already been computed
    and supplied to this function. Less so but still faster when H_U already computed.
    Relative benefit of precomputing these not as extreme as for get_impulse, since
    obtaining and solving with H_U is a less dominant component of cost for getting Gs.

    Parameters
    ----------
    blocks   : list, simple blocks or jacdicts
    exogenous    : list of str, names of exogenous shocks in DAG
    unknowns     : list of str, names of unknowns in DAG
    targets      : list of str, names of targets in DAG
    T            : [optional] int, truncation horizon
    ss           : [optional] dict, steady state required if blocks contains non-jacdicts
    outputs      : [optional] list of str, variables we want impulse responses for
    Js           : [optional] dict of {str: JacobianDict}}, supply saved Jacobians

    Returns
    -------
    G : dict of dict, Jacobians for general equilibrium mapping from exogenous to outputs
    """

    # step 1: do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(blocks, unknowns + exogenous, ss, T, Js)

    # step 2: do (matrix) forward accumulation to get
    # H_U = J^(curlyH, curlyU) [if not provided], H_Z = J^(curlyH, curlyZ)
    J_curlyH_U = forward_accumulate(curlyJs, unknowns, targets, required)
    J_curlyH_Z = forward_accumulate(curlyJs, exogenous, targets, required)

    # step 3: solve for G^U, unpack
    H_U = J_curlyH_U[targets, unknowns].pack(T)
    H_Z = J_curlyH_Z[targets, exogenous].pack(T)

    G_U = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, exogenous, T)

    # step 4: forward accumulation to get all outputs starting with G_U
    # by default, don't calculate targets!
    curlyJs = [G_U] + curlyJs
    if outputs is None:
        outputs = set().union(*(curlyJ.outputs for curlyJ in curlyJs)) - set(targets)
    return forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))


def curlyJ_sorted(blocks, inputs, ss=None, T=None, Js=None):
    """
    Sort blocks along DAG and calculate their Jacobians (if not already provided) with respect to inputs
    and with respect to outputs of other blocks

    Parameters
    ----------
    blocks : list, simple blocks or jacdicts
    inputs     : list, input names we need to differentiate with respect to
    ss         : [optional] dict, steady state, needed if blocks includes blocks themselves
    T          : [optional] int, horizon for differentiation, needed if blocks includes hetblock itself
    Js         : [optional] dict of {str: JacobianDict}}, supply saved Jacobians

    Returns
    -------
    curlyJs : list of dict of dict, curlyJ for each block in order of topological sort
    required : list, outputs of some blocks that are needed as inputs by others
    """

    # step 1: get topological sort and required
    topsorted = graph.block_sort(blocks)
    required = graph.find_outputs_that_are_intermediate_inputs(blocks)

    # Remove any vector-valued outputs that are intermediate inputs, since we don't want
    # to compute Jacobians with respect to vector-valued variables
    if ss is not None:
        vv_vars = set([k for k, v in ss.items() if np.size(v) > 1])
        required -= vv_vars

    # step 2: compute Jacobians and put them in right order
    curlyJs = []
    shocks = set(inputs) | required
    for num in topsorted:
        block = blocks[num]
        jac = block.jacobian(ss, exogenous=list(shocks), Js=Js, **{k: v for k, v in {"T": T}.items()
                                                                   if k in misc.input_kwarg_list(block.jacobian)})

        # If the returned Jacobian is empty (i.e. the shocks do not affect any outputs from the block)
        # then don't add it to the list of curlyJs to be returned
        if not jac:
            continue
        else:
            curlyJs.append(JacobianDict(jac))

    return curlyJs, required


def forward_accumulate(curlyJs, inputs, outputs=None, required=None):
    """
    Use forward accumulation on topologically sorted Jacobians in curlyJs to get
    all cumulative Jacobians with respect to 'inputs' if inputs is a list of names,
    or get outcome of apply to 'inputs' if inputs is dict.

    Optionally only find outputs in 'outputs', especially if we have knowledge of
    what is required for later Jacobians.

    Note that the overloading of @ means that this works automatically whether curlyJs are ordinary
    matrices, simple_block.SimpleSparse objects, or asymptotic.AsymptoticTimeInvariant objects,
    as long as the first and third are not mixed (since multiplication not defined for them).

    Much-extended version of chain_jacobians.

    Parameters
    ----------
    curlyJs  : list of dict of dict, curlyJ for each block in order of topological sort
    inputs   : list or dict, input names to differentiate with respect to, OR dict of input vectors
    outputs  : [optional] list or set, outputs we're interested in
    required : [optional] list or set, outputs needed for later curlyJs (only useful w/outputs)

    Returns
    -------
    out : dict of dict or dict, either total J for each output wrt all inputs or
            outcome from applying all curlyJs
    """

    if outputs is not None and required is not None:
        # if list of outputs provided, we need to obtain these and 'required' along the way
        alloutputs = set(outputs) | set(required)
    else:
        # otherwise, set to None, implies default behavior of obtaining all outputs in curlyJs
        alloutputs = None

    # if inputs is list (jacflag=True), interpret as list of inputs for which we want to calculate jacs
    # if inputs is dict, interpret as input *paths* to which we apply all Jacobians in curlyJs
    jacflag = not isinstance(inputs, dict)

    if jacflag:
        # Jacobians of inputs with respect to themselves are the identity, initialize with this
        # out = {i: {i: utils.special_matrices.IdentityMatrix()} for i in inputs}
        out = JacobianDict.identity(inputs)
    else:
        out = inputs.copy()

    # iterate through curlyJs, in what is presumed to be a topologically sorted order
    for curlyJ in curlyJs:
        curlyJ = JacobianDict(curlyJ).complete()
        if alloutputs is not None:
            # if we want specific list of outputs, restrict curlyJ to that before continuing
            curlyJ = curlyJ[[k for k in alloutputs if k in curlyJ.outputs]]
        if jacflag:
            out.update(curlyJ.compose(out))
        else:
            out.update(curlyJ.apply(out))

    if outputs is not None:
        # if we want specific list of outputs, restrict to that
        # (dropping 'required' in 'alloutputs' that was needed for intermediate computations)
        return out[[k for k in outputs if k in out.outputs]]
    else:
        return out