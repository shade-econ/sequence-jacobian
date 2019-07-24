import numpy as np
import copy
import utils
import simple_block as sim
import het_block as het
import solved_block as sol
import asymptotic


'''Part 1: High-level convenience routines: 
    - get_H_U               : get H_U matrix mapping all unknowns to all targets
    - get_impulse           : get single GE impulse response
    - get_G                 : get G matrices characterizing all GE impulse responses

    - curlyJs_sorted        : get block Jacobians curlyJ and return them topologically sorted
    - forward_accumulate    : forward accumulation on DAG, taking in topologically sorted Jacobians
'''


def get_H_U(block_list, unknowns, targets, T, ss=None, asymptotic=False, Tpost=None, save=False, use_saved=False):
    """Get n_u*T by n_u*T matrix H_U, Jacobian mapping all unknowns to all targets.

    Parameters
    ----------
    block_list : list, simple blocks, het blocks, or jacdicts
    unknowns   : list of str, names of unknowns in DAG
    targets    : list of str, names of targets in DAG
    T          : int, truncation horizon
    ss         : [optional] dict, steady state required if block_list contains any non-jacdicts

    Returns
    -------
    H_U : array H_U, Jacobian mapping all unknowns to all targets.
    """

    # do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(block_list, unknowns, ss, T, asymptotic, Tpost, save, use_saved)

    # do matrix forward accumulation to get H_U = J^(curlyH, curlyU)
    H_U_unpacked = forward_accumulate(curlyJs, unknowns, targets, required)

    # "pack" these n_u*n_u matrices, each T*T, into a single matrix
    if not asymptotic:
        return pack_jacobians(H_U_unpacked, unknowns, targets, T)
    else:
        if Tpost is None:
            Tpost = 2*T
        return pack_asymptotic_jacobians(H_U_unpacked, unknowns, targets, Tpost)


def get_impulse(block_list, dZ, unknowns, targets, T=None, ss=None, outputs=None,
                                        H_U=None, H_U_factored=None, save=False, use_saved=False):
    """Get a single general equilibrium impulse response.

    Extremely fast when H_U_factored = utils.factor(get_HU(...)) has already been computed
    and supplied to this function. Less so but still faster when H_U already computed.

    Parameters
    ----------
    block_list   : list, simple blocks or jacdicts
    dZ           : dict, path of an exogenous variable
    unknowns     : list of str, names of unknowns in DAG
    targets      : list of str, names of targets in DAG
    T            : [optional] int, truncation horizon
    ss           : [optional] dict, steady state required if block_list contains non-jacdicts
    outputs      : [optional] list of str, variables we want impulse responses for
    H_U          : [optional] array, precomputed Jacobian mapping unknowns to targets
    H_U_factored : [optional] tuple of arrays, precomputed LU factorization utils.factor(H_U)

    Returns
    -------
    out : dict, impulse responses to shock dZ
    """
    # step 0 (preliminaries): infer T, do topological sort and get curlyJs
    if T is None:
        for x in dZ.values():
            T = len(x)
            break

    curlyJs, required = curlyJ_sorted(block_list, unknowns + list(dZ.keys()), ss, T,
                                      save=save, use_saved=use_saved)

    # step 1: if not provided, do (matrix) forward accumulation to get H_U = J^(curlyH, curlyU)
    if H_U is None and H_U_factored is None:
        H_U_unpacked = forward_accumulate(curlyJs, unknowns, targets, required)

    # step 2: do (vector) forward accumulation to get J^(o, curlyZ)dZ for all o in
    # 'alloutputs', the combination of outputs (if specified) and targets
    alloutputs = None
    if outputs is not None:
        alloutputs = set(outputs) | set(targets)

    J_curlyZ_dZ = forward_accumulate(curlyJs, dZ, alloutputs, required)

    # step 3: solve H_UdU = -H_ZdZ for dU
    if H_U is None and H_U_factored is None:
        H_U = pack_jacobians(H_U_unpacked, unknowns, targets, T)
    
    H_ZdZ_packed = pack_vectors(J_curlyZ_dZ, targets, T)

    if H_U_factored is None:
        dU_packed = - np.linalg.solve(H_U, H_ZdZ_packed)
    else:
        dU_packed = - utils.factored_solve(H_U_factored, H_ZdZ_packed)

    dU = unpack_vectors(dU_packed, unknowns, T)

    # step 4: do (vector) forward accumulation to get J^(o, curlyU)dU
    # then sum together with J^(o, curlyZ)dZ to get all output impulse responses
    J_curlyU_dU = forward_accumulate(curlyJs, dU, outputs, required)
    if outputs is None:
        outputs = J_curlyZ_dZ.keys() | J_curlyU_dU.keys()
    return {o: J_curlyZ_dZ.get(o, np.zeros(T)) + J_curlyU_dU.get(o, np.zeros(T)) for o in outputs}


def get_G(block_list, exogenous, unknowns, targets, T, ss=None, outputs=None, 
                            H_U=None, H_U_factored=None, save=False, use_saved=False):
    """Compute Jacobians G that fully characterize general equilibrium outputs in response
    to all exogenous shocks in 'exogenous'

    Faster when H_U_factored = utils.factor(get_HU(...)) has already been computed
    and supplied to this function. Less so but still faster when H_U already computed.
    Relative benefit of precomputing these not as extreme as for get_impulse, since
    obtaining and solving with H_U is a less dominant component of cost for getting Gs.

    Parameters
    ----------
    block_list   : list, simple blocks or jacdicts
    exogenous    : list of str, names of exogenous shocks in DAG
    unknowns     : list of str, names of unknowns in DAG
    targets      : list of str, names of targets in DAG
    T            : [optional] int, truncation horizon
    ss           : [optional] dict, steady state required if block_list contains non-jacdicts
    outputs      : [optional] list of str, variables we want impulse responses for
    H_U          : [optional] array, precomputed Jacobian mapping unknowns to targets
    H_U_factored : [optional] tuple of arrays, precomputed LU factorization utils.factor(H_U)

    Returns
    -------
    G : dict of dict, Jacobians for general equilibrium mapping from exogenous to outputs
    """

    # step 1: do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(block_list, unknowns + exogenous, ss, T,
                                      save=save, use_saved=use_saved)

    # step 2: do (matrix) forward accumulation to get
    # H_U = J^(curlyH, curlyU) [if not provided], H_Z = J^(curlyH, curlyZ)
    if H_U is None and H_U_factored is None:
        J_curlyH_U = forward_accumulate(curlyJs, unknowns, targets, required)
    J_curlyH_Z = forward_accumulate(curlyJs, exogenous, targets, required)

    # step 3: solve for G^U, unpack
    if H_U is None and H_U_factored is None:
        H_U = pack_jacobians(J_curlyH_U, unknowns, targets, T)
    H_Z = pack_jacobians(J_curlyH_Z, exogenous, targets, T)

    if H_U_factored is None:
        G_U = unpack_jacobians(-np.linalg.solve(H_U, H_Z), exogenous, unknowns, T)
    else:
        G_U = unpack_jacobians(-utils.factored_solve(H_U_factored, H_Z), exogenous, unknowns, T)

    # step 4: forward accumulation to get all outputs starting with G_U
    # by default, don't calculate targets!
    curlyJs = [G_U] + curlyJs
    if outputs is None:
        outputs = set().union(*(curlyJ.keys() for curlyJ in curlyJs)) - set(targets)
    return forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))


def curlyJ_sorted(block_list, inputs, ss=None, T=None, asymptotic=False, Tpost=None, save=False, use_saved=False):
    """
    Sort blocks along DAG and calculate their Jacobians (if not already provided) with respect to inputs
    and with respect to outputs of other blocks

    Parameters
    ----------
    block_list : list, simple blocks or jacdicts
    inputs     : list, input names we need to differentiate with respect to
    ss         : [optional] dict, steady state, needed if block_list includes blocks themselves
    T          : [optional] int, horizon for differentiation, needed if block_list includes hetblock itself

    Returns
    -------
    curlyJs : list of dict of dict, curlyJ for each block in order of topological sort
    required : list, outputs of some blocks that are needed as inputs by others
    """

    # step 1: get topological sort and required
    topsorted, required = utils.block_sort(block_list, findrequired=True)

    # step 2: compute Jacobians and put them in right order
    curlyJs = []
    shocks = set(inputs) | required
    for num in topsorted:
        block = block_list[num]
        if isinstance(block, sim.SimpleBlock):
            jac = block.jac(ss, shock_list=[i for i in block.inputs if i in shocks])
        elif isinstance(block, het.HetBlock) or isinstance(block, sol.SolvedBlock):
            if asymptotic:
                jac = block.ajac(ss, T=T,
                                 shock_list=[i for i in block.inputs if i in shocks], Tpost=Tpost, save=save, use_saved=use_saved)
            else:
                jac = block.jac(ss, T=T,
                                shock_list=[i for i in block.inputs if i in shocks], save=save, use_saved=use_saved)
        else:
            jac = block
        curlyJs.append(jac)

    return curlyJs, required


def forward_accumulate(curlyJs, inputs, outputs=None, required=None):
    """
    Use forward accumulation on topologically sorted Jacobians in curlyJs to get
    all cumulative Jacobians with respect to 'inputs' if inputs is a list of names,
    or get outcome of apply to 'inputs' if inputs is dict.

    Optionally only find outputs in 'outputs', especially if we have knowledge of
    what is required for later Jacobians.

    Much-extended version of chain_jacobians.

    Parameters
    ----------
    curlyJs  : list of dict of dict, curlyJ for each block in order of topological sort
    inputs   : list or dict, input names to differentiate with respect to, OR dict of input vectors
    outputs  : [optional] list or set, outputs we're interested in
    required : [optional] list or set, outputs needed for later curlyJs (only useful w/outputs)

    Returns
    -------
    out : dict of dict or dict, either total J for each output wrt all inputs or outcome from applying all curlyJs
    """

    if outputs is not None and required is not None:
        alloutputs = set(outputs) | set(required)
    else:
        alloutputs = None

    jacflag = not isinstance(inputs, dict)

    if jacflag:
        out = {i: {i: IdentityMatrix()} for i in inputs}
    else:
        out = inputs.copy()

    for curlyJ in curlyJs:
        if alloutputs is not None:
            curlyJ = {k: v for k, v in curlyJ.items() if k in alloutputs}
        if jacflag:
            out.update(compose_jacobians(out, curlyJ))
        else:
            out.update(apply_jacobians(curlyJ, out))

    if outputs is not None:
        return {k: out[k] for k in outputs if k in out}
    else:
        if jacflag:
            return {k: v for k, v in out.items() if k not in inputs}
        else:
            return out


'''Part 2: Somewhat lower-level routines for handling Jacobians'''


def chain_jacobians(jacdicts, inputs):
    """Obtain complete Jacobian of every output in jacdicts with respect to inputs, by applying chain rule."""
    cumulative_jacdict = {i: {i: IdentityMatrix()} for i in inputs}
    for jacdict in jacdicts:
        cumulative_jacdict.update(compose_jacobians(cumulative_jacdict, jacdict))
    return cumulative_jacdict


def compose_jacobians(jacdict2, jacdict1):
    """Compose Jacobians via the chain rule."""
    jacdict = {}
    for output, innerjac1 in jacdict1.items():
        jacdict[output] = {}
        for middle, jac1 in innerjac1.items():
            innerjac2 = jacdict2.get(middle, {})
            for inp, jac2 in innerjac2.items():
                if inp in jacdict[output]:
                    jacdict[output][inp] += jac1 @ jac2
                else:
                    jacdict[output][inp] = jac1 @ jac2
    return jacdict


def apply_jacobians(jacdict, indict):
    """Apply Jacobians in jacdict to indict to obtain outputs."""
    outdict = {}
    for myout, innerjacdict in jacdict.items():
        for myin, jac in innerjacdict.items():
            if myin in indict:
                if myout in outdict:
                    outdict[myout] += jac @ indict[myin]
                else:
                    outdict[myout] = jac @ indict[myin]

    return outdict


def pack_jacobians(jacdict, inputs, outputs, T):
    """If we have T*T jacobians from nI inputs to nO outputs in jacdict, combine into (nO*T)*(nI*T) jacobian matrix."""
    nI, nO = len(inputs), len(outputs)

    outjac = np.empty((nO * T, nI * T))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            outjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = make_matrix(subdict.get(inputs[iI],
                                                                                               np.zeros((T, T))), T)
    return outjac


def unpack_jacobians(bigjac, inputs, outputs, T):
    """If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary"""
    nI, nO = len(inputs), len(outputs)

    jacdict = {}
    for iO in range(nO):
        jacdict[outputs[iO]] = {}
        for iI in range(nI):
            jacdict[outputs[iO]][inputs[iI]] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
    return jacdict


def pack_asymptotic_jacobians(jacdict, inputs, outputs, tau):
    nI, nO = len(inputs), len(outputs)
    A = np.empty((2*tau-1, nI, nO))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            if inputs[iI] in subdict:
                A[:, iO, iI] = make_ATI(jacdict[outputs[iO]][inputs[iI]], tau)
            else:
                A[:, iO, iI] = 0
    return A


def pack_vectors(vs, names, T):
    v = np.zeros(len(names)*T)
    for i, name in enumerate(names):
        if name in vs:
            v[i*T:(i+1)*T] = vs[name]
    return v


def unpack_vectors(v, names, T):
    vs = {}
    for i, name in enumerate(names):
        vs[name] = v[i*T:(i+1)*T]
    return vs


def make_matrix(A, T):
    """
    If A is not an outright ndarray, e.g. it is SimpleSparse, call its .matrix(T) method to convert it to T*T array.
    """
    if not isinstance(A, np.ndarray):
        return A.matrix(T)
    else:
        return A


def make_ATI(x, tau):
    if not isinstance(x, asymptotic.AsymptoticTimeInvariant):
        return x.asymptotic_time_invariant.changetau(tau).v
    else:
        return x.v


'''Part 3: IdentityMatrix class'''


class IdentityMatrix:
    """Simple identity matrix class with which we can initialize chain_jacobians, avoiding costly explicit construction
    of and operations on identity matrices."""
    __array_priority__ = 10_000

    def sparse(self):
        """Equivalent SimpleSparse representation, less efficient operations but more general."""
        return sim.SimpleSparse({(0, 0): 1})

    def matrix(self, T):
        return np.eye(T)

    def __matmul__(self, other):
        """Identity matrix knows to simply return 'other' whenever it's multiplied by 'other'."""
        return copy.deepcopy(other)

    def __rmatmul__(self, other):
        return copy.deepcopy(other)

    def __mul__(self, a):
        return a*self.sparse()

    def __rmul__(self, a):
        return self.sparse()*a

    def __add__(self, x):
        return self.sparse() + x

    def __radd__(self, x):
        return x + self.sparse()

    def __sub__(self, x):
        return self.sparse() - x

    def __rsub__(self, x):
        return x - self.sparse()

    def __neg__(self):
        return -self.sparse()

    def __pos__(self):
        return self

    def __repr__(self):
        return 'IdentityMatrix'