"""Methods for computing and manipulating both block-level and model-level Jacobians"""

import numpy as np
import copy
from numba import njit

from . import utilities as utils
from . import asymptotic
from .utilities.special_matrices import IdentityMatrix, ZeroMatrix

'''Part 1: High-level convenience routines: 
    - get_H_U               : get H_U matrix mapping all unknowns to all targets
    - get_impulse           : get single GE impulse response
    - get_G                 : get G matrices characterizing all GE impulse responses
    - get_G_asymptotic      : get asymptotic diagonals of the G matrices returned by get_G

    - curlyJs_sorted        : get block Jacobians curlyJ and return them topologically sorted
    - forward_accumulate    : forward accumulation on DAG, taking in topologically sorted Jacobians
'''


def get_H_U(block_list, unknowns, targets, T, ss=None, asymptotic=False, Tpost=None, save=False, use_saved=False):
    """Get T*n_u by T*n_u matrix H_U, Jacobian mapping all unknowns to all targets.

    Parameters
    ----------
    block_list : list, simple blocks, het blocks, or jacdicts
    unknowns   : list of str, names of unknowns in DAG
    targets    : list of str, names of targets in DAG
    T          : int, truncation horizon
                    (if asymptotic, truncation horizon for backward iteration in HetBlocks)
    ss         : [optional] dict, steady state required if block_list contains any non-jacdicts
    asymptotic : [optional] bool, flag for returning asymptotic H_U
    Tpost      : [optional] int, truncation horizon for asymptotic -(Tpost-1),...,0,...,(Tpost-1)
    save       : [optional] bool, flag for saving Jacobians inside HetBlocks
    use_saved  : [optional] bool, flag for using saved Jacobians inside HetBlocks

    Returns
    -------
    H_U : 
        if asymptotic=False:
            array(T*n_u*T*n_u) H_U, Jacobian mapping all unknowns to all targets
        is asymptotic=True:
            array((2*Tpost-1)*n_u*n_u), representation of asymptotic columns of H_U
    """

    # do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(block_list, unknowns, ss, T, asymptotic, Tpost, save, use_saved)

    # do matrix forward accumulation to get H_U = J^(curlyH, curlyU)
    H_U_unpacked = forward_accumulate(curlyJs, unknowns, targets, required)

    if not asymptotic:
        # pack these n_u^2 matrices, each T*T, into a single matrix
        return H_U_unpacked[targets, unknowns].pack(T)
        #return pack_jacobians(H_U_unpacked, unknowns, targets, T)
    else:
        # pack these n_u^2 AsymptoticTimeInvariant objects into a single (2*Tpost-1,n_u,n_u) array
        if Tpost is None:
            Tpost = 2*T
        return pack_asymptotic_jacobians(H_U_unpacked, unknowns, targets, Tpost)


def get_impulse(block_list, dZ, unknowns, targets, T=None, ss=None, outputs=None,
                H_U=None, H_U_factored=None, save=False, use_saved=False):
    """Get a single general equilibrium impulse response.

    Extremely fast when H_U_factored = utils.misc.factor(get_HU(...)) has already been computed
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
    H_U_factored : [optional] tuple of arrays, precomputed LU factorization utils.misc.factor(H_U)
    save         : [optional] bool, flag for saving Jacobians inside HetBlocks
    use_saved    : [optional] bool, flag for using saved Jacobians inside HetBlocks

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
        H_U = H_U_unpacked[targets, unknowns].pack(T)
        #H_U = pack_jacobians(H_U_unpacked, unknowns, targets, T)
    
    H_ZdZ_packed = pack_vectors(J_curlyZ_dZ, targets, T)

    if H_U_factored is None:
        dU_packed = - np.linalg.solve(H_U, H_ZdZ_packed)
    else:
        dU_packed = - utils.misc.factored_solve(H_U_factored, H_ZdZ_packed)

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

    Faster when H_U_factored = utils.misc.factor(get_HU(...)) has already been computed
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
    H_U_factored : [optional] tuple of arrays, precomputed LU factorization utils.misc.factor(H_U)
    save         : [optional] bool, flag for saving Jacobians inside HetBlocks
    use_saved    : [optional] bool, flag for using saved Jacobians inside HetBlocks

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
        H_U = J_curlyH_U[targets, unknowns].pack(T)
        #H_U = pack_jacobians(J_curlyH_U, unknowns, targets, T)
    H_Z = J_curlyH_Z[targets, exogenous].pack(T)
    #H_Z = pack_jacobians(J_curlyH_Z, exogenous, targets, T)

    if H_U_factored is None:
        G_U = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, exogenous, T)
    else:
        G_U = JacobianDict.unpack(-utils.misc.factored_solve(H_U_factored, H_Z), unknowns, exogenous, T)

    # step 4: forward accumulation to get all outputs starting with G_U
    # by default, don't calculate targets!
    curlyJs = [G_U] + curlyJs
    if outputs is None:
        outputs = set().union(*(curlyJ.outputs for curlyJ in curlyJs)) - set(targets)
    return forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))


def get_G_asymptotic(block_list, exogenous, unknowns, targets, T, ss=None, outputs=None, 
                     save=False, use_saved=False, Tpost=None):
    """Like get_G, but rather than returning the actual matrices G, return
    asymptotic.AsymptoticTimeInvariant objects representing their asymptotic columns."""

    # step 1: do topological sort and get curlyJs
    curlyJs, required = curlyJ_sorted(block_list, unknowns + exogenous, ss, T, save=save, 
                                      use_saved=use_saved, asymptotic=True, Tpost=Tpost)

    # step 2: do (matrix) forward accumulation to get
    # H_U = J^(curlyH, curlyU)
    J_curlyH_U = forward_accumulate(curlyJs, unknowns, targets, required)   

    # step 3: invert H_U and forward accumulate to get G_U = H_U^(-1)H_Z
    U_H_unpacked = asymptotic.invert_jacdict(J_curlyH_U, unknowns, targets, Tpost)
    G_U = forward_accumulate(curlyJs + [U_H_unpacked], exogenous, unknowns, required | set(targets))

    # step 4: forward accumulation to get all outputs starting with G_U
    # by default, don't calculate targets!
    curlyJs = [G_U] + curlyJs
    if outputs is None:
        outputs = set().union(*(curlyJ.outputs for curlyJ in curlyJs)) - set(targets)
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
    asymptotic : [optional] bool, flag for returning asymptotic Jacobians
    Tpost      : [optional] int, truncation horizon for asymptotic -(Tpost-1),...,0,...,(Tpost-1)
    save       : [optional] bool, flag for saving Jacobians inside HetBlocks
    use_saved  : [optional] bool, flag for using saved Jacobians inside HetBlocks

    Returns
    -------
    curlyJs : list of dict of dict, curlyJ for each block in order of topological sort
    required : list, outputs of some blocks that are needed as inputs by others
    """

    # step 1: get topological sort and required
    topsorted = utils.graph.block_sort(block_list, ignore_helpers=True)
    required = utils.graph.find_outputs_that_are_intermediate_inputs(block_list, ignore_helpers=True)

    # Remove any vector-valued outputs that are intermediate inputs, since we don't want
    # to compute Jacobians with respect to vector-valued variables
    if ss is not None:
        vv_vars = set([k for k, v in ss.items() if np.size(v) > 1])
        required -= vv_vars

    # step 2: compute Jacobians and put them in right order
    curlyJs = []
    shocks = set(inputs) | required
    for num in topsorted:
        block = block_list[num]
        if hasattr(block, 'ajac'):
            # has 'ajac' function, is some block other than SimpleBlock
            if asymptotic:
                jac = block.ajac(ss, T=T, shock_list=list(shocks), Tpost=Tpost, save=save, use_saved=use_saved)
            else:
                jac = block.jac(ss, T=T, shock_list=list(shocks), save=save, use_saved=use_saved)
        elif hasattr(block, 'jac'):
            # has 'jac' but not 'ajac', must be SimpleBlock where no distinction (given SimpleSparse)
            jac = block.jac(ss, shock_list=list(shocks))
        else:
            # doesn't have 'jac', must be nested dict that is jac directly
            jac = block

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
        #out = {i: {i: utils.special_matrices.IdentityMatrix()} for i in inputs}
        out = JacobianDict.identity(inputs)
    else:
        out = inputs.copy()

    # iterate through curlyJs, in what is presumed to be a topologically sorted order
    for curlyJ in curlyJs:
        curlyJ = JacobianDict(curlyJ).complete()
        if alloutputs is not None:
            # if we want specific list of outputs, restrict curlyJ to that before continuing
            #curlyJ = {k: v for k, v in curlyJ.items() if k in alloutputs}
            curlyJ = curlyJ[[k for k in alloutputs if k in curlyJ.outputs]]
        if jacflag:
            #out.update(compose_jacobians(out, curlyJ))
            out.update(curlyJ.compose(out))
        else:
            #out.update(apply_jacobians(curlyJ, out))
            out.update(curlyJ.apply(out))

    if outputs is not None:
        # if we want specific list of outputs, restrict to that
        # (dropping 'required' in 'alloutputs' that was needed for intermediate computations)
        #return {k: out[k] for k in outputs if k in out}
        return out[[k for k in outputs if k in out.outputs]]
    else:
        return out
        # if jacflag:
        #     # default behavior for Jacobian case: return all Jacobians we used/calculated along the way
        #     # except the (redundant) IdentityMatrix objects mapping inputs to themselves
        #     return {k: v for k, v in out.items() if k not in inputs}
        # else:
        #     # default behavior for case where we're calculating paths: return everything, including inputs
        #     return out


'''Part 2: Somewhat lower-level routines for handling Jacobians'''


def pack_asymptotic_jacobians(jacdict, inputs, outputs, tau):
    """If we have -(tau-1),...,(tau-1) AsymptoticTimeInvariant Jacobians (or SimpleSparse) from
    nI inputs to nO outputs in jacdict, combine into (2*tau-1,nO,nI) array A"""
    nI, nO = len(inputs), len(outputs)
    A = np.empty((2*tau-1, nI, nO))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            if inputs[iI] in subdict:
                A[:, iO, iI] = make_ATI_v(jacdict[outputs[iO]][inputs[iI]], tau)
            else:
                A[:, iO, iI] = 0
    return A


def unpack_asymptotic_jacobians(A, inputs, outputs, tau):
    """If we have (2*tau-1, nO, nI) array A where each A[:,o,i] is vector for AsymptoticTimeInvariant
    Jacobian mapping output o to output i, output nested dict of AsymptoticTimeInvariant objects"""
    nI, nO = len(inputs), len(outputs)

    jacdict = {}
    for iO in range(nO):
        jacdict[outputs[iO]] = {}
        for iI in range(nI):
            jacdict[outputs[iO]][inputs[iI]] = asymptotic.AsymptoticTimeInvariant(A[:, iO, iI])
    return jacdict


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
    """If A is not an outright ndarray, e.g. it is SimpleSparse, call its .matrix(T) method
    to convert it to T*T array."""
    if not isinstance(A, np.ndarray):
        return A.matrix(T)
    else:
        return A


def make_ATI_v(x, tau):
    """If x is either a AsymptoticTimeInvariant or something that can be converted to it, e.g.
    SimpleSparse, report the underlying length 2*tau-1 vector with entries -(tau-1),...,(tau-1)"""
    if not isinstance(x, asymptotic.AsymptoticTimeInvariant):
        return x.asymptotic_time_invariant.changetau(tau).v
    else:
        return x.v


"""Experimental new jacdict feature"""

class NestedDict:
    def __init__(self, nesteddict, outputs=None, inputs=None):
        if isinstance(nesteddict, NestedDict):
            self.nesteddict = nesteddict.nesteddict
            self.outputs = nesteddict.outputs
            self.inputs = nesteddict.inputs
        else:
            self.nesteddict = nesteddict
            if outputs is None:
                outputs = list(nesteddict.keys())
            if inputs is None:
                inputs = []
                for v in nesteddict.values():
                    inputs.extend(list(v))
                inputs = deduplicate(inputs)
            
            self.outputs = list(outputs)
            self.inputs = list(inputs)
    
    def __repr__(self):
        return f'<{type(self).__name__} outputs={self.outputs}, inputs={self.inputs}>'

    def __iter__(self):
        return iter(self.outputs)

    def __getitem__(self, x):
        if isinstance(x, str):
            # case 1: just a single output, give subdict
            return self.nesteddict[x]
        elif isinstance(x, tuple):
            # case 2: tuple, referring to output and input
            o, i = x
            o = self.outputs if o == slice(None, None, None) else o
            i = self.inputs if i == slice(None, None, None) else i
            if isinstance(o, str):
                if isinstance(i, str):
                    # case 2a: one output, one input, return single Jacobian
                    return self.nesteddict[o][i]
                else:
                    # case 2b: one output, multiple inputs, return dict
                    return {ii: self.nesteddict[o][ii] for ii in i}
            else:
                # case 2c: multiple outputs, one or more inputs, return JacobianDict with outputs o and inputs i
                i = (i,) if isinstance(i, str) else i
                return JacobianDict({oo: {ii: self.nesteddict[oo][ii] for ii in i} for oo in o}, o, i)
        elif isinstance(x, list) or isinstance(x, set):
            # case 3: assume that list or set refers just to outputs, get all of those
            return JacobianDict({oo: self.nesteddict[oo] for oo in x}, x, self.inputs)
        else:
            raise ValueError(f'Tried to get impermissible item {x}')

    def get(self, *args, **kwargs):
        # this is for compatibilty, not a huge fan
        return self.nesteddict.get(*args, **kwargs)

    
    def update(self, J):
        if set(self.inputs) != set(J.inputs):
            raise ValueError(f'Cannot merge JacobianDicts with non-overlapping inputs {set(self.inputs) ^ set(J.inputs)}')
        if not set(self.outputs).isdisjoint(J.outputs):
            raise ValueError(f'Cannot merge JacobianDicts with overlapping outputs {set(self.outputs) & set(J.outputs)}')
        self.outputs = self.outputs + J.outputs
        self.nesteddict = {**self.nesteddict, **J.nesteddict}

    def complete(self, filler):
        nesteddict = {}
        for o in self.outputs:
            nesteddict[o] = dict(self.nesteddict[o])
            for i in self.inputs:
                if i not in nesteddict[o]:
                    nesteddict[o][i] = filler
        return JacobianDict(nesteddict, self.outputs, self.inputs)


def deduplicate(mylist):
    """Remove duplicates while otherwise maintaining order"""
    return list(dict.fromkeys(mylist))


class JacobianDict(NestedDict):
    @staticmethod
    def identity(ks):
        return JacobianDict({k: {k: IdentityMatrix()} for k in ks}, ks, ks).complete()

    def complete(self):
        return super().complete(ZeroMatrix())

    def __matmul__(self, x):
        if isinstance(x, JacobianDict):
            return self.compose(x)
        else:
            return self.apply(x)

    def compose(self, J):
        o_list = self.outputs
        m_list = tuple(set(self.inputs) & set(J.outputs))
        i_list = J.inputs

        J_om = self.nesteddict
        J_mi = J.nesteddict
        J_oi = {}

        for o in o_list:
            J_oi[o] = {}
            for i in i_list:
                Jout = ZeroMatrix()
                for m in m_list:
                    J_om[o][m]
                    J_mi[m][i]
                    Jout += J_om[o][m] @ J_mi[m][i]
                J_oi[o][i] = Jout

        return JacobianDict(J_oi, o_list, i_list)

    def apply(self, x):
        # assume that all entries in x have some length T, and infer it
        T = len(next(iter(x.values())))
        
        inputs = x.keys() & set(self.inputs)
        J_oi = self.nesteddict
        y = {}

        for o in self.outputs:
            y[o] = np.zeros(T)
            for i in inputs:
                y[o] += J_oi[o][i] @ x[i]
        
        return y

    def pack(self, T):
        J = np.empty((len(self.outputs) * T, len(self.inputs) * T))
        for iO, O in enumerate(self.outputs):
            for iI, I in enumerate(self.inputs):
                J[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = make_matrix(self[O, I], T)
        return J

    @staticmethod
    def unpack(bigjac, outputs, inputs, T):
        """If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary"""
        jacdict = {}
        for iO, O in enumerate(outputs):
            jacdict[O] = {}
            for iI, I in enumerate(inputs):
                jacdict[O][I] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
        return JacobianDict(jacdict, outputs, inputs)

