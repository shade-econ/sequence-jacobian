import numpy as np
import copy
from numba import njit

from . import utils
from . import asymptotic
from .blocks import simple_block as sim


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
        return pack_jacobians(H_U_unpacked, unknowns, targets, T)
    else:
        # pack these n_u^2 AsymptoticTimeInvariant objects into a single (2*Tpost-1,n_u,n_u) array
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
    topsorted = utils.block_sort(block_list, ignore_helpers=True)
    required = utils.find_outputs_that_are_intermediate_inputs(block_list, ignore_helpers=True)

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
            curlyJs.append(jac)

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
        out = {i: {i: IdentityMatrix()} for i in inputs}
    else:
        out = inputs.copy()

    # iterate through curlyJs, in what is presumed to be a topologically sorted order
    for curlyJ in curlyJs:
        if alloutputs is not None:
            # if we want specific list of outputs, restrict curlyJ to that before continuing
            curlyJ = {k: v for k, v in curlyJ.items() if k in alloutputs}
        if jacflag:
            out.update(compose_jacobians(out, curlyJ))
        else:
            out.update(apply_jacobians(curlyJ, out))

    if outputs is not None:
        # if we want specific list of outputs, restrict to that
        # (dropping 'required' in 'alloutputs' that was needed for intermediate computations)
        return {k: out[k] for k in outputs if k in out}
    else:
        if jacflag:
            # default behavior for Jacobian case: return all Jacobians we used/calculated along the way
            # except the (redundant) IdentityMatrix objects mapping inputs to themselves
            return {k: v for k, v in out.items() if k not in inputs}
        else:
            # default behavior for case where we're calculating paths: return everything, including inputs
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


'''Part 3: SimpleSparse and IdentityMatrix classes and related helpers'''
class SimpleSparse:
    """Efficient representation of sparse linear operators, which are linear combinations of basis
    operators represented by pairs (i, m), where i is the index of diagonal on which there are 1s
    (measured by # above main diagonal) and m is number of initial entries missing.

    Examples of such basis operators:
        - (0, 0) is identity operator
        - (0, 2) is identity operator with first two '1's on main diagonal missing
        - (1, 0) has 1s on diagonal above main diagonal: "left-shift" operator
        - (-1, 1) has 1s on diagonal below main diagonal, except first column

    The linear combination of these basis operators that makes up a given SimpleSparse object is
    stored as a dict 'elements' mapping (i, m) -> x.

    The Jacobian of a SimpleBlock is a SimpleSparse operator combining basis elements (i, 0). We need
    the more general basis (i, m) to ensure closure under multiplication.

    These (i, m) correspond to the Q_(-i, m) operators defined for Proposition 2 of the Sequence Space
    Jacobian paper. The flipped sign in the code is so that the index 'i' matches the k(i) notation
    for writing SimpleBlock functions.

    The "dunder" methods x.__add__(y), x.__matmul__(y), x.__rsub__(y), etc. in Python implement infix
    operations x + y, x @ y, y - x, etc. Defining these allows us to use these more-or-less
    interchangeably with ordinary NumPy matrices.
    """

    # when performing binary operations on SimpleSparse and a NumPy array, use SimpleSparse's rules
    __array_priority__ = 1000

    def __init__(self, elements):
        self.elements = elements
        self.indices, self.xs = None, None

    @staticmethod
    def from_simple_diagonals(elements):
        """Take dict i -> x, i.e. from SimpleBlock differentiation, convert to SimpleSparse (i, 0) -> x"""
        return SimpleSparse({(i, 0): x for i, x in elements.items()})

    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""
        return self + np.zeros((T, T))

    def array(self):
        """Rewrite dict (i, m) -> x as pair of NumPy arrays, one size-N*2 array of ints with rows (i, m)
        and one size-N array of floats with entries x.

        This is needed for Numba to take as input. Cache for efficiency.
        """
        if self.indices is not None:
            return self.indices, self.xs
        else:
            indices, xs = zip(*self.elements.items())
            self.indices, self.xs = np.array(indices), np.array(xs)
            return self.indices, self.xs

    @property
    def asymptotic_time_invariant(self):
        indices, xs = self.array()
        tau = np.max(np.abs(indices[:, 0]))+1 # how far out do we go?
        v = np.zeros(2*tau-1)
        #v[indices[:, 0]+tau-1] = xs
        v[-indices[:, 0]+tau-1] = xs # switch from asymptotic ROW to asymptotic COLUMN
        return asymptotic.AsymptoticTimeInvariant(v)

    @property
    def T(self):
        """Transpose"""
        return SimpleSparse({(-i, m): x for (i, m), x in self.elements.items()})

    @property
    def iszero(self):
        return not self.nonzero().elements

    def nonzero(self):
        elements = self.elements.copy()
        for im, x in self.elements.items():
            # safeguard to retain sparsity: disregard extremely small elements (num error)
            if abs(elements[im]) < 1E-14:
                del elements[im]
        return SimpleSparse(elements)

    def __pos__(self):
        return self

    def __neg__(self):
        return SimpleSparse({im: -x for im, x in self.elements.items()})

    def __matmul__(self, A):
        if isinstance(A, SimpleSparse):
            # multiply SimpleSparse by SimpleSparse, simple analytical rules in multiply_rs_rs
            return multiply_rs_rs(self, A)
        elif isinstance(A, np.ndarray):
            # multiply SimpleSparse by matrix or vector, multiply_rs_matrix uses slicing
            indices, xs = self.array()
            if A.ndim == 2:
                return multiply_rs_matrix(indices, xs, A)
            elif A.ndim == 1:
                return multiply_rs_matrix(indices, xs, A[:, np.newaxis])[:, 0]
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __rmatmul__(self, A):
        # multiplication rule when this object is on right (will only be called when left is matrix)
        # for simplicity, just use transpose to reduce this to previous cases
        return (self.T @ A.T).T

    def __add__(self, A):
        if isinstance(A, SimpleSparse):
            # add SimpleSparse to SimpleSparse, combining dicts, summing x when (i, m) overlap
            elements = self.elements.copy()
            for im, x in A.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x
            return SimpleSparse(elements)
        else:
            # add SimpleSparse to T*T matrix
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
                return NotImplemented
            T = A.shape[0]

            # fancy trick to do this efficiently by writing A as flat vector
            # then (i, m) can be mapped directly to NumPy slicing!
            A = A.flatten()     # use flatten, not ravel, since we'll modify A and want a copy
            for (i, m), x in self.elements.items():
                if i < 0:
                    A[T * (-i) + (T + 1) * m::T + 1] += x
                else:
                    A[i + (T + 1) * m:(T - i) * T:T + 1] += x
            return A.reshape((T, T))

    def __radd__(self, A):
        try:
            return self + A
        except:
            print(self)
            print(A)
            raise

    def __sub__(self, A):
        # slightly inefficient implementation with temporary for simplicity
        return self + (-A)

    def __rsub__(self, A):
        return -self + A

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return SimpleSparse({im: a * x for im, x in self.elements.items()})

    def __rmul__(self, a):
        return self * a

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x:.3f}' for (i, m), x in self.elements.items()) + '}'
        return f'SimpleSparse({formatted})'

    def __eq__(self, s):
        return self.elements == s.elements


def multiply_basis(t1, t2):
    """Matrix multiplication operation mapping two sparse basis elements to another."""
    # equivalent to formula in Proposition 2 of Sequence Space Jacobian paper, but with
    # signs of i and j flipped to reflect different sign convention used here
    i, m = t1
    j, n = t2
    k = i + j
    if i >= 0:
        if j >= 0:
            l = max(m, n - i)
        elif k >= 0:
            l = max(m, n - k)
        else:
            l = max(m + k, n)
    else:
        if j <= 0:
            l = max(m + j, n)
        else:
            l = max(m, n) + min(-i, j)
    return k, l


def multiply_rs_rs(s1, s2):
    """Matrix multiplication operation on two SimpleSparse objects."""
    # iterate over all pairs (i, m) -> x and (j, n) -> y in objects,
    # add all pairwise products to get overall product
    elements = {}
    for im, x in s1.elements.items():
        for jn, y in s2.elements.items():
            kl = multiply_basis(im, jn)
            if kl in elements:
                elements[kl] += x * y
            else:
                elements[kl] = x * y
    return SimpleSparse(elements)


@njit
def multiply_rs_matrix(indices, xs, A):
    """Matrix multiplication of SimpleSparse object ('indices' and 'xs') and matrix A.
    Much more computationally demanding than multiplying two SimpleSparse (which is almost
    free with simple analytical formula), so we implement as jitted function."""
    n = indices.shape[0]
    T = A.shape[0]
    S = A.shape[1]
    Aout = np.zeros((T, S))

    for count in range(n):
        # for Numba to jit easily, SimpleSparse with basis elements '(i, m)' with coefs 'x'
        # was stored in 'indices' and 'xs'
        i = indices[count, 0]
        m = indices[count, 1]
        x = xs[count]

        # loop faster than vectorized when jitted
        # directly use def of basis element (i, m), displacement of i and ignore first m
        if i == 0:
            for t in range(m, T):
                for s in range(S):
                    Aout[t, s] += x * A[t, s]
        elif i > 0:
            for t in range(m, T - i):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
        else:
            for t in range(m - i, T):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
    return Aout


class IdentityMatrix:
    """Simple identity matrix class with which we can initialize chain_jacobians and forward_accumulate,
    avoiding costly explicit construction of and operations on identity matrices."""
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