import numpy as np
from scipy.stats import norm
from numba import njit, guvectorize
import scipy.linalg

'''Part 1: Efficient interpolation'''


# Numba's guvectorize decorator compiles functions and allows them to be broadcast by NumPy when dimensions differ.
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    yq : array (nq), empty to be filled with interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """
    Efficient linear interpolation exploiting monotonicity. xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi

def interpolate_coord_robust(x, xq, check_increasing=False):
    """Wrapper for interpolate_coord_robust_vector that works if xq not vector,
    but still require x to be a vector"""
    if x.ndim != 1:
        raise ValueError('Data input to interpolate_coord_robust must have exactly one dimension')

    if check_increasing and np.any(x[:-1] >= x[1:]):
        raise ValueError('Data input to interpolate_coord_robust must be strictly increasing')

    if xq.ndim == 1:
        return interpolate_coord_robust_vector(x, xq)
    else:
        i, pi = interpolate_coord_robust_vector(x, xq.ravel())
        return i.reshape(xq.shape), pi.reshape(xq.shape)

@njit
def interpolate_coord_robust_vector(x, xq):
    """
    Linear interpolation exploiting monotonicity ONLY in data x, not in query points xq.
    Simple binary search, LESS EFFICIENT but more robust.
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Not guvectorized, so x and xq must have a single dimension (if x has more, just use ravel/reshape),
    main application intended to be universally-valid interpolation of SS policy rules.

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), query points (in any order)

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """

    n = len(x)
    nq = len(xq)
    xqi = np.empty(nq, dtype=np.uint32)
    xqpi = np.empty(nq)

    for iq in range(nq):
        if xq[iq] < x[0]:
            ilow = 0
        elif xq[iq] > x[-2]:
            ilow = n-2
        else:
            # start binary search
            # should end with ilow and ihigh exactly 1 apart, bracketing variable
            ihigh = n-1
            ilow = 0
            while ihigh - ilow > 1:
                imid = (ihigh + ilow) // 2
                if xq[iq] > x[imid]:
                    ilow = imid
                else:
                    ihigh = imid
         
        xqi[iq] = ilow
        xqpi[iq] = (x[ilow+1] - xq[iq]) / (x[ilow+1] - x[ilow])

    return xqi, xqpi


'''Part 2: Operations on the discretized distribution'''

"""OLD STUFF HERE"""

@njit
def forward_step(D, Pi_T, a_pol_i, a_pol_pi):
    """Single forward step to update distribution using an arbitrary asset policy.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of Lam_{t-1}.

    Parameters
    ----------
    D        : array (S*A), beginning-of-period distribution over s_t, a_(t-1)
    Pi_T     : array (S*S), transpose Markov matrix that maps s_t to s_(t+1)
    a_pol_i  : int array (S*A), left gridpoint of asset policy
    a_pol_pi : array (S*A), weight on left gridpoint of asset policy

    Returns
    ----------
    Dnew : array (S*A), beginning-of-next-period dist s_(t+1), a_t
    """

    # first create Dnew from updating asset state
    Dnew = np.zeros_like(D)
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            d = D[s, i]
            Dnew[s, apol] += d * api
            Dnew[s, apol + 1] += d * (1 - api)

    # then use transpose Markov matrix to update income state
    Dnew = Pi_T @ Dnew

    return Dnew

@njit
def forward_step_transpose(D, Pi, a_pol_i, a_pol_pi):
    """Efficient implementation of D_t =  Lam_{t-1} @ D_{t-1}' using sparsity of Lam_{t-1}."""
    D = Pi @ D
    Dnew = np.empty_like(D)
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            Dnew[s, i] = api * D[s, apol] + (1 - api) * D[s, apol + 1]
    return Dnew


@njit
def forward_step_policy_shock(Dss, Pi_T, a_pol_i_ss, a_pol_pi_shock):
    """Update distribution of agents with policy function perturbed around ss."""
    Dnew = np.zeros_like(Dss)
    for s in range(Dss.shape[0]):
        for i in range(Dss.shape[1]):
            apol = a_pol_i_ss[s, i]
            dshock = a_pol_pi_shock[s, i] * Dss[s, i]
            Dnew[s, apol] += dshock
            Dnew[s, apol + 1] -= dshock
    Dnew = Pi_T @ Dnew
    return Dnew

"""BEGIN NEW STUFF HERE"""

@njit
def forward_step_1d(D, Pi_T, x_i, x_pi):
    """Single forward step to update distribution using exogenous Markov transition Pi and
    policy x_i and x_pi for one-dimensional endogenous state.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of the endogenous
    part of Lam_{t-1}'.

    Note that it takes Pi_T, the transpose of Pi, as input rather than transposing itself;
    this is so that when it is applied repeatedly, we can precalculate a transpose stored in
    correct order rather than a view.

    Parameters
    ----------
    D : array (S*X), beginning-of-period distribution over s_t, x_(t-1)
    Pi_T : array (S*S), transpose Markov matrix that maps s_t to s_(t+1)
    x_i : int array (S*X), left gridpoint of endogenous policy
    x_pi : array (S*X), weight on left gridpoint of endogenous policy

    Returns
    ----------
    Dnew : array (S*X), beginning-of-next-period dist s_(t+1), x_t
    """

    # first update using endogenous policy
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            d = D[iz, ix]
            Dnew[iz, i] += d * pi
            Dnew[iz, i+1] += d * (1 - pi)

    # then using exogenous transition matrix
    return Pi_T @ Dnew


@njit
def forward_step_2d(D, Pi_T, x_i, y_i, x_pi, y_pi):
    """Like forward_step_1d but with two-dimensional endogenous state, policies given by x and y"""

    # first update using endogenous policy
    nZ, nX, nY = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i[iz, ix, iy]
                iyp = y_i[iz, ix, iy]
                beta = x_pi[iz, ix, iy]
                alpha = y_pi[iz, ix, iy]

                Dnew[iz, ixp, iyp] += alpha * beta * D[iz, ix, iy]
                Dnew[iz, ixp+1, iyp] += alpha * (1 - beta) * D[iz, ix, iy]
                Dnew[iz, ixp, iyp+1] += (1 - alpha) * beta * D[iz, ix, iy]
                Dnew[iz, ixp+1, iyp+1] += (1 - alpha) * (1 - beta) * D[iz, ix, iy]

    # then using exogenous transition matrix 
    return (Pi_T @ Dnew.reshape(nZ, -1)).reshape(nZ, nX, nY)


@njit
def forward_step_transpose_1d(D, Pi, x_i, x_pi):
    """Transpose of forward_step_1d"""
    # first update using exogenous transition matrix
    D = Pi @ D

    # then update using (transpose) endogenous policy
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            Dnew[iz, ix] = pi * D[iz, i] + (1-pi) * D[iz, i+1]
    return Dnew


@njit
def forward_step_transpose_2d(D, Pi, x_i, y_i, x_pi, y_pi):
    """Transpose of forward_step_2d."""
    # first update using exogenous transition matrix
    nZ, nX, nY = D.shape
    D = (Pi @ D.reshape(nZ, -1)).reshape(nZ, nX, nY)

    # then update using (transpose) endogenous policy
    Dnew = np.empty_like(D)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i[iz, ix, iy]
                iyp = y_i[iz, ix, iy]
                alpha = x_pi[iz, ix, iy]
                beta = y_pi[iz, ix, iy]

                Dnew[iz, ix, iy] = (alpha * beta * D[iz, ixp, iyp] + alpha * (1-beta) * D[iz, ixp, iyp+1] + 
                                    (1-alpha) * beta * D[iz, ixp+1, iyp] +
                                    (1-alpha) * (1-beta) * D[iz, ixp+1, iyp+1])
    return Dnew


@njit
def forward_step_shock_1d(Dss, Pi_T, x_i_ss, x_pi_shock):
    """forward_step_1d linearized wrt x_pi"""
    # first find effect of shock to endogenous policy
    nZ, nX = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            i = x_i_ss[iz, ix]
            dshock = x_pi_shock[iz, ix] * Dss[iz, ix]
            Dshock[iz, i] += dshock
            Dshock[iz, i + 1] -= dshock

    # then apply exogenous transition matrix to update
    return Pi_T @ Dshock


@njit
def forward_step_shock_2d(Dss, Pi_T, x_i_ss, y_i_ss, x_pi_ss, y_pi_ss, x_pi_shock, y_pi_shock):
    """forward_step_2d linearized wrt x_pi and y_pi"""
    # first find effect of shock to endogenous policy
    nZ, nX, nY = Dss.shape
    Dshock = np.zeros_like(Dss)
    for iz in range(nZ):
        for ix in range(nX):
            for iy in range(nY):
                ixp = x_i_ss[iz, ix, iy]
                iyp = y_i_ss[iz, ix, iy]
                alpha = x_pi_ss[iz, ix, iy]
                beta = y_pi_ss[iz, ix, iy]

                dalpha = x_pi_shock[iz, ix, iy] * Dss[iz, ix, iy]
                dbeta = y_pi_shock[iz, ix, iy] * Dss[iz, ix, iy]

                Dshock[iz, ixp, iyp] += dalpha * beta + alpha * dbeta
                Dshock[iz, ixp+1, iyp] += dbeta * (1-alpha) - beta * dalpha
                Dshock[iz, ixp, iyp+1] += dalpha * (1-beta) - alpha * dbeta
                Dshock[iz, ixp+1, iyp+1] -= dalpha * (1-beta) + dbeta * (1-alpha)

    # then apply exogenous transition matrix to update
    return (Pi_T @ Dshock.reshape(nZ, -1)).reshape(nZ, nX, nY)

@njit
def fast_aggregate(X, Y):
    """If X has dims (T, ...) and Y has dims (T, ...), do dot product T-by-T to get length-T vector,
    avoids costly creation of intermediates with np.sum(X*Y, axis=(...)) pattern for aggregation in td"""
    T = X.shape[0]
    Xnew = X.reshape(T, -1)
    Ynew = Y.reshape(T, -1)
    Z = np.empty(T)
    for t in range(T):
        Z[t] = Xnew[t, :] @ Ynew[t, :]
    return Z

def make_tuple(x):
    return (x,) if not (isinstance(x, tuple) or isinstance(x, list)) else x

def dist_ss(a_pol, Pi, a_grid, D_seed=None, pi_seed=None, tol=1E-10, maxit=100_000):
    """Iterate to find steady-state distribution."""
    if D_seed is None:
        # compute separately stationary dist of s, to start there assume a uniform distribution on assets otherwise
        pi = stationary(Pi, pi_seed)
        D = np.tile(pi[:, np.newaxis], (1, a_grid.shape[0])) / a_grid.shape[0]
    else:
        D = D_seed

    # obtain interpolated-coordinate asset policy rule
    a_pol_i, a_pol_pi = interpolate_coord(a_grid, a_pol)

    # to make matrix multiplication more efficient, make separate copy of Pi transpose
    Pi_T = Pi.T.copy()

    # iterate until convergence by tol, or maxit
    for it in range(maxit):
        Dnew = forward_step(D, Pi_T, a_pol_i, a_pol_pi)

        # only check convergence every 10 2iterations for efficiency
        if it % 10 == 0 and within_tolerance(D, Dnew, tol):
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    return D


'''Part 3: grids and Markov chains'''


def agrid(amax, n, amin=0, pivot=0.25):
    """Create equidistant grid in logspace between amin-pivot and amax+pivot."""
    a_grid = np.geomspace(amin + pivot, amax + pivot, n) - pivot
    a_grid[0] = amin  # make sure *exactly* equal to amin
    return a_grid


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)


def markov_tauchen(rho, sigma, N=7, m=3):
    """Tauchen method discretizing AR(1) s_t = rho*s_(t-1) + eps_t.

    Parameters
    ----------
    rho   : scalar, persistence
    sigma : scalar, unconditional sd of s_t
    N     : int, number of states in discretized Markov process
    m     : scalar, discretized s goes from approx -m*sigma to m*sigma

    Returns
    ----------
    y  : array (N), states proportional to exp(s) s.t. E[y] = 1
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    """

    # make normalized grid, start with cross-sectional sd of 1
    s = np.linspace(-m, m, N)
    ds = s[1] - s[0]
    sd_innov = np.sqrt(1 - rho ** 2)

    # standard Tauchen method to generate Pi given N and m
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sd_innov) -
                    norm.cdf(s[j] - rho * s - ds / 2, scale=sd_innov))

    # invariant distribution and scaling
    pi = stationary(Pi)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi


def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi


'''Part 4: topological sort and related code'''


class SetStack:
    """Stack implemented with list but tests membership with set to be efficient in big cases"""

    def __init__(self):
        self.myset = set()
        self.mylist = []

    def add(self, x):
        self.myset.add(x)
        self.mylist.append(x)

    def pop(self):
        x = self.mylist.pop()
        self.myset.remove(x)
        return x

    def top(self):
        return self.mylist[-1]

    def index(self, x):
        return self.mylist.index(x)

    def __contains__(self, x):
        return x in self.myset

    def __len__(self):
        return len(self.mylist)

    def __getitem__(self, i):
        return self.mylist.__getitem__(i)

    def __repr__(self):
        return self.mylist.__repr__()


def complete_reverse_graph(gph):
    """Given directed graph represented as a dict from nodes to iterables of nodes, return representation of graph that
    is complete (i.e. has each vertex pointing to some iterable, even if empty), and a complete version of reversed too.
    Have returns be sets, for easy removal"""

    revgph = {n: set() for n in gph}
    for n, e in gph.items():
        for n2 in e:
            n2_edges = revgph.setdefault(n2, set())
            n2_edges.add(n)

    gph_missing_n = revgph.keys() - gph.keys()
    gph = {**{k: set(v) for k, v in gph.items()}, **{n: set() for n in gph_missing_n}}
    return gph, revgph


def find_cycle(dep, onlyset=None):
    """Return list giving cycle if there is one, otherwise None"""

    # supposed to look only within 'onlyset', so filter out everything else
    if onlyset is not None:
        dep = {k: (set(v) & set(onlyset)) for k, v in dep.items() if k in onlyset}

    tovisit = set(dep.keys())
    stack = SetStack()
    while tovisit or stack:
        if stack:
            # if stack has something, still need to proceed with DFS
            n = stack.top()
            if dep[n]:
                # if there are any dependencies left, let's look at them
                n2 = dep[n].pop()
                if n2 in stack:
                    # we have a cycle, since this is already in our stack
                    i2loc = stack.index(n2)
                    return stack[i2loc:] + [stack[i2loc]]
                else:
                    # no cycle, visit this node only if we haven't already visited it
                    if n2 in tovisit:
                        tovisit.remove(n2)
                        stack.add(n2)
            else:
                # if no dependencies left, then we're done with this node, so let's forget about it
                stack.pop(n)
        else:
            # nothing left on stack, let's start the DFS from something new
            n = tovisit.pop()
            stack.add(n)

    # if we never find a cycle, we're done
    return None


def topological_sort(dep, names=None):
    """Given directed graph pointing from each node to the nodes it depends on, topologically sort nodes"""

    # get complete set version of dep, and its reversal, and build initial stack of nodes with no dependencies
    dep, revdep = complete_reverse_graph(dep)
    nodeps = [n for n in dep if not dep[n]]
    topsorted = []

    # Kahn's algorithm: find something with no dependency, delete its edges and update
    while nodeps:
        n = nodeps.pop()
        topsorted.append(n)
        for n2 in revdep[n]:
            dep[n2].remove(n)
            if not dep[n2]:
                nodeps.append(n2)

    # should be done: topsorted should be topologically sorted with same # of elements as original graphs!
    if len(topsorted) != len(dep):
        cycle_ints = find_cycle(dep, dep.keys() - set(topsorted))
        assert cycle_ints is not None, 'topological sort failed but no cycle, THIS SHOULD NEVER EVER HAPPEN'
        cycle = [names[i] for i in cycle_ints] if names else cycle_ints
        raise Exception(f'Topological sort failed: cyclic dependency {" -> ".join(cycle)}')

    return topsorted


def block_sort(block_list, findrequired=False):
    """Given list of blocks (either blocks themselves or dicts of Jacobians), find a topological sort and also
    optionally return which outputs must be computed as inputs of later blocks.
    
    Relies on blocks having 'inputs' and 'outputs' attributes (unless they are dicts of Jacobians, in which case it's
    inferred) that indicate their aggregate inputs and outputs"""
    # step 1: map outputs to blocks for topological sort
    outmap = dict()
    for num, block in enumerate(block_list):
        if hasattr(block, 'outputs'):
            outputs = block.outputs
        elif isinstance(block, dict):
            outputs = block.keys()
        else:
            raise ValueError(f'{block} is not recognized as block or does not provide outputs')

        for o in outputs:
            if o in outmap:
                raise ValueError(f'{o} is output twice')
            outmap[o] = num

    # step 2: dependency graph for topological sort and input list
    dep = {num: set() for num in range(len(block_list))}
    if findrequired:
        required = set()
    for num, block in enumerate(block_list):
        if hasattr(block, 'inputs'):
            inputs = block.inputs
        else:
            inputs = set(i for o in block for i in block[o])

        for i in inputs:
            if i in outmap:
                dep[num].add(outmap[i])
                if findrequired:
                    required.add(i)

    # step 3: return topological sort, also 'required' if wanted
    if findrequired:
        return topological_sort(dep), required
    else:
        return topological_sort(dep)


'''Part 5: nonlinear solvers'''


def obtain_J(f, x, y, h=1E-5):
    """finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((nx, ny))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx) - y) / h
    return J


def broyden_update(J, dx, dy):
    """Returns Broyden update to approximate Jacobian J given that last
    change in inputs to function was dx and led to output change of dy"""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)


def printit(it, x, y, **kwargs):
    """Convenience printing function for noisy iterations"""
    print(f'On iteration {it}')
    print(('x = %.3f' + ',%.3f' * (len(x) - 1)) % tuple(x))
    print(('y = %.3f' + ',%.3f' * (len(y) - 1)) % tuple(y))
    for kw, val in kwargs.items():
        print(f'{kw} = {val:.3f}')
    print('\n')


def newton_solver(f, x, y=None, tol=1E-9, maxcount=100, backtrack_c=0.5, noisy=True):
    """Simple line search solver in Newton direction, backtracks if input
    invalid or if improvement is not at least half the predicted improvement"""
    if y is None:
        y = f(x)

    for count in range(maxcount):
        if noisy:
            printit(count, x, y)

        if np.max(np.abs(y)) < tol:
            return x, y

        J = obtain_J(f, x, y)
        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                predicted_improvement = -np.sum((J @ dx) * y) * ((1 - 1 / 2 ** bcount) + 1) / 2
                actual_improvement = (np.sum(y ** 2) - np.sum(ynew ** 2)) / 2
                if actual_improvement < predicted_improvement / 2:
                    if noisy:
                        print('backtracking\n')
                    dx *= backtrack_c
                else:
                    y = ynew
                    x += dx
                    break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')


def broyden_solver(f, x, y=None, tol=1E-9, maxcount=100, backtrack_c=0.5, noisy=True):
    """Simple line search solver in approximate Newton direction, obtaining approximate J from Broyden updating."""
    if y is None:
        y = f(x)

    # initialize J with Newton!
    J = obtain_J(f, x, y)
    for count in range(maxcount):
        if noisy:
            printit(count, x, y)

        if np.max(np.abs(y)) < tol:
            return x, y

        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            # note: can't test for improvement with Broyden because maybe
            # the function doesn't improve locally in this direction, since
            # J isn't the exact Jacobian
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                J = broyden_update(J, dx, ynew - y)
                y = ynew
                x += dx
                break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')


'''Part 6: Other utilities'''


@njit
def setmin(x, xmin):
    """Set 2-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                x[i, j] = xmin
            else:
                break


@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


def numerical_diff(func, ssinputs_dict, shock_dict, h=1E-4, y_ss_list=None):
    """Differentiate function via forward difference."""
    # compute ss output if not supplied
    if y_ss_list is None:
        y_ss_list = func(**ssinputs_dict)

    # response to small shock
    shocked_inputs = {**ssinputs_dict, **{k: ssinputs_dict[k] + h * shock for k, shock in shock_dict.items()}}
    y_list = func(**shocked_inputs)

    # scale responses back up
    dy_list = [(y - y_ss) / h for y, y_ss in zip(y_list, y_ss_list)]

    return dy_list


def factor(X):
    return scipy.linalg.lu_factor(X)

def factored_solve(Z, y):
    return scipy.linalg.lu_solve(Z, y)