import numpy as np
import inspect
import re
import utils


class HetBlock:
    def __init__(self, back_step_fun, exogenous, policy, backward):
        """Construct HetBlock from backward iteration function.

        Parameters
        ----------
        back_step_fun : function
            backward iteration function
        exogenous : str
            names of Markov transition matrix for exogenous variable
            (now only single allowed for simplicity; use Kronecker product for more)
        policy : str or sequence of str
            names of policy variables of endogenous, continuous state variables
            e.g. assets 'a', must be returned by function
        backward : str or sequence of str
            variables that together comprise the 'v' that we use for iterating backward
            must appear both as outputs and as arguments

        It is assumed that every output of the function (except possibly backward), including policy,
        will be on a grid of dimension 1 + len(policy), where the first dimension is the exogenous
        variable and then the remaining dimensions are each of the continuous policy variables.

        Currently, we only support up to two policy variables.
        """

        self.back_step_fun = back_step_fun

        self.all_outputs_order = re.findall('return (.*?)\n',
                                            inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')
        all_outputs = set(self.all_outputs_order)
        self.all_inputs = set(inspect.getfullargspec(back_step_fun).args)

        self.exogenous = exogenous
        self.policy, self.backward = (utils.make_tuple(x) for x in (policy, backward))

        if len(self.policy) > 2:
            raise ValueError(f"More than two endogenous policies in {back_step_fun.__name__}, not yet supported")

        self.inputs_p = {self.exogenous} | set(self.backward)

        # input checking
        if self.exogenous + '_p' not in self.all_inputs:
            raise ValueError(f"Markov matrix '{self.exogenous}_p' not included as argument in {back_step_fun.__name__}")
        
        for pol in self.policy:
            if pol not in all_outputs:
                raise ValueError(f"Policy '{pol}' not included as output in {back_step_fun.__name__}")

        for back in self.backward:
            if back + '_p' not in self.all_inputs:
                raise ValueError(f"Backward variable '{back}_p' not included as argument in {back_step_fun.__name__}")

            if back not in all_outputs:
                raise ValueError(f"Backward variable '{back}' not included as output in {back_step_fun.__name__}")

        self.non_back_outputs = all_outputs - set(self.backward)
        for out in self.non_back_outputs:
            if out.isupper():
                raise ValueError("Output '{out}' is uppercase in {back_step_fun.__name__}, not allowed")

        self.non_back_inputs = self.all_inputs - set(self.backward)

        # note: should do more input checking to ensure certain choices not made: 'D' not input, etc.

    def make_inputs(self, indict):
        """Extract from ssin exactly the inputs needed for self.back_step_fun"""
        indict_new = {k: indict[k] for k in self.all_inputs - self.inputs_p if k in indict}
        try:
            return {**indict_new, **{k + '_p': indict[k] for k in self.inputs_p}}
        except KeyError as e:
            print(f'Missing backward variable or Markov matrix {e} for {self.back_step_fun.__name__}!')
            raise

    def policy_ss(self, ssin, tol=1E-8, maxit=5000):
        """Find steady-state policies and v through backward iteration until convergence.

        Parameters
        ----------
        ssin : dict
            all steady-state inputs to back_step_fun, including seed values for backward variables
        tol : float, optional
            absolute tolerance for max diff between consecutive iterations for policy variables
        maxit : int, optional
            maximum number of iterations, if 'tol' not reached by then, raise error

        Returns
        ----------
        sspol : dict
            all steady-state outputs of backward iteration
        """

        ssin = self.make_inputs(ssin)
        
        old = {}
        for it in range(maxit):
            try:
                # run and store results of backward iteration, which come as tuple, in dict
                sspol = {k: v for k, v in zip(self.all_outputs_order, self.back_step_fun(**ssin))}
            except KeyError as e:
                print(f'Missing input {e} to {self.back_step_fun.__name__}!')
                raise

            # only check convergence every 10 iterations for efficiency
            if it % 10 == 1 and all(utils.within_tolerance(sspol[k], old[k], tol) for k in self.policy):
                break

            # update 'old' for comparison during next iteration, prepare 'ssin' as input for next iteration
            old.update({k: sspol[k] for k in self.policy})
            ssin.update({k + '_p': sspol[k] for k in self.backward})
        else:
            raise ValueError(f'No convergence of policy functions after {maxit} backward iterations!')
        
        return sspol

    def dist_ss(self, Pi, sspol, grid, tol=1E-10, maxit=5000, D_seed=None, pi_seed=None):
        """Find steady-state distribution through forward iteration until convergence.

        Parameters
        ----------
        Pi : array
            steady-state Markov matrix for exogenous variable
        sspol : dict
            steady-state policies on grid for all policy variables in self.policy
        grid : dict
            grids for all policy variables in self.policy
        tol : float, optional
            absolute tolerance for max diff between consecutive iterations for distribution
        maxit : int, optional
            maximum number of iterations, if 'tol' not reached by then, raise error
        D_seed : array, optional
            initial seed for overall distribution
        pi_seed : array, optional
            initial seed for stationary dist of Pi, if no D_seed

        Returns
        ----------
        D : array
            steady-state distribution
        """

        # first obtain initial distribution D
        if D_seed is None:
            # compute stationary distribution for exogenous variable
            pi = utils.stationary(Pi, pi_seed)
            
            # now initialize full distribution with this, assuming uniform distribution on endogenous vars
            endogenous_dims = [grid[k].shape[0] for k in self.policy]
            D = np.tile(pi, endogenous_dims[::-1] + [1]).T / np.prod(endogenous_dims)
        else:
            D = D_seed

        # obtain interpolated policy rule for each dimension of endogenous policy
        sspol_i = {}
        sspol_pi = {}
        for pol in self.policy:
            # use robust binary search-based method that only requires grids, not policies, to be monotonic
            sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], sspol[pol])

        # iterate until convergence by tol, or maxit
        Pi_T = Pi.T.copy()
        for it in range(maxit):
            Dnew = self.forward_step(D, Pi_T, sspol_i, sspol_pi)

            # only check convergence every 10 iterations for efficiency
            if it % 10 == 0 and utils.within_tolerance(D, Dnew, tol):
                break
            D = Dnew
        else:
            raise ValueError(f'No convergence after {maxit} forward iterations!')
        
        return D

    def ss(self, backward_tol=1E-8, backward_maxit=5000, forward_tol=1E-10, forward_maxit=5000, **kwargs):
        """Evaluate steady state hetblock using keyword arguments for all inputs, analogous to
        simple block ss method.
        
        Need all inputs to policy_ss, also '{pol}_grid' argument for each policy variable pol.
        
        If 'D' included, will be treated as seed. Seed for dist for exog Markov Pi is Pi_seed"""

        # extract information from kwargs
        Pi = kwargs[self.exogenous]
        grid = {k: kwargs[k+'_grid'] for k in self.policy}
        D_seed = kwargs.get('D', None)
        pi_seed = kwargs.get(self.exogenous + '_seed', None)

        # run backward iteration
        sspol = self.policy_ss(kwargs, backward_tol, backward_maxit)

        # run forward iteration
        D = self.dist_ss(Pi, sspol, grid, forward_tol, forward_maxit, D_seed, pi_seed)

        # aggregate all outputs other than backward variables on grid, capitalize
        aggs = {k.upper(): np.vdot(D, sspol[k]) for k in self.non_back_outputs}

        # report all other inputs as well
        report = {k: kwargs[k] for k in self.all_inputs - self.inputs_p if k in kwargs}

        return {**sspol, **aggs, **report, 'Pi': Pi, 'D': D}

    def td(self, ss, monotonic=False, returnindividual=False, **kwargs):
        """Evaluate dynamics for hetblock given dynamic paths for inputs in kwargs,
        starting and ending in steady state ss (assume everything not given in kwargs
        is constant at ss value).

        If monotonic=True, use faster interpolation method for policies against grid,
        otherwise use slower interpolation.

        Do not return full individual outputs and distribution unless returnindividual=True.

        Cannot alter Markov transition matrix or grid for now."""

        # infer T from kwargs, check that all shocks have same length
        shock_lengths = [x.shape[0] for x in kwargs.values()]
        if shock_lengths[1:] != shock_lengths[:-1]:
            raise ValueError('Not all shocks in kwargs are same length!')
        T = shock_lengths[0]

        # copy from ss info
        Pi_T = ss[self.exogenous].T.copy()
        grid = {k: ss[k+'_grid'] for k in self.policy}
        D = ss['D']

        # allocate empty arrays to store result, assume all like D
        individual_paths = {k: np.empty((T,) + D.shape) for k in self.non_back_outputs}

        # backward iteration
        backdict = ss.copy()
        for t in reversed(range(T)):
            # be careful: if you include vars from self.backward variables in kwargs, agents will use them!
            backdict.update({k: v[t,...] for k, v in kwargs.items()})
            individual = {k: v for k, v in zip(self.all_outputs_order,
                                    self.back_step_fun(**self.make_inputs(backdict)))}
            backdict.update({k: individual[k] for k in self.backward})
            for k in self.non_back_outputs:
                individual_paths[k][t, ...] = individual[k]

        D_path = np.empty((T,) + D.shape)
        D_path[0, ...] = D
        for t in range(T-1):
            # have to interpolate policy separately for each t to get sparse transition matrices
            sspol_i = {}
            sspol_pi = {}
            for pol in self.policy:
                if monotonic:
                    sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord(grid[pol], individual_paths[pol][t, ...])
                else:
                    sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], individual_paths[pol][t, ...])

            # step forward
            D_path[t+1, ...]= self.forward_step(D_path[t, ...], Pi_T, sspol_i, sspol_pi)

        # obtain aggregates of all outputs, made uppercase
        aggregates = {k.upper(): utils.fast_aggregate(D_path, individual_paths[k]) for k in self.non_back_outputs}

        # return either this, or also include distributional information
        if returnindividual:
            return {**aggregates, **individual, 'D': D_path}
        else:
            return aggregates

    def forward_step(self, D, Pi_T, pol_i, pol_pi):
        """Update distribution, calling on 1d and 2d-specific compiled routines.

        Parameters
        ----------
        D : array, beginning-of-period distribution
        Pi_T : array, transpose Markov matrix
        pol_i : dict, indices on lower bracketing gridpoint for all in self.policy
        pol_pi : dict, weights on lower bracketing gridpoint for all in self.policy

        Returns
        ----------
        Dnew : array, beginning-of-next-period distribution
        """
        if len(self.policy) == 1:
            p, = self.policy
            return utils.forward_step_1d(D, Pi_T, pol_i[p], pol_pi[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step_2d(D, Pi_T, pol_i[p1], pol_i[p2], pol_pi[p1], pol_pi[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

    def forward_step_transpose(self, D, Pi, pol_i, pol_pi):
        """Transpose of forward_step (note: this takes Pi rather than Pi_T as argument!)"""
        if len(self.policy) == 1:
            p, = self.policy
            return utils.forward_step_transpose_1d(D, Pi, pol_i[p], pol_pi[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step_transpose_2d(D, Pi, pol_i[p1], pol_i[p2], pol_pi[p1], pol_pi[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

    def forward_step_shock(self, Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_pi_shock):
        """Forward_step linearized with respect to pol_pi"""
        if len(self.policy) == 1:
            p, = self.policy
            return utils.forward_step_shock_1d(Dss, Pi_T, pol_i_ss[p], pol_pi_shock[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step_shock_2d(Dss, Pi_T, pol_i_ss[p1], pol_i_ss[p2],
                                               pol_pi_ss[p1], pol_pi_ss[p2], pol_pi_shock[p1], pol_pi_shock[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

    def backward_step_fakenews(self, din_dict, desired_outputs, ssin_dict, ssout_list, 
                               Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_space_ss, h=1E-4):
        # shock perturbs outputs
        shocked_outputs = {k: v for k, v in zip(self.all_outputs_order,
                                                utils.numerical_diff(self.back_step_fun, ssin_dict, din_dict, h,
                                                                     ssout_list))}
        curlyV = {k: shocked_outputs[k] for k in self.backward}

        # which affects the distribution tomorrow
        pol_pi_shock = {k: -shocked_outputs[k]/pol_space_ss[k] for k in self.policy}
        curlyD = self.forward_step_shock(Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_pi_shock)

        # and the aggregate outcomes today
        curlyY = {k: np.vdot(Dss, shocked_outputs[k]) for k in desired_outputs}

        return curlyV, curlyD, curlyY

    def backward_iteration_fakenews(self, din_dict, desired_outputs, ssin_dict, ssout_list,
                                    Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_space_ss, T, h=1E-4):
        """Iterate policy steps backward T times for a single shock."""

        # contemporaneous response to unit scalar shock
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, desired_outputs, ssin_dict, ssout_list, 
                                                             Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_space_ss, h)
        
        # infer dimensions from this and initialize empty arrays
        curlyDs = np.empty((T,) + curlyD.shape)
        curlyYs = {k: np.empty(T) for k in curlyY.keys()}

        # fill in current effect of shock
        curlyDs[0, ...] = curlyD
        for k in curlyY.keys():
            curlyYs[k][0] = curlyY[k]

        # fill in anticipation effects
        for t in range(1, T):
            curlyV, curlyDs[t, ...], curlyY = self.backward_step_fakenews({k+'_p': v for k, v in curlyV.items()},
                                                    desired_outputs, ssin_dict, ssout_list, 
                                                    Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_space_ss, h)
            for k in curlyY.keys():
                curlyYs[k][t] = curlyY[k]

        return curlyYs, curlyDs

    def forward_iteration_fakenews(self, o_ss, Pi, pol_i_ss, pol_pi_ss, T):
        """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome."""
        curlyPs = np.empty((T,) + o_ss.shape)
        curlyPs[0, ...] = o_ss
        for t in range(1, T):
            curlyPs[t, ...] = self.forward_step_transpose(curlyPs[t-1, ...], Pi, pol_i_ss, pol_pi_ss)
        return curlyPs

    def all_Js(self, ss, T, shock_dict, desired_outputs, h=1E-4):
        # preliminary a: obtain ss inputs and other info, run once to get baseline for numerical differentiation
        ssin_dict = self.make_inputs(ss)
        Pi = ss[self.exogenous]
        Pi_T = Pi.T.copy()
        grid = {k: ss[k+'_grid'] for k in self.policy}
        ssout_list = self.back_step_fun(**ssin_dict)

        # preliminary b: get sparse representations of policy rules, and distance between neighboring policy gridpoints
        sspol_i = {}
        sspol_pi = {}
        sspol_space = {}
        for pol in self.policy:
            # use robust binary-search-based method that only requires grids to be monotonic
            sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], ss[pol])
            sspol_space[pol] = grid[pol][sspol_i[pol]+1] - grid[pol][sspol_i[pol]]

        # step 1: compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = dict(), dict()
        for i, shock in shock_dict.items():
            curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(shock, desired_outputs, ssin_dict, ssout_list,
                                                                ss['D'], Pi_T, sspol_i, sspol_pi, sspol_space, T, h)

        # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = dict()
        for o in desired_outputs:
            curlyPs[o] = self.forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T)

        # steps 3-4: make fake news matrix and Jacobian for each outcome-input pair
        J = {o.upper(): {} for o in desired_outputs}
        for o in desired_outputs:
            for i in shock_dict:
                F = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.upper()][i] = J_from_F(F)

        # report Jacobians
        return J


def het(exogenous, policy, backward):
    def decorator(back_step_fun):
        return HetBlock(back_step_fun, exogenous, policy, backward)
    return decorator


"""OLD JACOBIAN CODE BEGINS HERE"""

'''Part 0: Set up the stage for general HA jacobian code.'''


def extract_info(back_step_fun, ss):
    """Process source code of a backward iteration function.

    Parameters
    ----------
    back_step_fun : function
        backward iteration function
    ss : dict
        steady state dictionary
    Returns
    ----------
    ssinput_dict : dict
      {name: ss value} for all inputs to back_step_fun
    ssy_list : list
      steady state value of outputs of back_step_fun in same order
    outcome_list : list
      names of variables returned by back_step_fun other than V
    V_name : str
      name of backward variable
    """
    V_name, *outcome_list = re.findall('return (.*?)\n',
                                       inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')

    # ssy_list = [ss[k] for k in [V_name] + outcome_list]

    input_names = inspect.getfullargspec(back_step_fun).args
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    # want numerical differentiation to subtract against this for greater accuracy,
    # since steady state is not *exactly* a fixed point of back_step_fun
    ssy_list = back_step_fun(**ssinput_dict)

    return ssinput_dict, ssy_list, outcome_list, V_name


'''
Part 1: get curlyYs and curlyDs

That is, dynamic effects that propagate through changes in policy holding distribution constant.    
'''


def backward_step(dinput_dict, back_step_fun, ssinput_dict, ssy_list, outcome_list, D, Pi, a_pol_i, a_space, h=1E-4):
    # shock perturbs policies
    curlyV, da, *dy_list = utils.numerical_diff(back_step_fun, ssinput_dict, dinput_dict, h, ssy_list)

    # which affects the distribution tomorrow
    da_pol_pi = -da / a_space
    curlyD = utils.forward_step_policy_shock(D, Pi.T, a_pol_i, da_pol_pi)

    # and the aggregate outcomes today
    curlyY = {name: np.vdot(D, dy) for name, dy in zip(outcome_list, [da] + dy_list)}

    return curlyV, curlyD, curlyY


def backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, V_name, D, Pi, a_pol_i, a_space, T):
    """Iterate policy steps backward T times for a single shock."""
    # contemporaneous response to unit scalar shock
    curlyV, curlyD, curlyY = backward_step(shock, back_step_fun, ssinput_dict,
                                           ssy_list, outcome_list, D, Pi, a_pol_i, a_space)

    # infer dimensions from this and initialize empty arrays
    curlyDs = np.empty((T,) + curlyD.shape)
    curlyYs = {k: np.empty(T) for k in curlyY.keys()}

    # fill in current effect of shock
    curlyDs[0, ...] = curlyD
    for k in curlyY.keys():
        curlyYs[k][0] = curlyY[k]

    # fill in anticipation effects
    for t in range(1, T):
        curlyV, curlyDs[t, ...], curlyY = backward_step({V_name + '_p': curlyV}, back_step_fun, ssinput_dict,
                                                        ssy_list, outcome_list, D, Pi, a_pol_i, a_space)
        for k in curlyY.keys():
            curlyYs[k][t] = curlyY[k]

    return curlyYs, curlyDs


'''
Part 2: get curlyPs, aka prediction vectors

That is dynamic effects that propagate through the distribution's law of motion holding policy constant.    
'''


def forward_iteration_transpose(y_ss, Pi, a_pol_i, a_pol_pi, T):
    """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome."""
    curlyPs = np.empty((T,) + y_ss.shape)
    curlyPs[0, ...] = y_ss
    for t in range(1, T):
        curlyPs[t, ...] = utils.forward_step_transpose(curlyPs[t-1, ...], Pi, a_pol_i, a_pol_pi)
    return curlyPs


'''
Part 3: construct fake news matrix, and then Jacobian.
'''


def build_F(curlyYs, curlyDs, curlyPs):
    T = curlyDs.shape[0]
    F = np.empty((T, T))
    F[0, :] = curlyYs
    F[1:, :] = curlyPs[:T - 1, ...].reshape((T - 1, -1)) @ curlyDs.reshape((T, -1)).T
    return F


def J_from_F(F):
    J = F.copy()
    for t in range(1, J.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


'''Part 4: Putting it all together'''


def all_Js(back_step_fun, ss, T, shock_dict):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule, then distance between neighboring policy gridpoints
    a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], ss['a'])
    a_space = ss['a_grid'][a_pol_i + 1] - ss['a_grid'][a_pol_i]

    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                    V_name, ss['D'], ss['Pi'], a_pol_i, a_space, T)

    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[1:]):
        curlyPs[o] = forward_iteration_transpose(ssy, ss['Pi'], a_pol_i, a_pol_pi, T)

    # step 3: make fake news matrix and Jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
            J[o][i] = J_from_F(F)

    # remap outcomes to capital letters to avoid conflicts
    for k in list(J.keys()):
        K = k.upper()
        J[K] = J.pop(k)

    # report Jacobians
    return J
