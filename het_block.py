import numpy as np
import utils
import copy
import asymptotic


def het(exogenous, policy, backward):
    def decorator(back_step_fun):
        return HetBlock(back_step_fun, exogenous, policy, backward)
    return decorator


class HetBlock:
    """Part 1: Initializer for HetBlock, intended to be called via @het() decorator on back it function."""

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
        variable and then the remaining dimensions are each of the continuous policy variables, in the
        same order they are listed in 'policy'.

        The Markov transition matrix between the current and future period and backward iteration
        variables should appear in the backward iteration function with '_p' subscripts ("prime") to
        indicate that they come from the next period.

        Currently, we only support up to two policy variables.
        """

        self.back_step_fun = back_step_fun

        self.all_outputs_order = utils.output_list(back_step_fun)
        all_outputs = set(self.all_outputs_order)
        self.all_inputs = set(utils.input_list(back_step_fun))

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

        # aggregate outputs and inputs for utils.block_sort
        self.inputs = self.all_inputs - {k + '_p' for k in self.backward}
        self.inputs.remove(exogenous + '_p')
        self.inputs.add(exogenous)
        
        self.outputs = {k.upper() for k in self.non_back_outputs}

        # start without a hetinput
        self.hetinput = None
        self.hetinput_inputs = set()
        self.hetinput_outputs_order = tuple()

        # 'saved' arguments start empty
        self.saved = {}
        self.prelim_saved = {}
        self.saved_shock_list = []
        self.saved_output_list = []

        # note: should do more input checking to ensure certain choices not made: 'D' not input, etc.

    def __repr__(self):
        """Nice string representation of HetBlock for printing to console"""
        if self.hetinput is not None:
            return f"<HetBlock '{self.back_step_fun.__name__}' with hetinput '{self.hetinput.__name__}'>"
        else:
            return f"<HetBlock '{self.back_step_fun.__name__}'>"

    '''Part 2: high-level routines, with first three called analogously to SimpleBlock counterparts
        - ss    : do backward and forward iteration until convergence to get complete steady state
        - td    : do backward and forward iteration up to T to compute dynamics given some shocks
        - jac   : compute jacobians of outputs with respect to shocked inputs, using fake news algorithm
        - ajac  : compute asymptotic columns of jacobians output by jac, also using fake news algorithm

        - attach_hetinput : make new HetBlock that first processes inputs through function hetinput
    '''

    def ss(self, backward_tol=1E-8, backward_maxit=5000, forward_tol=1E-10, forward_maxit=100_000, **kwargs):
        """Evaluate steady state HetBlock using keyword args for all inputs. Analog to SimpleBlock.ss.

        Parameters
        ----------
        backward_tol : [optional] float
            in backward iteration, max abs diff between policy in consecutive steps needed for convergence
        backward_maxit : [optional] int
            maximum number of backward iterations, if 'backward_tol' not reached by then, raise error
        forward_tol : [optional] float
            in forward iteration, max abs diff between dist in consecutive steps needed for convergence
        forward_maxit : [optional] int
            maximum number of forward iterations, if 'forward_tol' not reached by then, raise error

        kwargs : dict
            The following inputs are required as keyword arguments, which show up in 'kwargs':
                - The exogenous Markov matrix, e.g. Pi=... if self.exogenous=='Pi'
                - A seed for each backward variable, e.g. Va=... and Vb=... if self.backward==('Va','Vb')
                - A grid for each policy variable, e.g. a_grid=... and b_grid=... if self.policy==('a','b')
                - All other inputs to the backward iteration function self.back_step_fun, except _p added to
                    for self.exogenous and self.backward, for which the method uses steady-state values.
                    If there is a self.hetinput, then we need the inputs to that, not to self.back_step_fun.

            Other inputs in 'kwargs' are optional:
                - A seed for the distribution: D=...
                - If no seed for the distribution is provided, a seed for the invariant distribution
                    of the Markov process, e.g. Pi_seed=... if self.exogenous=='Pi'

        Returns
        ----------
        ss : dict, contains
            - ss inputs of self.back_step_fun and (if present) self.hetinput
            - ss outputs of self.back_step_fun
            - ss distribution 'D'
            - ss aggregates (in uppercase) for all outputs of self.back_step_fun except self.backward
        """

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

        # clear any previously saved Jacobian info for safety, since we're computing new SS
        self.clear_saved()

        return {**sspol, **aggs, 'D': D}

    def td(self, ss, monotonic=False, returnindividual=False, **kwargs):
        """Evaluate transitional dynamics for HetBlock given dynamic paths for inputs in kwargs,
        assuming that we start and end in steady state ss, and that all inputs not specified in
        kwargs are constant at their ss values. Analog to SimpleBlock.td.

        CANNOT provide time-varying paths of grid or Markov transition matrix for now.
        
        Parameters
        ----------
        ss : dict
            all steady-state info, intended to be from .ss()
        monotonic : [optional] bool
            flag indicating date-t policies are monotonic in same date-(t-1) policies, allows us
            to use faster interpolation routines, otherwise use slower robust to nonmonotonicity
        returnindividual : [optional] bool
            return distribution and full outputs on grid
        kwargs : dict of {str : array(T, ...)}
            all time-varying inputs here, with first dimension being time
            this must have same length T for all entries (all outputs will be calculated up to T)

        Returns
        ----------
        td : dict
            if returnindividual = False, time paths for aggregates (uppercase) for all outputs
                of self.back_step_fun except self.backward
            if returnindividual = True, additionally time paths for distribution and for all outputs
                of self.back_Step_fun on the full grid
        """

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
                    # TODO: change for two-asset case so assumption is monotonicity in own asset, not anything else
                    sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord(grid[pol], individual_paths[pol][t, ...])
                else:
                    sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], individual_paths[pol][t, ...])

            # step forward
            D_path[t+1, ...]= self.forward_step(D_path[t, ...], Pi_T, sspol_i, sspol_pi)

        # obtain aggregates of all outputs, made uppercase
        aggregates = {k.upper(): utils.fast_aggregate(D_path, individual_paths[k]) for k in self.non_back_outputs}

        # return either this, or also include distributional information
        if returnindividual:
            return {**aggregates, **individual_paths, 'D': D_path}
        else:
            return aggregates

    def jac(self, ss, T, shock_list, output_list=None, h=1E-4, save=False, use_saved=False):
        """Assemble nested dict of Jacobians of agg outputs vs. inputs, using fake news algorithm.

        Parameters
        ----------
        ss : dict,
            all steady-state info, intended to be from .ss()
        T : [optional] int
            number of time periods for T*T Jacobian
        shock_list : list of str
            names of input variables to differentiate wrt (main cost scales with # of inputs)
        output_list : list of str
            names of output variables to get derivatives of, if not provided assume all outputs of
            self.back_step_fun except self.backward
        h : [optional] float
            h for numerical differentiation of backward iteration
        save : [optional] bool
            store curlyYs, curlyDs, curlyPs, F, and J from calculation inside HetBlock itself
            useful to avoid redundant work when evaluating .jac or .ajac again
        use_saved : [optional] bool
            use J stored inside HetBlock to calculate the Jacobian, raises error if not available

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives T*T Jacobian of o with respect to i
        """
        # default outputs are just all outputs of back it function except backward variables
        if output_list is None:
            output_list = self.non_back_outputs

        # if we're supposed to use saved Jacobian, extract T-by-T submatrices for each (o,i)
        if use_saved:
            return utils.extract_nested_dict(savedA=self.saved['J'],
                        keys1=[o.upper() for o in output_list], keys2=shock_list, shape=(T, T))

        # step 0: preliminary processing of steady state
        (ssin_dict, Pi, ssout_list, ss_for_hetinput, 
                                    sspol_i, sspol_pi, sspol_space) = self.jac_prelim(ss, save)

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = {}, {}
        for i in shock_list:
            curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(i, output_list, ssin_dict, ssout_list,
                                                ss['D'], Pi.T.copy(), sspol_i, sspol_pi, sspol_space, T, h,
                                                ss_for_hetinput)

        # step 2 of fake news algorithm
        # compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = {}
        for o in output_list:
            curlyPs[o] = self.forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T-1)

        # steps 3-4 of fake news algorithm
        # make fake news matrix and Jacobian for each outcome-input pair
        F = {o.upper(): {} for o in output_list}
        J = {o.upper(): {} for o in output_list}
        for o in output_list:
            for i in shock_list:
                F[o.upper()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.upper()][i] = HetBlock.J_from_F(F[o.upper()][i])

        if save:
            self.saved_shock_list, self.saved_output_list = shock_list, output_list
            self.saved = {'curlyYs' : curlyYs, 'curlyDs' : curlyDs, 'curlyPs' : curlyPs, 'F': F, 'J': J}

        return J

    def ajac(self, ss, T, shock_list, output_list=None, h=1E-4, Tpost=None, save=False, use_saved=False):
        """Like .jac, but outputs asymptotic columns of Jacobians as AsymptoticTimeInvariant objects
        with nonzero entries -(T-1),...,(Tpost-1) representing asymptotic entries in diagonals,
        measured relative to main diagonal.
        
        Does additional iteration on curlyPs as necessary to extend Tpost beyond T, since common case
        is that curlyYs and curlyDs from backward iteration converge to zero much more quickly than
        curlyPs from forward iteration."""

        # default outputs are just all outputs of back it function except backward variables
        if output_list is None:
            output_list = self.non_back_outputs

        # if Tpost not provided, assume it is 2*T by default
        if Tpost is None:
            Tpost = 2*T
        elif Tpost < T:
            raise ValueError(f'must have Tpost={Tpost} less than T={T}')

        # saved last by ajac, directly extract
        if use_saved and 'curlyYs' not in self.saved:
            asympJ = {}
            for o in output_list:
                asympJ[o.upper()] = {}
                for i in shock_list:
                    asympJ[o.upper()][i] = asymptotic.AsymptoticTimeInvariant(
                                                self.saved['asympJ'][o.upper()][i][-(Tpost-1): Tpost])
            return asympJ

        # was either saved last by jac or not saved at all, need to do more work!

        # step 0: preliminary processing of steady state
        (ssin_dict, Pi, ssout_list, ss_for_hetinput, 
                        sspol_i, sspol_pi, sspol_space) = self.jac_prelim(ss, save, use_saved)

        if use_saved and 'curlyYs' in self.saved:
            # was saved by jac, first copy curlyYs, curlyDs, curlyPs
            curlyYs = utils.extract_nested_dict(savedA=self.saved['curlyYs'],
                                                keys1=shock_list, keys2=output_list, shape=(T,))
            curlyDs = utils.extract_dict(savedA=self.saved['curlyDs'], keys=shock_list, shape=(T,))
            curlyPs_old = utils.extract_dict(savedA=self.saved['curlyPs'], keys=output_list, shape=(T-1,))

            # now need curlyPs that go to T+Tpost-1, not just T
            curlyPs = {}
            for o in output_list:
                curlyP_extrarows = self.forward_iteration_fakenews(curlyPs_old[o][-1, ...], 
                                                                   Pi, sspol_i, sspol_pi, Tpost)
                curlyPs[o] = np.concatenate((curlyPs_old[o][:-1, ...], curlyP_extrarows), axis=0)
        else:
            # was not saved at all, get curlyYs, curlyDs, curlyPs for ourselves
            # step 1: compute curlyY and curlyD (backward iteration) for each input i (same as jac)
            curlyYs, curlyDs = {}, {}
            for i in shock_list:
                curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(i, output_list, 
                                                    ssin_dict, ssout_list, ss['D'], Pi.T.copy(), sspol_i, 
                                                    sspol_pi, sspol_space, T, h, ss_for_hetinput)

            # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
            # here go to (T-1) + (Tpost-1) rather than (T-1)
            curlyPs = {}
            for o in output_list:
                curlyPs[o] = self.forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T-1+Tpost-1)

        # steps 3-4: make fake news matrix and Jacobian for each outcome-input pair
        J = {o.upper(): {} for o in output_list}
        asympJ = {o.upper(): {} for o in output_list}
        for o in output_list:
            for i in shock_list:
                F = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.upper()][i] = HetBlock.J_from_F(F)
                asympJ[o.upper()][i] = asymptotic.AsymptoticTimeInvariant(
                    np.concatenate((np.zeros(Tpost-T), J[o.upper()][i][:, -1])))

        # if supposed to save, record J and asympJ for use by jac or ajac
        if save:
            self.saved_shock_list, self.saved_output_list = shock_list, output_list
            self.saved = {'J' : J, 'asympJ' : asympJ}

        return asympJ

    def attach_hetinput(self, hetinput):
        """Make new HetBlock that first processes inputs through function hetinput.
        Assumes 'self' currently does not have hetinput."""
        if self.hetinput is not None:
            raise ValueError('Trying to attach hetinput when it is already there!')

        newself = copy.deepcopy(self)
        newself.hetinput = hetinput
        newself.hetinput_inputs = set(utils.input_list(hetinput))
        newself.hetinput_outputs_order = utils.output_list(hetinput)

        # modify inputs to include hetinput's additional inputs, remove outputs
        newself.inputs |= newself.hetinput_inputs
        newself.inputs -= set(newself.hetinput_outputs_order)
        return newself

    '''Part 3: components of ss():
        - policy_ss : backward iteration to get steady-state policies and other outcomes
        - dist_ss   : forward iteration to get steady-state distribution and compute aggregates
    '''

    def policy_ss(self, ssin, tol=1E-8, maxit=5000):
        """Find steady-state policies and backward variables through backward iteration until convergence.

        Parameters
        ----------
        ssin : dict
            all steady-state inputs to back_step_fun, including seed values for backward variables
        tol : [optional] float
            max diff between consecutive iterations of policy variables needed for convergence
        maxit : [optional] int
            maximum number of iterations, if 'tol' not reached by then, raise error

        Returns
        ----------
        sspol : dict
            all steady-state outputs of backward iteration, combined with inputs to backward iteration
        """

        original_ssin = ssin
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
        
        # want to record inputs in ssin, but remove _p, add in hetinput inputs if there
        for k in self.inputs_p:
            ssin[k] = ssin[k + '_p']
            del ssin[k + '_p']
        if self.hetinput is not None:
            for k in self.hetinput_inputs:
                if k in original_ssin:
                    ssin[k] = original_ssin[k]
        return {**ssin, **sspol}

    def dist_ss(self, Pi, sspol, grid, tol=1E-10, maxit=100_000, D_seed=None, pi_seed=None):
        """Find steady-state distribution through forward iteration until convergence.

        Parameters
        ----------
        Pi : array
            steady-state Markov matrix for exogenous variable
        sspol : dict
            steady-state policies on grid for all policy variables in self.policy
        grid : dict
            grids for all policy variables in self.policy
        tol : [optional] float
            absolute tolerance for max diff between consecutive iterations for distribution
        maxit : [optional] int
            maximum number of iterations, if 'tol' not reached by then, raise error
        D_seed : [optional] array
            initial seed for overall distribution
        pi_seed : [optional] array
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

    '''Part 4: components of jac(), corresponding to *4 steps of fake news algorithm* in paper
        - Step 1: backward_step_fakenews and backward_iteration_fakenews to get curlyYs and curlyDs
        - Step 2: forward_iteration_fakenews to get curlyPs
        - Step 3: build_F to get fake news matrix from curlyYs, curlyDs, curlyPs
        - Step 4: J_from_F to get Jacobian from fake news matrix
    '''

    def backward_step_fakenews(self, din_dict, output_list, ssin_dict, ssout_list, 
                               Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h=1E-4):
        # shock perturbs outputs
        shocked_outputs = {k: v for k, v in zip(self.all_outputs_order,
                                                utils.numerical_diff(self.back_step_fun, ssin_dict, din_dict, h,
                                                                     ssout_list))}
        curlyV = {k: shocked_outputs[k] for k in self.backward}

        # which affects the distribution tomorrow
        pol_pi_shock = {k: -shocked_outputs[k]/sspol_space[k] for k in self.policy}
        curlyD = self.forward_step_shock(Dss, Pi_T, sspol_i, sspol_pi, pol_pi_shock)

        # and the aggregate outcomes today
        curlyY = {k: np.vdot(Dss, shocked_outputs[k]) for k in output_list}

        return curlyV, curlyD, curlyY

    def backward_iteration_fakenews(self, input_shocked, output_list, ssin_dict, ssout_list, Dss, Pi_T, 
                                    sspol_i, sspol_pi, sspol_space, T, h=1E-4, ss_for_hetinput=None):
        """Iterate policy steps backward T times for a single shock."""
        if self.hetinput is not None and input_shocked in self.hetinput_inputs:
            # if input_shocked is an input to hetinput, take numerical diff to get response
            din_dict = dict(zip(self.hetinput_outputs_order, 
                                utils.numerical_diff_symmetric(self.hetinput, ss_for_hetinput, {input_shocked: 1}, h)))
        else:
            # otherwise, we just have that one shock
            din_dict = {input_shocked: 1}

        # contemporaneous response to unit scalar shock
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, ssin_dict, ssout_list, 
                                                             Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h)
        
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
                                                    output_list, ssin_dict, ssout_list, 
                                                    Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h)
            for k in curlyY.keys():
                curlyYs[k][t] = curlyY[k]

        return curlyYs, curlyDs

    def forward_iteration_fakenews(self, o_ss, Pi, pol_i_ss, pol_pi_ss, T):
        """Iterate transpose forward T steps to get full set of curlyPs for a given outcome.
        
        Note we depart from definition in paper by applying the demeaning operator in addition to Lambda
        at each step. This does not affect products with curlyD (which are the only way curlyPs enter
        Jacobian) since perturbations to distribution always have mean zero. It has numerical benefits
        since curlyPs now go to zero for high t (used in paper in proof of Proposition 1).
        """
        curlyPs = np.empty((T,) + o_ss.shape)
        curlyPs[0, ...] = utils.demean(o_ss)
        for t in range(1, T):
            curlyPs[t, ...] = utils.demean(self.forward_step_transpose(curlyPs[t-1, ...], Pi, pol_i_ss, pol_pi_ss))
        return curlyPs

    @staticmethod
    def build_F(curlyYs, curlyDs, curlyPs):
        T = curlyDs.shape[0]
        Tpost = curlyPs.shape[0] - T + 2
        F = np.empty((Tpost + T - 1, T))
        F[0, :] = curlyYs
        F[1:, :] = curlyPs.reshape((Tpost + T - 2, -1)) @ curlyDs.reshape((T, -1)).T
        return F

    @staticmethod
    def J_from_F(F):
        J = F.copy()
        for t in range(1, J.shape[1]):
            J[1:, t] += J[:-1, t - 1]
        return J

    '''Part 5: helpers for .jac and .ajac: preliminary processing and clearing saved info'''

    def jac_prelim(self, ss, save=False, use_saved=False):
        """Helper that does preliminary processing of steady state for fake news algorithm.

        Parameters
        ----------
        ss : dict, all steady-state info, intended to be from .ss()
        save : [optional] bool, whether to store results in .prelim_saved attribute
        use_saved : [optional] bool, whether to use already-stored results in .prelim_saved
        
        Returns
        ----------
        ssin_dict       : dict, ss vals of exactly the inputs needed by self.back_step_fun for backward step
        Pi              : array (S*S), Markov matrix for exogenous state
        ssout_list      : tuple, what self.back_step_fun returns when given ssin_dict (not exactly the same
                            as steady-state numerically since SS convergence was to some tolerance threshold)
        ss_for_hetinput : dict, ss vals of exactly the inputs needed by self.hetinput (if it exists)
        sspol_i         : dict, indices on lower bracketing gridpoint for all in self.policy
        sspol_pi        : dict, weights on lower bracketing gridpoint for all in self.policy
        sspol_space     : dict, space between lower and upper bracketing gridpoints for all in self.policy
        """
        output_names = ('ssin_dict', 'Pi', 'ssout_list',
                            'ss_for_hetinput', 'sspol_i', 'sspol_pi', 'sspol_space')

        if use_saved:
            if self.prelim_saved:
                return tuple(self.prelim_saved[k] for k in output_names)
            else:
                raise ValueError('Nothing saved to be used by jac_prelim!')
        
        # preliminary a: obtain ss inputs and other info, run once to get baseline for numerical differentiation
        ssin_dict = self.make_inputs(ss)
        Pi = ss[self.exogenous]
        grid = {k: ss[k+'_grid'] for k in self.policy}
        ssout_list = self.back_step_fun(**ssin_dict)

        ss_for_hetinput = None
        if self.hetinput is not None:
            ss_for_hetinput = {k: ss[k] for k in self.hetinput_inputs if k in ss}

        # preliminary b: get sparse representations of policy rules, and distance between neighboring policy gridpoints
        sspol_i = {}
        sspol_pi = {}
        sspol_space = {}
        for pol in self.policy:
            # use robust binary-search-based method that only requires grids to be monotonic
            sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], ss[pol])
            sspol_space[pol] = grid[pol][sspol_i[pol]+1] - grid[pol][sspol_i[pol]]

        toreturn = (ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space)
        if save:
            self.prelim_saved = {k: v for (k, v) in zip(output_names, toreturn)}
        return toreturn

    def clear_saved(self):
        """Erase any saved Jacobian information from .jac or .ajac (e.g. if steady state changes)"""
        self.saved = {}
        self.prelim_saved = {}
        self.saved_shock_list = []
        self.saved_output_list = []

    '''Part 6: helper to extract inputs and potentially process them through hetinput'''

    def make_inputs(self, indict):
        """Extract from indict exactly the inputs needed for self.back_step_fun,
        process stuff through hetinput first if it's there"""
        if self.hetinput is not None:
            outputs_as_tuple = utils.make_tuple(self.hetinput(**{k: indict[k] for k in self.hetinput_inputs if k in indict}))
            indict.update(dict(zip(self.hetinput_outputs_order, outputs_as_tuple)))

        indict_new = {k: indict[k] for k in self.all_inputs - self.inputs_p if k in indict}
        try:
            return {**indict_new, **{k + '_p': indict[k] for k in self.inputs_p}}
        except KeyError as e:
            print(f'Missing backward variable or Markov matrix {e} for {self.back_step_fun.__name__}!')
            raise

    '''Part 7: routines to do forward steps of different kinds, all wrap functions in utils'''

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
