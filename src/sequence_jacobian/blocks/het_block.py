import copy
import numpy as np

from .support.impulse import ImpulseDict
from .support.bijection import Bijection
from ..primitives import Block
from .. import utilities as utils
from ..steady_state.classes import SteadyStateDict
from ..jacobian.classes import JacobianDict
from .support.bijection import Bijection
from ..utilities.function import DifferentiableExtendedFunction, ExtendedFunction


def het(exogenous, policy, backward, backward_init=None):
    def decorator(back_step_fun):
        return HetBlock(back_step_fun, exogenous, policy, backward, backward_init=backward_init)
    return decorator


class HetBlock(Block):
    """Part 1: Initializer for HetBlock, intended to be called via @het() decorator on backward step function.

    IMPORTANT: All `policy` and non-aggregate output variables of this HetBlock need to be *lower-case*, since
    the methods that compute steady state, transitional dynamics, and Jacobians for HetBlocks automatically handle
    aggregation of non-aggregate outputs across the distribution and return aggregates as upper-case equivalents
    of the `policy` and non-aggregate output variables specified in the backward step function.
    """

    def __init__(self, back_step_fun, exogenous, policy, backward, backward_init=None):
        """Construct HetBlock from backward iteration function.

        Parameters
        ----------
        back_step_fun : function
            backward iteration function
        exogenous : str
            name of Markov transition matrix for exogenous variable
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
        self.name = back_step_fun.__name__
        super().__init__()

        # self.back_step_fun is one iteration of the backward step function pertaining to a given HetBlock.
        # i.e. the function pertaining to equation (14) in the paper: v_t = curlyV(v_{t+1}, X_t)
        self.back_step_fun = ExtendedFunction(back_step_fun)

        # See the docstring of HetBlock for details on the attributes directly below
        self.exogenous = exogenous
        self.policy, self.back_iter_vars = (utils.misc.make_tuple(x) for x in (policy, backward))

        # self.inputs_to_be_primed indicates all variables that enter into self.back_step_fun whose name has "_p"
        # (read as prime). Because it's the case that the initial dict of input arguments for self.back_step_fun
        # contains the names of these variables that omit the "_p", we need to swap the key from the unprimed to
        # the primed key name, such that self.back_step_fun will properly call those variables.
        # e.g. the key "Va" will become "Va_p", associated to the same value.
        self.inputs_to_be_primed = {self.exogenous} | set(self.back_iter_vars)

        # self.non_back_iter_outputs are all of the outputs from self.back_step_fun excluding the backward
        # iteration variables themselves.
        self.non_back_iter_outputs = self.back_step_fun.outputs - set(self.back_iter_vars)

        # self.outputs and self.inputs are the *aggregate* outputs and inputs of this HetBlock, which are used
        # in utils.graph.block_sort to topologically sort blocks along the DAG
        # according to their aggregate outputs and inputs.
        # TODO: go back from capitalize to upper!!! (ask Michael first)
        self.outputs = {o.capitalize() for o in self.non_back_iter_outputs}
        self.M_outputs = Bijection({o: o.capitalize() for o in self.non_back_iter_outputs})
        self.inputs = self.back_step_fun.inputs - {k + '_p' for k in self.back_iter_vars}
        self.inputs.remove(exogenous + '_p')
        self.inputs.add(exogenous)

        # A HetBlock can have heterogeneous inputs and heterogeneous outputs, henceforth `hetinput` and `hetoutput`.
        # See docstring for methods `add_hetinput` and `add_hetoutput` for more details.
        self.hetinput = None
        self.hetoutput = None

        # The set of variables that will be wrapped in a separate namespace for this HetBlock
        # as opposed to being available at the top level
        self.internal = utils.misc.smart_set(self.back_step_fun.outputs) | utils.misc.smart_set(self.exogenous) | {"D"}

        if len(self.policy) > 2:
            raise ValueError(f"More than two endogenous policies in {back_step_fun.__name__}, not yet supported")

        # Checking that the various inputs/outputs attributes are correctly set
        if self.exogenous + '_p' not in self.back_step_fun.inputs:
            raise ValueError(f"Markov matrix '{self.exogenous}_p' not included as argument in {back_step_fun.__name__}")

        for pol in self.policy:
            if pol not in self.back_step_fun.outputs:
                raise ValueError(f"Policy '{pol}' not included as output in {back_step_fun.__name__}")
            if pol[0].isupper():
                raise ValueError(f"Policy '{pol}' is uppercase in {back_step_fun.__name__}, which is not allowed")

        for back in self.back_iter_vars:
            if back + '_p' not in self.back_step_fun.inputs:
                raise ValueError(f"Backward variable '{back}_p' not included as argument in {back_step_fun.__name__}")

            if back not in self.back_step_fun.outputs:
                raise ValueError(f"Backward variable '{back}' not included as output in {back_step_fun.__name__}")

        for out in self.non_back_iter_outputs:
            if out[0].isupper():
                raise ValueError("Output '{out}' is uppercase in {back_step_fun.__name__}, which is not allowed")

        if backward_init is not None:
            backward_init = ExtendedFunction(backward_init)
        self.backward_init = backward_init

        # note: should do more input checking to ensure certain choices not made: 'D' not input, etc.

    def __repr__(self):
        """Nice string representation of HetBlock for printing to console"""
        if self.hetinput is not None:
            if self.hetoutput is not None:
                return f"<HetBlock '{self.name}' with hetinput '{self.hetinput.name}'" \
                       f" and with hetoutput `{self.hetoutput.name}'>"
            else:
                return f"<HetBlock '{self.name}' with hetinput '{self.hetinput.name}'>"
        else:
            return f"<HetBlock '{self.name}'>"

    '''Part 2: high-level routines, with first three called analogously to SimpleBlock counterparts
        - steady_state      : do backward and forward iteration until convergence to get complete steady state
        - impulse_nonlinear : do backward and forward iteration up to T to compute dynamics given some shocks
        - impulse_linear    : apply jacobians to compute linearized dynamics given some shocks
        - jacobian          : compute jacobians of outputs with respect to shocked inputs, using fake news algorithm

        - add_hetinput : add a hetinput to the HetBlock that first processes inputs through function hetinput
        - add_hetoutput: add a hetoutput to the HetBlock that is computed after the entire ss computation, or after
                         each backward iteration step in td, jacobian is not computed for these!
    '''

    def _steady_state(self, calibration, backward_tol=1E-8, backward_maxit=5000,
                      forward_tol=1E-10, forward_maxit=100_000):
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
                - A seed for each backward variable, e.g. Va=... and Vb=... if self.back_iter_vars==('Va','Vb')
                - A grid for each policy variable, e.g. a_grid=... and b_grid=... if self.policy==('a','b')
                - All other inputs to the backward iteration function self.back_step_fun, except _p added to
                    for self.exogenous and self.back_iter_vars, for which the method uses steady-state values.
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
            - ss aggregates (in uppercase) for all outputs of self.back_step_fun except self.back_iter_vars
        """

        ss = calibration.toplevel.copy()

        # extract information from calibration
        Pi = ss[self.exogenous]
        grid = {k: ss[k+'_grid'] for k in self.policy}
        D_seed = ss.get('D', None)
        pi_seed = ss.get(self.exogenous + '_seed', None)

        # run backward iteration
        sspol = self.policy_ss(ss, tol=backward_tol, maxit=backward_maxit)
        ss.update(sspol)

        # run forward iteration
        D = self.dist_ss(Pi, sspol, grid, forward_tol, forward_maxit, D_seed, pi_seed)
        ss.update({"D": D})

        # run hetoutput if it's there
        if self.hetoutput is not None:
            ss.update(self.hetoutput(ss))

        # aggregate all outputs other than backward variables on grid, capitalize
        toreturn = self.non_back_iter_outputs
        if self.hetoutput is not None:
            toreturn = toreturn | self.hetoutput.outputs
        aggregates = {o.capitalize(): np.vdot(D, ss[o]) for o in toreturn}
        ss.update(aggregates)

        return SteadyStateDict({k: ss[k] for k in ss if k not in self.internal},
                               {self.name: {k: ss[k] for k in ss if k in self.internal}})

    def _impulse_nonlinear(self, ss, inputs, outputs, Js, monotonic=False, returnindividual=False, grid_paths=None):
        """Evaluate transitional dynamics for HetBlock given dynamic paths for `inputs`,
        assuming that we start and end in steady state `ss`, and that all inputs not specified in
        `inputs` are constant at their ss values.

        CANNOT provide time-varying Markov transition matrix for now.

        Block-specific inputs
        ---------------------
        monotonic : [optional] bool
            flag indicating date-t policies are monotonic in same date-(t-1) policies, allows us
            to use faster interpolation routines, otherwise use slower robust to nonmonotonicity
        returnindividual : [optional] bool
            return distribution and full outputs on grid
        grid_paths: [optional] dict of {str: array(T, Number of grid points)}
            time-varying grids for policies
        """
        T = inputs.T
        Pi_T = ss.internal[self.name][self.exogenous].T.copy()
        D = ss.internal[self.name]['D']

        # construct grids for policy variables either from the steady state grid if the grid is meant to be
        # non-time-varying or from the provided `grid_path` if the grid is meant to be time-varying.
        grid = {}
        use_ss_grid = {}
        for k in self.policy:
            if grid_paths is not None and k in grid_paths:
                grid[k] = grid_paths[k]
                use_ss_grid[k] = False
            else:
                grid[k] = ss[k + "_grid"]
                use_ss_grid[k] = True

        # allocate empty arrays to store result, assume all like D
        toreturn = self.non_back_iter_outputs
        if self.hetoutput is not None:
            toreturn = toreturn | self.hetoutput.outputs
        individual_paths = {k: np.empty((T,) + D.shape) for k in toreturn}

        # backward iteration
        backdict = dict(ss.items())
        backdict.update(copy.deepcopy(ss.internal[self.name]))
        for t in reversed(range(T)):
            # be careful: if you include vars from self.back_iter_vars in exogenous, agents will use them!
            backdict.update({k: ss[k] + v[t, ...] for k, v in inputs.items()})
            individual = self.make_inputs(backdict)
            individual.update(self.back_step_fun(individual))
            backdict.update({k: individual[k] for k in self.back_iter_vars})

            if self.hetoutput is not None:
                individual.update(self.hetoutput(individual))

            for k in self.non_back_iter_outputs:
                individual_paths[k][t, ...] = individual[k]

        D_path = np.empty((T,) + D.shape)
        D_path[0, ...] = D
        for t in range(T-1):
            # have to interpolate policy separately for each t to get sparse transition matrices
            sspol_i = {}
            sspol_pi = {}
            for pol in self.policy:
                if use_ss_grid[pol]:
                    grid_var = grid[pol]
                else:
                    grid_var = grid[pol][t, ...]
                if monotonic:
                    # TODO: change for two-asset case so assumption is monotonicity in own asset, not anything else
                    sspol_i[pol], sspol_pi[pol] = utils.interpolate.interpolate_coord(grid_var,
                                                                                      individual_paths[pol][t, ...])
                else:
                    sspol_i[pol], sspol_pi[pol] =\
                        utils.interpolate.interpolate_coord_robust(grid_var, individual_paths[pol][t, ...])

            # step forward
            D_path[t+1, ...] = self.forward_step(D_path[t, ...], Pi_T, sspol_i, sspol_pi)

        # obtain aggregates of all outputs, made uppercase
        aggregates = {o.capitalize(): utils.optimized_routines.fast_aggregate(D_path, individual_paths[o])
                      for o in individual_paths}

        # return either this, or also include distributional information
        # TODO: rethink this
        if returnindividual:
            return ImpulseDict({**aggregates, **individual_paths, 'D': D_path}) - ss
        else:
            return ImpulseDict(aggregates)[outputs] - ss


    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))


    def _jacobian(self, ss, inputs, outputs, T, h=1E-4):
        # TODO: h is unusable for now, figure out how to suggest options
        """
        Block-specific inputs
        ---------------------
        h : [optional] float
            h for numerical differentiation of backward iteration
        """
        outputs = self.M_outputs.inv @ outputs # horrible

        # TODO: this is one instance of us letting people supply inputs that aren't actually inputs
        # This behavior should lead to an error instead (probably should be handled at top level)
        relevant_shocks = self.back_step_fun.inputs
        if self.hetinput is not None:
            relevant_shocks = relevant_shocks | self.hetinput.inputs
        relevant_shocks = relevant_shocks & inputs

        # step 0: preliminary processing of steady state
        Pi, differentiable_back_step_fun, differentiable_hetinput, sspol_i, sspol_pi, sspol_space = self.jac_prelim(ss, h)

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = {}, {}
        for i in relevant_shocks:
            curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(i, outputs, differentiable_back_step_fun,
                                                                      ss.internal[self.name]['D'], Pi.T.copy(),
                                                                      sspol_i, sspol_pi, sspol_space, T,
                                                                      differentiable_hetinput)

        # step 2 of fake news algorithm
        # compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = {}
        for o in outputs:
            curlyPs[o] = self.forward_iteration_fakenews(ss.internal[self.name][o], Pi, sspol_i, sspol_pi, T-1)

        # steps 3-4 of fake news algorithm
        # make fake news matrix and Jacobian for each outcome-input pair
        F, J = {}, {}
        for o in outputs:
            for i in relevant_shocks:
                if o.capitalize() not in F:
                    F[o.capitalize()] = {}
                if o.capitalize() not in J:
                    J[o.capitalize()] = {}
                F[o.capitalize()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.capitalize()][i] = HetBlock.J_from_F(F[o.capitalize()][i])

        return JacobianDict(J, name=self.name, T=T)

    def add_hetinput(self, hetinput, overwrite=False, verbose=True):
        # TODO: serious violation, this is mutating the block
        """Add a hetinput to this HetBlock. Any call to self.back_step_fun will first process
         inputs through the hetinput function.

        A `hetinput` is any non-scalar-valued input argument provided to the HetBlock's backward iteration function,
        self.back_step_fun, which is of the same dimensions as the distribution of agents in the HetBlock over
        the relevant idiosyncratic state variables, generally referred to as `D`. e.g. The one asset HANK model
        example provided in the models directory of sequence_jacobian has a hetinput `T`, which is skill-specific
        transfers.
        """
        if self.hetinput is not None and overwrite is False:
            raise ValueError('Trying to attach hetinput when one already exists!')
        else:
            if verbose:
                if self.hetinput is not None and overwrite is True:
                    print(f"Overwriting current hetinput, {self.hetinput.__name__} with new hetinput,"
                          f" {hetinput.__name__}!")
                else:
                    print(f"Added hetinput {hetinput.__name__} to the {self.back_step_fun.__name__} HetBlock")

            self.hetinput = ExtendedFunction(hetinput)
            self.inputs |= self.hetinput.inputs
            self.inputs -= self.hetinput.outputs

            self.internal |= self.hetinput.outputs

    def add_hetoutput(self, hetoutput, overwrite=False, verbose=True):
        """Add a hetoutput to this HetBlock. Any call to self.back_step_fun will first process
         inputs through the hetoutput function.

        A `hetoutput` is any *non-scalar-value* output that the user might desire to be calculated from
        the output arguments of the HetBlock's backward iteration function. Importantly, as of now the `hetoutput`
        cannot be a function of time displaced values of the HetBlock's outputs but rather must be able to
        be calculated from the outputs statically. e.g. The two asset HANK model example provided in the models
        directory of sequence_jacobian has a hetoutput, `chi`, the adjustment costs for any initial level of assets
        `a`, to any new level of assets `a'`.
         """
        if self.hetoutput is not None and overwrite is False:
            raise ValueError('Trying to attach hetoutput when one already exists!')
        else:
            if verbose:
                if self.hetoutput is not None and overwrite is True:
                    print(f"Overwriting current hetoutput, {self.hetoutput.name} with new hetoutput,"
                          f" {hetoutput.name}!")
                else:
                    print(f"Added hetoutput {hetoutput.name} to the {self.back_step_fun.__name__} HetBlock")

            self.hetoutput = ExtendedFunction(hetoutput)

            # Modify the HetBlock's inputs to include additional inputs required for computing both the hetoutput
            # and aggregating the hetoutput, but do not include:
            # 1) objects computed within the HetBlock's backward iteration that enter into the hetoutput computation
            # 2) objects computed within hetoutput that enter into hetoutput's aggregation (self.hetoutput.outputs)
            # 3) D, the cross-sectional distribution of agents, which is used in the hetoutput aggregation
            # but is computed after the backward iteration
            self.inputs |= (self.hetoutput.inputs - self.hetinput.outputs - self.back_step_fun.outputs - self.hetoutput.outputs - set("D"))
            # Modify the HetBlock's outputs to include the aggregated hetoutputs
            self.outputs |= set([o.capitalize() for o in self.hetoutput.outputs])
            self.M_outputs = Bijection({o: o.capitalize() for o in self.hetoutput.outputs}) @ self.M_outputs

            self.internal |= self.hetoutput.outputs

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

        # find initial values for backward iteration and account for hetinputs
        original_ssin = ssin
        ssin = self.make_inputs(ssin)

        old = {}
        for it in range(maxit):
            try:
                # run and store results of backward iteration, which come as tuple, in dict
                sspol = self.back_step_fun(ssin)
            except KeyError as e:
                print(f'Missing input {e} to {self.back_step_fun.__name__}!')
                raise

            # only check convergence every 10 iterations for efficiency
            if it % 10 == 1 and all(utils.optimized_routines.within_tolerance(sspol[k], old[k], tol)
                                    for k in self.policy):
                break

            # update 'old' for comparison during next iteration, prepare 'ssin' as input for next iteration
            old.update({k: sspol[k] for k in self.policy})
            ssin.update({k + '_p': sspol[k] for k in self.back_iter_vars})
        else:
            raise ValueError(f'No convergence of policy functions after {maxit} backward iterations!')

        # want to record inputs in ssin, but remove _p, add in hetinput inputs if there
        for k in self.inputs_to_be_primed:
            ssin[k] = ssin[k + '_p']
            del ssin[k + '_p']
        if self.hetinput is not None:
            for k in self.hetinput.inputs:
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
            pi = utils.discretize.stationary(Pi, pi_seed)

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
            sspol_i[pol], sspol_pi[pol] = utils.interpolate.interpolate_coord_robust(grid[pol], sspol[pol])

        # iterate until convergence by tol, or maxit
        Pi_T = Pi.T.copy()
        for it in range(maxit):
            Dnew = self.forward_step(D, Pi_T, sspol_i, sspol_pi)

            # only check convergence every 10 iterations for efficiency
            if it % 10 == 0 and utils.optimized_routines.within_tolerance(D, Dnew, tol):
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

    def backward_step_fakenews(self, din_dict, output_list, differentiable_back_step_fun,
                               Dss, Pi_T, sspol_i, sspol_pi, sspol_space):
        # shock perturbs outputs
        shocked_outputs = differentiable_back_step_fun.diff(din_dict)
        curlyV = {k: shocked_outputs[k] for k in self.back_iter_vars}

        # which affects the distribution tomorrow
        pol_pi_shock = {k: -shocked_outputs[k] / sspol_space[k] for k in self.policy}

        curlyD = self.forward_step_shock(Dss, Pi_T, sspol_i, sspol_pi, pol_pi_shock)

        # and the aggregate outcomes today
        curlyY = {k: np.vdot(Dss, shocked_outputs[k]) for k in output_list}

        return curlyV, curlyD, curlyY

    def backward_iteration_fakenews(self, input_shocked, output_list, differentiable_back_step_fun, Dss, Pi_T,
                                    sspol_i, sspol_pi, sspol_space, T, differentiable_hetinput):
        """Iterate policy steps backward T times for a single shock."""
        # TODO: Might need to add a check for ss_for_hetinput if self.hetinput is not None
        #   since unless self.hetinput_inputs is exactly equal to input_shocked, calling
        #   self.hetinput() inside the symmetric differentiation function will throw an error.
        #   It's probably better/more informative to throw that error out here.
        if self.hetinput is not None and input_shocked in self.hetinput.inputs:
            # if input_shocked is an input to hetinput, take numerical diff to get response
            din_dict = differentiable_hetinput.diff2({input_shocked: 1})
        else:
            # otherwise, we just have that one shock
            din_dict = {input_shocked: 1}

        # contemporaneous response to unit scalar shock
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, differentiable_back_step_fun,
                                                             Dss, Pi_T, sspol_i, sspol_pi, sspol_space)

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
                                                    output_list, differentiable_back_step_fun,
                                                    Dss, Pi_T, sspol_i, sspol_pi, sspol_space)
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
        curlyPs[0, ...] = utils.misc.demean(o_ss)
        for t in range(1, T):
            curlyPs[t, ...] = utils.misc.demean(self.forward_step_transpose(curlyPs[t - 1, ...],
                                                                            Pi, pol_i_ss, pol_pi_ss))
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

    '''Part 5: helpers for .jac and .ajac: preliminary processing'''

    def jac_prelim(self, ss, h):
        """Helper that does preliminary processing of steady state for fake news algorithm.

        Parameters
        ----------
        ss : dict, all steady-state info, intended to be from .ss()

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
        # preliminary a: obtain ss inputs and other info, run once to get baseline for numerical differentiation
        ssin_dict = self.make_inputs(ss)
        Pi = ss.internal[self.name][self.exogenous]
        grid = {k: ss[k+'_grid'] for k in self.policy}
        #ssout_list = self.back_step_fun(ssin_dict)
        differentiable_back_step_fun = self.back_step_fun.differentiable(ssin_dict, h=h)

        differentiable_hetinput = None
        if self.hetinput is not None:
            # ss_for_hetinput = {k: ss[k] for k in self.hetinput_inputs if k in ss}
            differentiable_hetinput = self.hetinput.differentiable(ss)

        # preliminary b: get sparse representations of policy rules, and distance between neighboring policy gridpoints
        sspol_i = {}
        sspol_pi = {}
        sspol_space = {}
        for pol in self.policy:
            # use robust binary-search-based method that only requires grids to be monotonic
            sspol_i[pol], sspol_pi[pol] = utils.interpolate.interpolate_coord_robust(grid[pol], ss.internal[self.name][pol])
            sspol_space[pol] = grid[pol][sspol_i[pol]+1] - grid[pol][sspol_i[pol]]

        return Pi, differentiable_back_step_fun, differentiable_hetinput, sspol_i, sspol_pi, sspol_space

    '''Part 6: helper to extract inputs and potentially process them through hetinput'''

    def make_inputs(self, back_step_inputs_dict):
        """Extract from back_step_inputs_dict exactly the inputs needed for self.back_step_fun,
        process stuff through self.hetinput first if it's there.
        """
        if isinstance(back_step_inputs_dict, SteadyStateDict):
            input_dict = {**back_step_inputs_dict.toplevel, **back_step_inputs_dict.internal[self.name]}
        else:
            input_dict = back_step_inputs_dict.copy()

        if self.hetinput is not None:
            input_dict.update(self.hetinput(input_dict))

        if not all(k in input_dict for k in self.back_iter_vars):
            input_dict.update(self.backward_init(input_dict))

        for i_p in self.inputs_to_be_primed:
            input_dict[i_p + "_p"] = input_dict[i_p]
            del input_dict[i_p]

        try:
            return {k: input_dict[k] for k in self.back_step_fun.inputs if k in input_dict}
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
            return utils.forward_step.forward_step_1d(D, Pi_T, pol_i[p], pol_pi[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step.forward_step_2d(D, Pi_T, pol_i[p1], pol_i[p2], pol_pi[p1], pol_pi[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

    def forward_step_transpose(self, D, Pi, pol_i, pol_pi):
        """Transpose of forward_step (note: this takes Pi rather than Pi_T as argument!)"""
        if len(self.policy) == 1:
            p, = self.policy
            return utils.forward_step.forward_step_transpose_1d(D, Pi, pol_i[p], pol_pi[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step.forward_step_transpose_2d(D, Pi, pol_i[p1], pol_i[p2], pol_pi[p1], pol_pi[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

    def forward_step_shock(self, Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_pi_shock):
        """Forward_step linearized with respect to pol_pi"""
        if len(self.policy) == 1:
            p, = self.policy
            return utils.forward_step.forward_step_shock_1d(Dss, Pi_T, pol_i_ss[p], pol_pi_shock[p])
        elif len(self.policy) == 2:
            p1, p2 = self.policy
            return utils.forward_step.forward_step_shock_2d(Dss, Pi_T, pol_i_ss[p1], pol_i_ss[p2],
                                                            pol_pi_ss[p1], pol_pi_ss[p2],
                                                            pol_pi_shock[p1], pol_pi_shock[p2])
        else:
            raise ValueError(f"{len(self.policy)} policy variables, only up to 2 implemented!")

