import warnings
import copy
import numpy as np

from .support.impulse import ImpulseDict
from ..primitives import Block
from .. import utilities as utils
from ..jacobian.classes import JacobianDict
from ..devtools.deprecate import rename_output_list_to_outputs


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

        # self.back_step_fun is one iteration of the backward step function pertaining to a given HetBlock.
        # i.e. the function pertaining to equation (14) in the paper: v_t = curlyV(v_{t+1}, X_t)
        self.back_step_fun = back_step_fun

        # self.back_step_outputs and self.back_step_inputs are all of the output and input arguments of
        # self.back_step_fun, the variables used in the backward iteration,
        # which generally include value and/or policy functions.
        self.back_step_output_list = utils.misc.output_list(back_step_fun)
        self.back_step_outputs = set(self.back_step_output_list)
        self.back_step_inputs = set(utils.misc.input_list(back_step_fun))

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
        self.non_back_iter_outputs = self.back_step_outputs - set(self.back_iter_vars)

        # self.outputs and self.inputs are the *aggregate* outputs and inputs of this HetBlock, which are used
        # in utils.graph.block_sort to topologically sort blocks along the DAG
        # according to their aggregate outputs and inputs.
        self.outputs = {o.capitalize() for o in self.non_back_iter_outputs}
        self.inputs = self.back_step_inputs - {k + '_p' for k in self.back_iter_vars}
        self.inputs.remove(exogenous + '_p')
        self.inputs.add(exogenous)

        # A HetBlock can have heterogeneous inputs and heterogeneous outputs, henceforth `hetinput` and `hetoutput`.
        # See docstring for methods `add_hetinput` and `add_hetoutput` for more details.
        self.hetinput = None
        self.hetinput_inputs = set()
        self.hetinput_outputs = set()
        self.hetinput_outputs_order = tuple()

        # start without a hetoutput
        self.hetoutput = None
        self.hetoutput_inputs = set()
        self.hetoutput_outputs = set()
        self.hetoutput_outputs_order = tuple()

        if len(self.policy) > 2:
            raise ValueError(f"More than two endogenous policies in {back_step_fun.__name__}, not yet supported")

        # Checking that the various inputs/outputs attributes are correctly set
        if self.exogenous + '_p' not in self.back_step_inputs:
            raise ValueError(f"Markov matrix '{self.exogenous}_p' not included as argument in {back_step_fun.__name__}")
        
        for pol in self.policy:
            if pol not in self.back_step_outputs:
                raise ValueError(f"Policy '{pol}' not included as output in {back_step_fun.__name__}")
            if pol[0].isupper():
                raise ValueError(f"Policy '{pol}' is uppercase in {back_step_fun.__name__}, which is not allowed")

        for back in self.back_iter_vars:
            if back + '_p' not in self.back_step_inputs:
                raise ValueError(f"Backward variable '{back}_p' not included as argument in {back_step_fun.__name__}")

            if back not in self.back_step_outputs:
                raise ValueError(f"Backward variable '{back}' not included as output in {back_step_fun.__name__}")

        for out in self.non_back_iter_outputs:
            if out[0].isupper():
                raise ValueError("Output '{out}' is uppercase in {back_step_fun.__name__}, which is not allowed")

        # Add the backward iteration initializer function (the initial guesses for self.back_iter_vars)
        if backward_init is None:
            # TODO: Think about implementing some "automated way" of providing
            #  an initial guess for the backward iteration.
            self.backward_init = backward_init
        else:
            self.backward_init = backward_init

        # note: should do more input checking to ensure certain choices not made: 'D' not input, etc.

    def __repr__(self):
        """Nice string representation of HetBlock for printing to console"""
        if self.hetinput is not None:
            if self.hetoutput is not None:
                return f"<HetBlock '{self.back_step_fun.__name__}' with hetinput '{self.hetinput.__name__}'" \
                       f" and with hetoutput `{self.hetoutput.name}'>"
            else:
                return f"<HetBlock '{self.back_step_fun.__name__}' with hetinput '{self.hetinput.__name__}'>"
        else:
            return f"<HetBlock '{self.back_step_fun.__name__}'>"

    '''Part 2: high-level routines, with first three called analogously to SimpleBlock counterparts
        - ss    : do backward and forward iteration until convergence to get complete steady state
        - td    : do backward and forward iteration up to T to compute dynamics given some shocks
        - jac   : compute jacobians of outputs with respect to shocked inputs, using fake news algorithm
        - ajac  : compute asymptotic columns of jacobians output by jac, also using fake news algorithm

        - add_hetinput : add a hetinput to the HetBlock that first processes inputs through function hetinput
        - add_hetoutput: add a hetoutput to the HetBlock that is computed after the entire ss computation, or after
                         each backward iteration step in td
    '''

    # TODO: Deprecated methods, to be removed!
    def ss(self, **kwargs):
        warnings.warn("This method has been deprecated. Please invoke by calling .steady_state", DeprecationWarning)
        return self.steady_state(kwargs)

    def td(self, ss, **kwargs):
        warnings.warn("This method has been deprecated. Please invoke by calling .impulse_nonlinear",
                      DeprecationWarning)
        return self.impulse_nonlinear(ss, **kwargs)

    def jac(self, ss, shock_list=None, T=None, **kwargs):
        if shock_list is None:
            shock_list = list(self.inputs)
        warnings.warn("This method has been deprecated. Please invoke by calling .jacobian.\n"
                      "Also, note that the kwarg `shock_list` in .jacobian has been renamed to `shocked_vars`",
                      DeprecationWarning)
        return self.jacobian(ss, shock_list, T, **kwargs)

    def steady_state(self, calibration, backward_tol=1E-8, backward_maxit=5000,
                     forward_tol=1E-10, forward_maxit=100_000, hetoutput=False):
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

        ss = copy.deepcopy(calibration)

        # extract information from calibration
        Pi = calibration[self.exogenous]
        grid = {k: calibration[k+'_grid'] for k in self.policy}
        D_seed = calibration.get('D', None)
        pi_seed = calibration.get(self.exogenous + '_seed', None)

        # run backward iteration
        sspol = self.policy_ss(calibration, tol=backward_tol, maxit=backward_maxit)
        ss.update(sspol)

        # run forward iteration
        D = self.dist_ss(Pi, sspol, grid, forward_tol, forward_maxit, D_seed, pi_seed)
        ss.update({"D": D})

        # aggregate all outputs other than backward variables on grid, capitalize
        aggregates = {o.capitalize(): np.vdot(D, sspol[o]) for o in self.non_back_iter_outputs}
        ss.update(aggregates)

        if hetoutput:
            hetoutputs = self.hetoutput.evaluate(ss)
            aggregate_hetoutputs = self.hetoutput.aggregate(hetoutputs, D, ss, mode="ss")
        else:
            hetoutputs = {}
            aggregate_hetoutputs = {}
        ss.update({**hetoutputs, **aggregate_hetoutputs})

        return ss

    def impulse_nonlinear(self, ss, exogenous, monotonic=False, returnindividual=False, grid_paths=None):
        """Evaluate transitional dynamics for HetBlock given dynamic paths for inputs in kwargs,
        assuming that we start and end in steady state ss, and that all inputs not specified in
        kwargs are constant at their ss values. Analog to SimpleBlock.td.

        CANNOT provide time-varying paths of grid or Markov transition matrix for now.
        
        Parameters
        ----------
        ss : dict
            all steady-state info, intended to be from .ss()
        exogenous : dict of {str : array(T, ...)}
            all time-varying inputs here (in deviations), with first dimension being time
            this must have same length T for all entries (all outputs will be calculated up to T)
        monotonic : [optional] bool
            flag indicating date-t policies are monotonic in same date-(t-1) policies, allows us
            to use faster interpolation routines, otherwise use slower robust to nonmonotonicity
        returnindividual : [optional] bool
            return distribution and full outputs on grid
        grid_paths: [optional] dict of {str: array(T, Number of grid points)}
            time-varying grids for policies

        Returns
        ----------
        td : dict
            if returnindividual = False, time paths for aggregates (uppercase) for all outputs
                of self.back_step_fun except self.back_iter_vars
            if returnindividual = True, additionally time paths for distribution and for all outputs
                of self.back_Step_fun on the full grid
        """
        # infer T from exogenous, check that all shocks have same length
        shock_lengths = [x.shape[0] for x in exogenous.values()]
        if shock_lengths[1:] != shock_lengths[:-1]:
            raise ValueError('Not all shocks in kwargs (exogenous) are same length!')
        T = shock_lengths[0]

        # copy from ss info
        Pi_T = ss[self.exogenous].T.copy()
        D = ss['D']

        # construct grids for policy variables either from the steady state grid if the grid is meant to be
        # non-time-varying or from the provided `grid_path` if the grid is meant to be time-varying.
        grid = {}
        use_ss_grid = {}
        for k in self.policy:
            if grid_paths is not None and k in grid_paths:
                grid[k] = grid_paths[k]
                use_ss_grid[k] = False
            else:
                grid[k] = ss[k+"_grid"]
                use_ss_grid[k] = True

        # allocate empty arrays to store result, assume all like D
        individual_paths = {k: np.empty((T,) + D.shape) for k in self.non_back_iter_outputs}
        hetoutput_paths = {k: np.empty((T,) + D.shape) for k in self.hetoutput_outputs}

        # backward iteration
        backdict = ss.copy()
        for t in reversed(range(T)):
            # be careful: if you include vars from self.back_iter_vars in exogenous, agents will use them!
            backdict.update({k: ss[k] + v[t, ...] for k, v in exogenous.items()})
            individual = {k: v for k, v in zip(self.back_step_output_list,
                                               self.back_step_fun(**self.make_inputs(backdict)))}
            backdict.update({k: individual[k] for k in self.back_iter_vars})

            if self.hetoutput is not None:
                hetoutput = self.hetoutput.evaluate(backdict)
                for k in self.hetoutput_outputs:
                    hetoutput_paths[k][t, ...] = hetoutput[k]

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
                      for o in self.non_back_iter_outputs}
        if self.hetoutput:
            aggregate_hetoutputs = self.hetoutput.aggregate(hetoutput_paths, D_path, backdict, mode="td")
        else:
            aggregate_hetoutputs = {}

        # return either this, or also include distributional information
        if returnindividual:
            return ImpulseDict({**aggregates, **aggregate_hetoutputs, **individual_paths, **hetoutput_paths,
                                'D': D_path}, ss)
        else:
            return ImpulseDict({**aggregates, **aggregate_hetoutputs}, ss)

    def impulse_linear(self, ss, exogenous, T=None, **kwargs):
        # infer T from exogenous, check that all shocks have same length
        shock_lengths = [x.shape[0] for x in exogenous.values()]
        if shock_lengths[1:] != shock_lengths[:-1]:
            raise ValueError('Not all shocks in kwargs (exogenous) are same length!')
        T = shock_lengths[0]

        return ImpulseDict(self.jacobian(ss, list(exogenous.keys()), T=T, **kwargs).apply(exogenous), ss)

    def jacobian(self, ss, exogenous=None, T=300, outputs=None, output_list=None, h=1E-4):
        """Assemble nested dict of Jacobians of agg outputs vs. inputs, using fake news algorithm.

        Parameters
        ----------
        ss : dict,
            all steady-state info, intended to be from .ss()
        T : [optional] int
            number of time periods for T*T Jacobian
        exogenous : list of str
            names of input variables to differentiate wrt (main cost scales with # of inputs)
        outputs : list of str
            names of output variables to get derivatives of, if not provided assume all outputs of
            self.back_step_fun except self.back_iter_vars
        h : [optional] float
            h for numerical differentiation of backward iteration

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives T*T Jacobian of o with respect to i
        """
        # The default set of outputs are all outputs of the backward iteration function
        # except for the backward iteration variables themselves
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None or output_list is None:
            outputs = self.non_back_iter_outputs
        else:
            outputs = rename_output_list_to_outputs(outputs=outputs, output_list=output_list)

        relevant_shocks = [i for i in self.back_step_inputs | self.hetinput_inputs if i in exogenous]

        # TODO: get rid of this
        # if we're supposed to use saved Jacobian, extract T-by-T submatrices for each (o,i)
        # if use_saved:
        #     return utils.misc.extract_nested_dict(savedA=self.saved['J'],
        #                                           keys1=[o.capitalize() for o in outputs],
        #                                           keys2=relevant_shocks, shape=(T, T))

        # step 0: preliminary processing of steady state
        (ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space) = self.jac_prelim(ss)

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = {}, {}
        for i in relevant_shocks:
            curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(i, outputs, ssin_dict, ssout_list,
                                                ss['D'], Pi.T.copy(), sspol_i, sspol_pi, sspol_space, T, h,
                                                ss_for_hetinput)

        # step 2 of fake news algorithm
        # compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = {}
        for o in outputs:
            curlyPs[o] = self.forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T-1)

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

        return JacobianDict(J, name=self.name)

    def add_hetinput(self, hetinput, overwrite=False, verbose=True):
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

            self.hetinput = hetinput
            self.hetinput_inputs = set(utils.misc.input_list(hetinput))
            self.hetinput_outputs = set(utils.misc.output_list(hetinput))
            self.hetinput_outputs_order = utils.misc.output_list(hetinput)

            # modify inputs to include hetinput's additional inputs, remove outputs
            self.inputs |= self.hetinput_inputs
            self.inputs -= self.hetinput_outputs

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

            self.hetoutput = hetoutput
            self.hetoutput_inputs = set(hetoutput.input_list)
            self.hetoutput_outputs = set(hetoutput.output_list)
            self.hetoutput_outputs_order = hetoutput.output_list

            # Modify the HetBlock's inputs to include additional inputs required for computing both the hetoutput
            # and aggregating the hetoutput, but do not include:
            # 1) objects computed within the HetBlock's backward iteration that enter into the hetoutput computation
            # 2) objects computed within hetoutput that enter into hetoutput's aggregation (self.hetoutput.outputs)
            # 3) D, the cross-sectional distribution of agents, which is used in the hetoutput aggregation
            # but is computed after the backward iteration
            self.inputs |= (self.hetoutput_inputs - self.hetinput_outputs - self.back_step_outputs - self.hetoutput_outputs - set("D"))
            # Modify the HetBlock's outputs to include the aggregated hetoutputs
            self.outputs |= set([o.capitalize() for o in self.hetoutput_outputs])

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
                sspol = {k: v for k, v in zip(self.back_step_output_list, self.back_step_fun(**ssin))}
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

    def backward_step_fakenews(self, din_dict, output_list, ssin_dict, ssout_list, 
                               Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h=1E-4):
        # shock perturbs outputs
        shocked_outputs = {k: v for k, v in zip(self.back_step_output_list,
                                                utils.differentiate.numerical_diff(self.back_step_fun,
                                                                                   ssin_dict, din_dict, h,
                                                                                   ssout_list))}
        curlyV = {k: shocked_outputs[k] for k in self.back_iter_vars}

        # which affects the distribution tomorrow
        pol_pi_shock = {k: -shocked_outputs[k] / sspol_space[k] for k in self.policy}

        # Include an additional term to account for the effect of a deleveraging shock affecting the grid
        if "delev_exante" in din_dict:
            dx = np.zeros_like(sspol_pi["a"])
            dx[sspol_i["a"] == 0] = 1.
            add_term = sspol_pi["a"] * dx / sspol_space["a"]
            pol_pi_shock["a"] += add_term

        curlyD = self.forward_step_shock(Dss, Pi_T, sspol_i, sspol_pi, pol_pi_shock)

        # and the aggregate outcomes today
        curlyY = {k: np.vdot(Dss, shocked_outputs[k]) for k in output_list}

        return curlyV, curlyD, curlyY

    def backward_iteration_fakenews(self, input_shocked, output_list, ssin_dict, ssout_list, Dss, Pi_T, 
                                    sspol_i, sspol_pi, sspol_space, T, h=1E-4, ss_for_hetinput=None):
        """Iterate policy steps backward T times for a single shock."""
        # TODO: Might need to add a check for ss_for_hetinput if self.hetinput is not None
        #   since unless self.hetinput_inputs is exactly equal to input_shocked, calling
        #   self.hetinput() inside the symmetric differentiation function will throw an error.
        #   It's probably better/more informative to throw that error out here.
        if self.hetinput is not None and input_shocked in self.hetinput_inputs:
            # if input_shocked is an input to hetinput, take numerical diff to get response
            din_dict = dict(zip(self.hetinput_outputs_order,
                                utils.differentiate.numerical_diff_symmetric(self.hetinput,
                                                                             ss_for_hetinput, {input_shocked: 1}, h)))
        else:
            # otherwise, we just have that one shock
            din_dict = {input_shocked: 1}

        # contemporaneous response to unit scalar shock
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, ssin_dict, ssout_list, 
                                                             Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h=h)

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

    def jac_prelim(self, ss):
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
            sspol_i[pol], sspol_pi[pol] = utils.interpolate.interpolate_coord_robust(grid[pol], ss[pol])
            sspol_space[pol] = grid[pol][sspol_i[pol]+1] - grid[pol][sspol_i[pol]]

        toreturn = (ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space)

        return toreturn

    '''Part 6: helper to extract inputs and potentially process them through hetinput'''

    def make_inputs(self, back_step_inputs_dict):
        """Extract from back_step_inputs_dict exactly the inputs needed for self.back_step_fun,
        process stuff through self.hetinput first if it's there.
        """
        input_dict = copy.deepcopy(back_step_inputs_dict)

        # If this HetBlock has a hetinput, then we need to compute the outputs of the hetinput first and include
        # them as inputs for self.back_step_fun
        if self.hetinput is not None:
            outputs_as_tuple = utils.misc.make_tuple(self.hetinput(**{k: input_dict[k]
                                                                      for k in self.hetinput_inputs if k in input_dict}))
            input_dict.update(dict(zip(self.hetinput_outputs_order, outputs_as_tuple)))

        # Check if there are entries in indict corresponding to self.inputs_to_be_primed.
        # In particular, we are interested in knowing if an initial value
        # for the backward iteration variable has been provided.
        # If it has not been provided, then use self.backward_init to calculate the initial values.
        if not self.inputs_to_be_primed.issubset(set(input_dict.keys())):
            initial_value_input_args = [input_dict[arg_name] for arg_name in utils.misc.input_list(self.backward_init)]
            input_dict.update(zip(utils.misc.output_list(self.backward_init),
                              utils.misc.make_tuple(self.backward_init(*initial_value_input_args))))

        for i_p in self.inputs_to_be_primed:
            input_dict[i_p + "_p"] = input_dict[i_p]
            del input_dict[i_p]

        try:
            return {k: input_dict[k] for k in self.back_step_inputs if k in input_dict}
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


def hetoutput(custom_aggregation=None):
    def decorator(f):
        return HetOutput(f, custom_aggregation=custom_aggregation)
    return decorator


class HetOutput:
    def __init__(self, f, custom_aggregation=None):
        self.name = f.__name__
        self.f = f
        self.eval_input_list = utils.misc.input_list(f)

        self.custom_aggregation = custom_aggregation
        self.agg_input_list = [] if custom_aggregation is None else utils.misc.input_list(custom_aggregation)

        # We are distinguishing between the eval_input_list and agg_input_list because custom aggregation may require
        # certain arguments that are not required for simply evaluating the hetoutput
        self.input_list = list(set(self.eval_input_list).union(set(self.agg_input_list)))
        self.output_list = utils.misc.output_list(f)

    def evaluate(self, arg_dict):
        hetoutputs = dict(zip(self.output_list, utils.misc.make_tuple(self.f(*[arg_dict[i] for i
                                                                               in self.eval_input_list]))))
        return hetoutputs

    def aggregate(self, hetoutputs, D, custom_aggregation_args, mode="ss"):
        if self.custom_aggregation is not None:
            hetoutputs_w_std_aggregation = list(set(self.output_list) -
                                                set([utils.misc.uncapitalize(o) for o
                                                     in utils.misc.output_list(self.custom_aggregation)]))
            hetoutputs_w_custom_aggregation = list(set(self.output_list) - set(hetoutputs_w_std_aggregation))
        else:
            hetoutputs_w_std_aggregation = self.output_list
            hetoutputs_w_custom_aggregation = []

        # TODO: May need to check if this works properly for td
        if self.custom_aggregation is not None:
            hetoutputs_w_custom_aggregation_args = dict(zip(hetoutputs_w_custom_aggregation,
                                                        [hetoutputs[i] for i in hetoutputs_w_custom_aggregation]))
            custom_agg_inputs = {"D": D, **hetoutputs_w_custom_aggregation_args, **custom_aggregation_args}
            custom_aggregates = dict(zip([o.capitalize() for o in hetoutputs_w_custom_aggregation],
                                         utils.misc.make_tuple(self.custom_aggregation(*[custom_agg_inputs[i] for i
                                                                                         in self.agg_input_list]))))
        else:
            custom_aggregates = {}

        if mode == "ss":
            std_aggregates = {o.capitalize(): np.vdot(D, hetoutputs[o]) for o in hetoutputs_w_std_aggregation}
        elif mode == "td":
            std_aggregates = {o.capitalize(): utils.optimized_routines.fast_aggregate(D, hetoutputs[o])
                              for o in hetoutputs_w_std_aggregation}
        else:
            raise RuntimeError(f"Mode {mode} is not supported in HetOutput aggregation. Choose either 'ss' or 'td'")

        return {**std_aggregates, **custom_aggregates}
