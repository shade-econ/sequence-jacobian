import warnings
import copy
import numpy as np

from .support.impulse import ImpulseDict
from ..primitives import Block
from .. import utilities as utils
from ..steady_state.classes import SteadyStateDict
from ..jacobian.classes import JacobianDict
from ..utilities.misc import verify_saved_jacobian


def discont(exogenous, policy, disc_policy, backward, backward_init=None):
    def decorator(back_step_fun):
        return DiscontBlock(back_step_fun, exogenous, policy, disc_policy, backward, backward_init=backward_init)
    return decorator


class DiscontBlock(Block):
    """Part 1: Initializer for DiscontBlock, intended to be called via @hetdc() decorator on backward step function.

    IMPORTANT: All `policy` and non-aggregate output variables of this HetBlock need to be *lower-case*, since
    the methods that compute steady state, transitional dynamics, and Jacobians for HetBlocks automatically handle
    aggregation of non-aggregate outputs across the distribution and return aggregates as upper-case equivalents
    of the `policy` and non-aggregate output variables specified in the backward step function.
    """

    def __init__(self, back_step_fun, exogenous, policy, disc_policy, backward, backward_init=None):
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
        disc_policy: str
            name of policy function for discrete choices (probabilities)
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
        self.disc_policy = disc_policy
        self.policy = policy
        self.exogenous, self.back_iter_vars = (utils.misc.make_tuple(x) for x in (exogenous, backward))

        # self.inputs_to_be_primed indicates all variables that enter into self.back_step_fun whose name has "_p"
        # (read as prime). Because it's the case that the initial dict of input arguments for self.back_step_fun
        # contains the names of these variables that omit the "_p", we need to swap the key from the unprimed to
        # the primed key name, such that self.back_step_fun will properly call those variables.
        # e.g. the key "Va" will become "Va_p", associated to the same value.
        self.inputs_to_be_primed = set(self.exogenous) | set(self.back_iter_vars)

        # self.non_back_iter_outputs are all of the outputs from self.back_step_fun excluding the backward
        # iteration variables themselves.
        self.non_back_iter_outputs = self.back_step_outputs - set(self.back_iter_vars) - set(self.disc_policy)

        # self.outputs and self.inputs are the *aggregate* outputs and inputs of this HetBlock, which are used
        # in utils.graph.block_sort to topologically sort blocks along the DAG
        # according to their aggregate outputs and inputs.
        self.outputs = {o.capitalize() for o in self.non_back_iter_outputs}
        self.inputs = self.back_step_inputs - {k + '_p' for k in self.back_iter_vars}
        for ex in self.exogenous:
            self.inputs.remove(ex + '_p')
            self.inputs.add(ex)

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

        # The set of variables that will be wrapped in a separate namespace for this HetBlock
        # as opposed to being available at the top level
        self.internal = utils.misc.smart_set(self.back_step_outputs) | utils.misc.smart_set(self.exogenous) | {"D"}

        if len(self.policy) > 1:
            raise ValueError(f"More than one continuous states in {back_step_fun.__name__}, not yet supported")

        # Checking that the various inputs/outputs attributes are correctly set
        for ex in self.exogenous:
            if ex + '_p' not in self.back_step_inputs:
                raise ValueError(f"Markov matrix '{ex}_p' not included as argument in {back_step_fun.__name__}")

        if self.policy not in self.back_step_outputs:
            raise ValueError(f"Policy '{self.policy}' not included as output in {back_step_fun.__name__}")
        if self.policy[0].isupper():
            raise ValueError(f"Policy '{self.policy}' is uppercase in {back_step_fun.__name__}, which is not allowed")

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
                return f"<DiscontBlock '{self.back_step_fun.__name__}' with hetinput '{self.hetinput.__name__}'" \
                       f" and with hetoutput `{self.hetoutput.name}'>"
            else:
                return f"<DiscontBlock '{self.back_step_fun.__name__}' with hetinput '{self.hetinput.__name__}'>"
        else:
            return f"<DiscontBlock '{self.back_step_fun.__name__}'>"

    '''Part 2: high-level routines, with first three called analogously to SimpleBlock counterparts
        - ss    : do backward and forward iteration until convergence to get complete steady state
        - td    : do backward and forward iteration up to T to compute dynamics given some shocks
        - jac   : compute jacobians of outputs with respect to shocked inputs, using fake news algorithm
        - ajac  : compute asymptotic columns of jacobians output by jac, also using fake news algorithm

        - add_hetinput : add a hetinput to the HetBlock that first processes inputs through function hetinput
        - add_hetoutput: add a hetoutput to the HetBlock that is computed after the entire ss computation, or after
                         each backward iteration step in td
    '''


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
        grid = calibration[self.policy + '_grid']
        D_seed = calibration.get('D', None)

        # run backward iteration
        sspol = self.policy_ss(calibration, tol=backward_tol, maxit=backward_maxit)
        ss.update(sspol)

        # run forward iteration
        D = self.dist_ss(sspol, grid, forward_tol, forward_maxit, D_seed)
        ss.update({"D": D})

        # aggregate all outputs other than backward variables on grid, capitalize
        aggregates = {o.capitalize(): np.vdot(D, sspol[o]) for o in self.non_back_iter_outputs}
        ss.update(aggregates)

        if hetoutput and self.hetoutput is not None:
            hetoutputs = self.hetoutput.evaluate(ss)
            aggregate_hetoutputs = self.hetoutput.aggregate(hetoutputs, D, ss, mode="ss")
        else:
            hetoutputs = {}
            aggregate_hetoutputs = {}
        ss.update({**hetoutputs, **aggregate_hetoutputs})

        return SteadyStateDict(ss, internal=self)

    def impulse_nonlinear(self, ss, exogenous, returnindividual=False, grid_paths=None):
        """Evaluate transitional dynamics for DiscontBlock given dynamic paths for inputs in exogenous,
        assuming that we start and end in steady state ss, and that all inputs not specified in
        exogenous are constant at their ss values. Analog to SimpleBlock.td.

        Parameters
        ----------
        ss : SteadyStateDict
            all steady-state info, intended to be from .ss()
        exogenous : dict of {str : array(T, ...)}
            all time-varying inputs here (in deviations), with first dimension being time
            this must have same length T for all entries (all outputs will be calculated up to T)
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
        D, P = ss.internal[self.name]['D'], ss.internal[self.name][self.disc_policy]

        # construct grids for policy variables either from the steady state grid if the grid is meant to be
        # non-time-varying or from the provided `grid_path` if the grid is meant to be time-varying.
        if grid_paths is not None and self.policy in grid_paths:
            grid = grid_paths[self.policy]
            use_ss_grid = False
        else:
            grid = ss[self.policy + "_grid"]
            use_ss_grid = True
        # sspol_i, sspol_pi = utils.interpolate_coord_robust(grid, ss[self.policy])

        # allocate empty arrays to store result, assume all like D
        individual_paths = {k: np.empty((T,) + D.shape) for k in self.non_back_iter_outputs}
        hetoutput_paths = {k: np.empty((T,) + D.shape) for k in self.hetoutput_outputs}
        P_path = np.empty((T,) + P.shape)

        # obtain full path of multidimensional inputs
        multidim_inputs = {k: np.empty((T,) + ss.internal[self.name][k].shape) for k in self.hetinput_outputs_order}
        if self.hetinput is not None:
            indict = dict(ss.items())
            for t in range(T):
                indict.update({k: ss[k] + v[t, ...] for k, v in exogenous.items()})
                hetout = dict(zip(self.hetinput_outputs_order,
                                  self.hetinput(**{k: indict[k] for k in self.hetinput_inputs})))
                for k in self.hetinput_outputs_order:
                    multidim_inputs[k][t, ...] = hetout[k]

        # backward iteration
        backdict = dict(ss.items())
        backdict.update(copy.deepcopy(ss.internal[self.name]))
        for t in reversed(range(T)):
            # be careful: if you include vars from self.back_iter_vars in exogenous, agents will use them!
            backdict.update({k: ss[k] + v[t, ...] for k, v in exogenous.items()})

            # add in multidimensional inputs EXCEPT exogenous state transitions (at lead 0)
            backdict.update({k: ss.internal[self.name][k] + v[t, ...] for k, v in multidim_inputs.items() if k not in self.exogenous})

            # add in multidimensional inputs FOR exogenous state transitions (at lead 1)
            if t < T - 1:
                backdict.update({k: ss.internal[self.name][k] + v[t+1, ...] for k, v in multidim_inputs.items() if k in self.exogenous})

            # step back
            individual = {k: v for k, v in zip(self.back_step_output_list,
                                               self.back_step_fun(**self.make_inputs(backdict)))}

            # update backward variables
            backdict.update({k: individual[k] for k in self.back_iter_vars})

            # compute hetoutputs
            if self.hetoutput is not None:
                hetoutput = self.hetoutput.evaluate(backdict)
                for k in self.hetoutput_outputs:
                    hetoutput_paths[k][t, ...] = hetoutput[k]

            # save individual outputs of interest
            P_path[t, ...] = individual[self.disc_policy]
            for k in self.non_back_iter_outputs:
                individual_paths[k][t, ...] = individual[k]

        # forward iteration
        # initial markov matrix (may have been shocked)
        Pi_path = [[multidim_inputs[k][0, ...] if k in self.hetinput_outputs_order else ss[k] for k in self.exogenous]]

        # on impact: assets are predetermined, but Pi could be shocked, and P can change
        D_path = np.empty((T,) + D.shape)
        if use_ss_grid:
            grid_var = grid
        else:
            grid_var = grid[0, ...]
        sspol_i, sspol_pi = utils.interpolate.interpolate_coord_robust(grid_var, ss.internal[self.name][self.policy])
        D_path[0, ...] = self.forward_step(D, P_path[0, ...], Pi_path[0], sspol_i, sspol_pi)
        for t in range(T-1):
            # have to interpolate policy separately for each t to get sparse transition matrices
            if not use_ss_grid:
                grid_var = grid[t, ...]
            pol_i, pol_pi = utils.interpolate.interpolate_coord_robust(grid_var, individual_paths[self.policy][t, ...])

            # update exogenous Markov matrices
            Pi = [multidim_inputs[k][t+1, ...] if k in self.hetinput_outputs_order else ss[k] for k in self.exogenous]
            Pi_path.append(Pi)

            # step forward
            D_path[t+1, ...] = self.forward_step(D_path[t, ...], P_path[t+1, ...], Pi, pol_i, pol_pi)

        # obtain aggregates of all outputs, made uppercase
        aggregates = {o.capitalize(): utils.optimized_routines.fast_aggregate(D_path, individual_paths[o])
                      for o in self.non_back_iter_outputs}
        if self.hetoutput:
            aggregate_hetoutputs = self.hetoutput.aggregate(hetoutput_paths, D_path, backdict, mode="td")
        else:
            aggregate_hetoutputs = {}

        # return either this, or also include distributional information
        if returnindividual:
            return ImpulseDict({**aggregates, **aggregate_hetoutputs, **individual_paths, **multidim_inputs,
                                **hetoutput_paths, 'D': D_path, 'P_path': P_path, 'Pi_path': Pi_path}) - ss
        else:
            return ImpulseDict({**aggregates, **aggregate_hetoutputs}) - ss

    def impulse_linear(self, ss, exogenous, T=None, Js=None, **kwargs):
        # infer T from exogenous, check that all shocks have same length
        shock_lengths = [x.shape[0] for x in exogenous.values()]
        if shock_lengths[1:] != shock_lengths[:-1]:
            raise ValueError('Not all shocks in kwargs (exogenous) are same length!')
        T = shock_lengths[0]

        return ImpulseDict(self.jacobian(ss, list(exogenous.keys()), T=T, Js=Js, **kwargs).apply(exogenous))

    def jacobian(self, ss, exogenous=None, T=300, outputs=None, Js=None, h=1E-4):
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
        Js : [optional] dict of {str: JacobianDict}}
            supply saved Jacobians

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives T*T Jacobian of o with respect to i
        """
        # The default set of outputs are all outputs of the backward iteration function
        # except for the backward iteration variables themselves
        if exogenous is None:
            exogenous = list(self.inputs)
        if outputs is None:
            outputs = self.non_back_iter_outputs

        relevant_shocks = [i for i in self.back_step_inputs | self.hetinput_inputs if i in exogenous]

        # if we supply Jacobians, use them if possible, warn if they cannot be used
        if Js is not None:
            outputs_cap = [o.capitalize() for o in outputs]
            if verify_saved_jacobian(self.name, Js, outputs_cap, relevant_shocks, T):
                return Js[self.name]

        # step 0: preliminary processing of steady state
        (ssin_dict, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space, D0, D2, Pi, P) = self.jac_prelim(ss)

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        dYs, dDs, dD_ps, dD_direct = {}, {}, {}, {}
        for i in relevant_shocks:
            dYs[i], dDs[i], dD_ps[i], dD_direct[i] = self.backward_iteration_fakenews(i, outputs, ssin_dict, ssout_list,
                                                                                      ss.internal[self.name]['D'],
                                                                                      D0, D2, P, Pi, sspol_i, sspol_pi,
                                                                                      sspol_space, T, h,
                                                                                      ss_for_hetinput)

        # step 2 of fake news algorithm
        # compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = {}
        for o in outputs:
            curlyPs[o] = self.forward_iteration_fakenews(ss.internal[self.name][o], Pi, P, sspol_i, sspol_pi, T)

        # steps 3-4 of fake news algorithm
        # make fake news matrix and Jacobian for each outcome-input pair
        F, J = {}, {}
        for o in outputs:
            for i in relevant_shocks:
                if o.capitalize() not in F:
                    F[o.capitalize()] = {}
                if o.capitalize() not in J:
                    J[o.capitalize()] = {}
                F[o.capitalize()][i] = DiscontBlock.build_F(dYs[i][o], dD_ps[i], curlyPs[o], dD_direct[i], dDs[i])
                J[o.capitalize()][i] = DiscontBlock.J_from_F(F[o.capitalize()][i])

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

            self.internal |= self.hetinput_outputs

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

            self.internal |= self.hetoutput_outputs

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
                                    for k in self.back_iter_vars):
                break

            # update 'old' for comparison during next iteration, prepare 'ssin' as input for next iteration
            old.update({k: sspol[k] for k in self.back_iter_vars})
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

    def dist_ss(self, sspol, grid, tol=1E-10, maxit=100_000, D_seed=None):
        """Find steady-state distribution through forward iteration until convergence.

        Parameters
        ----------
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

        Returns
        ----------
        D : array
            steady-state distribution
        """
        # extract transition matrix for exogenous states
        Pi = [sspol[k] for k in self.exogenous]
        P = sspol[self.disc_policy]

        # first obtain initial distribution D
        if D_seed is None:
            # initialize at uniform distribution
            D = np.ones_like(sspol[self.policy]) / sspol[self.policy].size
        else:
            D = D_seed

        # obtain interpolated policy rule for each dimension of endogenous policy
        sspol_i, sspol_pi = utils.interpolate.interpolate_coord_robust(grid, sspol[self.policy])

        # iterate until convergence by tol, or maxit
        for it in range(maxit):
            Dnew = self.forward_step(D, P, Pi, sspol_i, sspol_pi)

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

    def shock_timing(self, input_shocked, D0, Pi_ss, P, ss_for_hetinput, h):
        """Figure out the details of how the scalar shock feeds into back_step_fun.

        Main complication: shocks to Pi transmit via hetinput with a lead of 1.
        """
        if self.hetinput is not None and input_shocked in self.hetinput_inputs:
            # if input_shocked is an input to hetinput, take numerical diff to get response
            din_dict = dict(zip(self.hetinput_outputs_order,
                                utils.differentiate.numerical_diff_symmetric(self.hetinput, ss_for_hetinput,
                                                                             {input_shocked: 1}, h)))

            if all(k not in din_dict.keys() for k in self.exogenous):
                # if Pi is not generated by hetinput, no work to be done
                lead = 0
                dD3_direct = None
            elif all(np.count_nonzero(din_dict[k]) == 0 for k in self.exogenous if k in din_dict):
                # if Pi is generated by hetinput but input_shocked does not affect it, replace Pi with Pi_p
                lead = 0
                dD3_direct = None
                for k in self.exogenous:
                    if k in din_dict.keys():
                        din_dict[k + '_p'] = din_dict.pop(k)
            else:
                # if Pi is generated by hetinput and input_shocked affects it, replace that with Pi_p at lead 1
                lead = 1
                Pi = [din_dict[k] if k in din_dict else Pi_ss[num] for num, k in enumerate(self.exogenous)]
                dD2_direct = utils.forward_step.forward_step_exo(D0, Pi)
                dD3_direct = utils.forward_step.forward_step_dpol(dD2_direct, P)
                for k in self.exogenous:
                    if k in din_dict.keys():
                        din_dict[k + '_p'] = din_dict.pop(k)
        else:
            # if input_shocked feeds directly into back_step_fun with lead 0, no work to be done
            lead = 0
            din_dict = {input_shocked: 1}
            dD3_direct = None

        return din_dict, lead, dD3_direct

    def backward_step_fakenews(self, din_dict, output_list, ssin_dict, ssout_list,
                               Dss, D2, P, Pi, sspol_i, sspol_pi, sspol_space, h=1E-4):
        # 1. shock perturbs outputs
        shocked_outputs = {k: v for k, v in zip(self.back_step_output_list,
                                                utils.differentiate.numerical_diff(self.back_step_fun,
                                                                                   ssin_dict, din_dict, h,
                                                                                   ssout_list))}
        dV = {k: shocked_outputs[k] for k in self.back_iter_vars}

        # 2. which affects the distribution tomorrow via the savings policy
        pol_pi_shock = -shocked_outputs[self.policy] / sspol_space
        if "delev_exante" in din_dict:
            # include an additional term to account for the effect of a deleveraging shock affecting the grid
            dx = np.zeros_like(sspol_pi)
            dx[sspol_i == 0] = 1.
            add_term = sspol_pi * dx / sspol_space
            pol_pi_shock += add_term
        dD3_p = self.forward_step_shock(Dss, sspol_i, pol_pi_shock, Pi, P)

        # 3. and the distribution today (and Dmid tomorrow) via the discrete choice
        P_shock = shocked_outputs[self.disc_policy]
        dD3 = utils.forward_step.forward_step_dpol(D2, P_shock)          # s[0], z[0], a[-1]

        # 4. and the aggregate outcomes today (ignoring dD and dD_direct)
        dY = {k: np.vdot(Dss, shocked_outputs[k]) for k in output_list}

        return dV, dD3, dD3_p, dY

    def backward_iteration_fakenews(self, input_shocked, output_list, ssin_dict, ssout_list, Dss, D0, D2, P, Pi,
                                    sspol_i, sspol_pi, sspol_space, T, h=1E-4, ss_for_hetinput=None):
        """Iterate policy steps backward T times for a single shock."""
        # map name of shocked input into a perturbation of the inputs of back_step_fun
        din_dict, lead, dD_direct = self.shock_timing(input_shocked, D0, Pi.copy(), P, ss_for_hetinput, h)

        # contemporaneous response to unit scalar shock
        dV, dD, dD_p, dY = self.backward_step_fakenews(din_dict, output_list, ssin_dict, ssout_list,
                                                       Dss, D2, P, Pi, sspol_i, sspol_pi, sspol_space, h=h)

        # infer dimensions from this and initialize empty arrays
        dDs = np.empty((T,) + dD.shape)
        dD_ps = np.empty((T,) + dD_p.shape)
        dYs = {k: np.empty(T) for k in dY.keys()}

        # fill in current effect of shock (be careful to handle lead = 1)
        dDs[:lead, ...], dD_ps[:lead, ...] = 0, 0
        dDs[lead, ...], dD_ps[lead, ...] = dD, dD_p
        for k in dY.keys():
            dYs[k][:lead] = 0
            dYs[k][lead] = dY[k]

        # fill in anticipation effects
        for t in range(lead + 1, T):
            dV, dDs[t, ...], dD_ps[t, ...], dY = self.backward_step_fakenews({k + '_p':
                                                                             v for k, v in dV.items()},
                                                                             output_list, ssin_dict, ssout_list,
                                                                             Dss, D2, P, Pi, sspol_i, sspol_pi,
                                                                             sspol_space, h)

            for k in dY.keys():
                dYs[k][t] = dY[k]

        return dYs, dDs, dD_ps, dD_direct

    def forward_iteration_fakenews(self, o_ss, Pi, P, pol_i_ss, pol_pi_ss, T):
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
                                                                            P, Pi, pol_i_ss, pol_pi_ss))
        return curlyPs

    @staticmethod
    def build_F(dYs, dD_ps, curlyPs, dD_direct, dDs):
        T = dYs.shape[0]
        F = np.empty((T, T))

        # standard effect
        F[0, :] = dYs
        F[1:, :] = curlyPs[:T-1, ...].reshape((T-1, -1)) @ dD_ps.reshape((T, -1)).T

        # contemporaneous effect via discrete choice
        if dDs is not None:
            F += curlyPs.reshape((T, -1)) @ dDs.reshape((T, -1)).T

        # direct effect of shock
        if dD_direct is not None:
            F[:, 0] += curlyPs.reshape((T, -1)) @ dD_direct.ravel()

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
        D0              : array (nS, nZ, nA), distribution over s[-1], z[-1], a[-1]
        ssout_list      : tuple, what self.back_step_fun returns when given ssin_dict (not exactly the same
                            as steady-state numerically since SS convergence was to some tolerance threshold)
        ss_for_hetinput : dict, ss vals of exactly the inputs needed by self.hetinput (if it exists)
        sspol_i         : dict, indices on lower bracketing gridpoint for all in self.policy
        sspol_pi        : dict, weights on lower bracketing gridpoint for all in self.policy
        sspol_space     : dict, space between lower and upper bracketing gridpoints for all in self.policy
        """
        # preliminary a: obtain ss inputs and other info, run once to get baseline for numerical differentiation
        ssin_dict = self.make_inputs(ss)
        ssout_list = self.back_step_fun(**ssin_dict)

        ss_for_hetinput = None
        if self.hetinput is not None:
            ss_for_hetinput = {k: ss[k] for k in self.hetinput_inputs if k in ss}

        # preliminary b: get sparse representations of policy rules and distance between neighboring policy gridpoints
        grid = ss[self.policy + '_grid']
        sspol_i, sspol_pi = utils.interpolate.interpolate_coord_robust(grid, ss.internal[self.name][self.policy])
        sspol_space = grid[sspol_i + 1] - grid[sspol_i]

        # preliminary c: get end-of-period distribution, need it when Pi is shocked
        Pi = [ss.internal[self.name][k] for k in self.exogenous]
        D = ss.internal[self.name]['D']
        D0 = utils.forward_step.forward_step_cpol(D, sspol_i, sspol_pi)
        D2 = utils.forward_step.forward_step_exo(D0, Pi)

        Pss = ss.internal[self.name][self.disc_policy]

        toreturn = (ssin_dict, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space, D0, D2, Pi, Pss)

        return toreturn

    '''Part 6: helper to extract inputs and potentially process them through hetinput'''

    def make_inputs(self, back_step_inputs_dict):
        """Extract from back_step_inputs_dict exactly the inputs needed for self.back_step_fun,
        process stuff through self.hetinput first if it's there.
        """
        input_dict = copy.deepcopy(back_step_inputs_dict)

        # TODO: This make_inputs function needs to be revisited since it creates inputs both for initial steady
        #   state computation as well as for Jacobian/impulse evaluation for HetBlocks,
        #   where in the former the hetinputs and value function have yet to be computed,
        #   whereas in the latter they have already been computed
        #   and hence do not need to be recomputed. There may be room to clean this function up a bit.
        if isinstance(back_step_inputs_dict, SteadyStateDict):
            input_dict = copy.deepcopy(back_step_inputs_dict.toplevel)
            input_dict.update({k: v for k, v in back_step_inputs_dict.internal[self.name].items()})
        else:
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


    def forward_step(self, D3_prev, P, Pi, a_i, a_pi):
        """Update distribution from (s[0], z[0], a[-1]) to (s[1], z[1], a[0])"""
        # update with continuous policy of last period
        D4 = utils.forward_step.forward_step_cpol(D3_prev, a_i, a_pi)

        # update with exogenous shocks today
        D2 = utils.forward_step.forward_step_exo(D4, Pi)

        # update with discrete choice today
        D3 = utils.forward_step.forward_step_dpol(D2, P)
        return D3

    def forward_step_shock(self, D0, pol_i, pol_pi_shock, Pi, P):
        """Forward_step linearized wrt pol_pi."""
        D4 = utils.forward_step.forward_step_cpol_shock(D0, pol_i, pol_pi_shock)
        D2 = utils.forward_step.forward_step_exo(D4, Pi)
        D3 = utils.forward_step.forward_step_dpol(D2, P)
        return D3

    def forward_step_transpose(self, D, P, Pi, a_i, a_pi):
        """Transpose of forward_step."""
        D1 = np.einsum('sza,xsza->xza', D, P)
        D2 = np.einsum('xpa,zp->xza', D1, Pi[1])
        D3 = np.einsum('xza,sx->sza', D2, Pi[0])
        D4 = utils.forward_step.forward_step_cpol_transpose(D3, a_i, a_pi)
        return D4


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
