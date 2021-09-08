import copy
import numpy as np
from typing import Optional, Dict

from .support.impulse import ImpulseDict
from .support.bijection import Bijection
from ..primitives import Block
from .. import utilities as utils
from ..steady_state.classes import SteadyStateDict
from ..jacobian.classes import JacobianDict
from .support.bijection import Bijection
from ..utilities.function import ExtendedFunction, ExtendedParallelFunction
from ..utilities.ordered_set import OrderedSet
from .support.het_support import ForwardShockableTransition, ExpectationShockableTransition, lottery_1d, lottery_2d, Markov, CombinedTransition


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

    def __init__(self, back_step_fun, exogenous, policy, backward, backward_init=None, hetinputs=None, hetoutputs=None):
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

        self.back_step_fun = ExtendedFunction(back_step_fun)

        self.exogenous = OrderedSet(utils.misc.make_tuple(exogenous))
        self.policy, self.back_iter_vars = (OrderedSet(utils.misc.make_tuple(x)) for x in (policy, backward))
        self.non_back_iter_outputs = self.back_step_fun.outputs - self.back_iter_vars

        self.outputs = OrderedSet([o.capitalize() for o in self.non_back_iter_outputs])
        self.M_outputs = Bijection({o: o.capitalize() for o in self.non_back_iter_outputs})
        self.inputs = self.back_step_fun.inputs - [k + '_p' for k in self.back_iter_vars]
        self.inputs |= self.exogenous
        self.internal = OrderedSet(['D', 'Dbeg']) | self.exogenous | self.back_step_fun.outputs

        # store "original" copies of these for use whenever we process new hetinputs/hetoutputs
        self.original_inputs = self.inputs
        self.original_outputs = self.outputs
        self.original_internal = self.internal
        self.original_M_outputs = self.M_outputs

        # A HetBlock can have heterogeneous inputs and heterogeneous outputs, henceforth `hetinput` and `hetoutput`.
        # See docstring for methods `add_hetinput` and `add_hetoutput` for more details.
        self.hetinputs = hetinputs
        self.hetoutputs = hetoutputs
        if hetinputs is not None or hetoutputs is not None:
            self.process_hetinputs_hetoutputs(hetinputs, hetoutputs, tocopy=False)

        if len(self.policy) > 2:
            raise ValueError(f"More than two endogenous policies in {self.name}, not yet supported")

        # Checking that the various inputs/outputs attributes are correctly set
        for pol in self.policy:
            if pol not in self.back_step_fun.outputs:
                raise ValueError(f"Policy '{pol}' not included as output in {self.name}")
            if pol[0].isupper():
                raise ValueError(f"Policy '{pol}' is uppercase in {self.name}, which is not allowed")

        for back in self.back_iter_vars:
            if back + '_p' not in self.back_step_fun.inputs:
                raise ValueError(f"Backward variable '{back}_p' not included as argument in {self.name}")

            if back not in self.back_step_fun.outputs:
                raise ValueError(f"Backward variable '{back}' not included as output in {self.name}")

        for out in self.non_back_iter_outputs:
            if out[0].isupper():
                raise ValueError("Output '{out}' is uppercase in {self.name}, which is not allowed")

        if backward_init is not None:
            backward_init = ExtendedFunction(backward_init)
        self.backward_init = backward_init

        # note: should do more input checking to ensure certain choices not made: 'D' not input, etc.

    def __repr__(self):
        """Nice string representation of HetBlock for printing to console"""
        if self.hetinputs is not None:
            if self.hetoutputs is not None:
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
        ss = self.extract_ss_dict(calibration)
        self.update_with_hetinputs(ss)
        self.initialize_backward(ss)

        # run backward iteration
        ss = self.policy_ss(ss, tol=backward_tol, maxit=backward_maxit)

        # run forward iteration
        Dbeg, D = self.dist_ss(ss, forward_tol, forward_maxit)
        ss.update({'Dbeg': Dbeg, "D": D})

        self.update_with_hetoutputs(ss)

        # aggregate all outputs other than backward variables on grid, capitalize
        toreturn = self.non_back_iter_outputs
        if self.hetoutputs is not None:
            toreturn = toreturn | self.hetoutputs.outputs
        aggregates = {o.capitalize(): np.vdot(D, ss[o]) for o in toreturn}
        ss.update(aggregates)

        return SteadyStateDict({k: ss[k] for k in ss if k not in self.internal},
                               {self.name: {k: ss[k] for k in ss if k in self.internal}})

    def _impulse_nonlinear(self, ssin, inputs, outputs, monotonic=False, returnindividual=False):
        ss = self.extract_ss_dict(ssin)
        Dbeg = ss['Dbeg']
        T = inputs.T

        # for now not allowing shocks to exog, just trying to get code to work!
        exog = self.make_exog_law_of_motion(ss)

        # allocate empty arrays to store result, assume all like D
        toreturn = self.non_back_iter_outputs
        if self.hetoutputs is not None:
            toreturn = toreturn | self.hetoutputs.outputs
        individual_paths = {k: np.empty((T,) + Dbeg.shape) for k in toreturn}

        # backward iteration
        backdict = ss.copy()

        for t in reversed(range(T)):
            for k in self.back_iter_vars:
                backdict[k + '_p'] = exog.expectation(backdict[k])
                del backdict[k]

            # be careful: if you include vars from self.back_iter_vars in exogenous, agents will use them!
            backdict.update({k: ss[k] + v[t, ...] for k, v in inputs.items()})
            self.update_with_hetinputs(backdict)
            backdict.update(self.back_step_fun(backdict))
            backdict.update({k: backdict[k] for k in self.back_iter_vars})

            self.update_with_hetoutputs(backdict)
 
            for k in individual_paths:
                individual_paths[k][t, ...] = backdict[k]

        Dbeg_path = np.empty((T,) + Dbeg.shape)
        Dbeg_path[0, ...] = Dbeg
        D_path = np.empty((T,) + Dbeg.shape)

        for t in range(T):
            # assemble dict for this period's law of motion and make law of motion object
            d = {k: individual_paths[k][t, ...] for k in self.policy}
            d.update({k + '_grid': ss[k + '_grid'] for k in self.policy})
            d.update({k: ss[k] for k in self.exogenous})
            exog = self.make_exog_law_of_motion(d)
            endog = self.make_endog_law_of_motion(d)

            # now step forward in two, first exogenous this period then endogenous
            D_path[t, ...] = exog.forward(Dbeg)

            if t < T-1:
                Dbeg = endog.forward(D_path[t, ...])
                Dbeg_path[t+1, ...] = Dbeg # make this optional

        # obtain aggregates of all outputs, made uppercase
        aggregates = {o.capitalize(): utils.optimized_routines.fast_aggregate(D_path, individual_paths[o])
                      for o in individual_paths}

        # return either this, or also include distributional information
        # TODO: rethink this
        if returnindividual:
            return ImpulseDict({**aggregates, **individual_paths, 'D': D_path}) - ssin
        else:
            return ImpulseDict(aggregates)[outputs] - ssin


    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))


    def _jacobian(self, ss, inputs, outputs, T, h=1E-4):
        # TODO: h is unusable for now, figure out how to suggest options
        ss = self.extract_ss_dict(ss)
        self.update_with_hetinputs(ss)
        outputs = self.M_outputs.inv @ outputs # horrible

        # step 0: preliminary processing of steady state
        exog = self.make_exog_law_of_motion(ss)
        endog = self.make_endog_law_of_motion(ss)
        differentiable_back_step_fun, differentiable_hetinputs, differentiable_hetoutputs = self.jac_backward_prelim(ss, h, exog)
        law_of_motion = CombinedTransition([exog, endog]).forward_shockable(ss['Dbeg'])
        exog_by_output = {k: exog.expectation_shockable(ss[k]) for k in outputs | self.back_iter_vars}

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = {}, {}
        for i in inputs:
            curlyYs[i], curlyDs[i] = self.backward_iteration_fakenews(i, outputs, T, differentiable_back_step_fun,
                                                                      differentiable_hetinputs, differentiable_hetoutputs,
                                                                      law_of_motion, exog_by_output)

        # step 2 of fake news algorithm
        # compute prediction vectors curlyP (forward iteration) for each outcome o
        curlyPs = {}
        for o in outputs:
            curlyPs[o] = self.forward_iteration_fakenews(ss[o], T-1, law_of_motion)

        # steps 3-4 of fake news algorithm
        # make fake news matrix and Jacobian for each outcome-input pair
        F, J = {}, {}
        for o in outputs:
            for i in inputs:
                if o.capitalize() not in F:
                    F[o.capitalize()] = {}
                if o.capitalize() not in J:
                    J[o.capitalize()] = {}
                F[o.capitalize()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.capitalize()][i] = HetBlock.J_from_F(F[o.capitalize()][i])

        return JacobianDict(J, name=self.name, T=T)

    """HetInput and HetOutput processing"""

    def process_hetinputs_hetoutputs(self, hetinputs: Optional[ExtendedParallelFunction], hetoutputs: Optional[ExtendedParallelFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        inputs = self.original_inputs.copy()
        outputs = self.original_outputs.copy()
        internal = self.original_internal.copy()

        if hetoutputs is not None:
            inputs |= (hetoutputs.inputs - self.back_step_fun.outputs - ['D'])
            outputs |= [o.capitalize() for o in hetoutputs.outputs]
            self.M_outputs = Bijection({o: o.capitalize() for o in hetoutputs.outputs}) @ self.original_M_outputs
            internal |= hetoutputs.outputs

        if hetinputs is not None:
            inputs |= hetinputs.inputs
            inputs -= hetinputs.outputs
            internal |= hetinputs.outputs

        self.inputs = inputs
        self.outputs = outputs
        self.internal = internal

        self.hetinputs = hetinputs
        self.hetoutputs = hetoutputs

        return self

    def add_hetinputs(self, functions):
        if self.hetinputs is None:
            return self.process_hetinputs_hetoutputs(ExtendedParallelFunction(functions), self.hetoutputs)
        else:
            return self.process_hetinputs_hetoutputs(self.hetinputs.add(functions), self.hetoutputs)

    def remove_hetinputs(self, names):
        return self.process_hetinputs_hetoutputs(self.hetinputs.remove(names), self.hetoutputs)

    def add_hetoutputs(self, functions):
        if self.hetoutputs is None:
            return self.process_hetinputs_hetoutputs(self.hetinputs, ExtendedParallelFunction(functions))
        else:
            return self.process_hetinputs_hetoutputs(self.hetinputs, self.hetoutputs.add(functions))

    def remove_hetoutputs(self, names):
        return self.process_hetinputs_hetoutputs(self.hetinputs, self.hetoutputs.remove(names))

    def update_with_hetinputs(self, d):
        if self.hetinputs is not None:
            d.update(self.hetinputs(d))

    def update_with_hetoutputs(self, d):
        if self.hetoutputs is not None:
            d.update(self.hetoutputs(d))

    '''Part 3: components of ss():
        - policy_ss : backward iteration to get steady-state policies and other outcomes
        - dist_ss   : forward iteration to get steady-state distribution and compute aggregates
    '''

    def policy_ss(self, ss, tol=1E-8, maxit=5000):
        ss = ss.copy()
        exog = self.make_exog_law_of_motion(ss)

        old = {}
        for it in range(maxit):
            for k in self.back_iter_vars:
                ss[k + '_p'] = exog.expectation(ss[k])
                del ss[k]

            ss.update(self.back_step_fun(ss))

            if it % 10 == 1 and all(utils.optimized_routines.within_tolerance(ss[k], old[k], tol)
                                    for k in self.policy):
                break

            old.update({k: ss[k] for k in self.policy})
        else:
            raise ValueError(f'No convergence of policy functions after {maxit} backward iterations!')

        for k in self.back_iter_vars:
            del ss[k + '_p']

        return ss

    def dist_ss(self, ss, tol=1E-10, maxit=100_000):
        exog = self.make_exog_law_of_motion(ss)
        endog = self.make_endog_law_of_motion(ss)
        
        Dbeg_seed = ss.get('Dbeg', None)
        pi_seeds = [ss.get(k + '_seed', None) for k in self.exogenous]

        # first obtain initial distribution D
        if Dbeg_seed is None:
            # stationary distribution of each exogenous
            pis = [exog[i].stationary(pi_seed) for i, pi_seed in enumerate(pi_seeds)]

            # uniform distribution over endogenous
            endog_uniform = [np.full(len(ss[k+'_grid']), 1/len(ss[k+'_grid'])) for k in self.policy]

            # initialize outer product of all these as guess
            Dbeg = utils.multidim.outer(pis + endog_uniform)
        else:
            Dbeg = Dbeg_seed

        # iterate until convergence by tol, or maxit
        D = exog.forward(Dbeg)
        for it in range(maxit):
            Dbeg_new = endog.forward(D)
            D_new = exog.forward(Dbeg_new)

            # only check convergence every 10 iterations for efficiency
            if it % 10 == 0 and utils.optimized_routines.within_tolerance(Dbeg, Dbeg_new, tol):
                break
            Dbeg = Dbeg_new
            D = D_new
        else:
            raise ValueError(f'No convergence after {maxit} forward iterations!')

        # "D" is after the exogenous shock, Dbeg is before it
        return Dbeg, D

    '''Part 4: components of jac(), corresponding to *4 steps of fake news algorithm* in paper
        - Step 1: backward_step_fakenews and backward_iteration_fakenews to get curlyYs and curlyDs
        - Step 2: forward_iteration_fakenews to get curlyPs
        - Step 3: build_F to get fake news matrix from curlyYs, curlyDs, curlyPs
        - Step 4: J_from_F to get Jacobian from fake news matrix
    '''

    def backward_step_fakenews(self, din_dict, output_list, differentiable_back_step_fun,
                               differentiable_hetoutput, law_of_motion: ForwardShockableTransition,
                               exog: Dict[str, ExpectationShockableTransition], maybe_exog_shock=False):

        Dbeg, D = law_of_motion[0].Dss, law_of_motion[1].Dss
                               
        # shock perturbs outputs
        shocked_outputs = differentiable_back_step_fun.diff(din_dict)
        curlyV = {k: law_of_motion[0].expectation(shocked_outputs[k]) for k in self.back_iter_vars}

        # if there might be a shock to exogenous processes, figure out what it is
        if maybe_exog_shock:
            shocks_to_exog = [din_dict.get(k, None) for k in self.exogenous]
        else:
            shocks_to_exog = None

        # perturbation to exog and outputs outputs affects distribution tomorrow
        policy_shock = [shocked_outputs[k] for k in self.policy]
        if len(policy_shock) == 1:
            policy_shock = policy_shock[0]
        curlyD = law_of_motion.forward_shock([shocks_to_exog, policy_shock])

        # and also affect aggregate outcomes today
        # TODO: seems wrong if hetoutput can take in hetinputs, since those might separately change it?
        if differentiable_hetoutput is not None and (output_list & differentiable_hetoutput.outputs):
            shocked_outputs.update(differentiable_hetoutput.diff(shocked_outputs))
        curlyY = {k: np.vdot(D, shocked_outputs[k]) for k in output_list}

        # add effects from perturbation to exog on beginning-of-period expectations in curlyV and curlyY
        if maybe_exog_shock:
            for k in curlyV:
                shock = exog[k].expectation_shock(shocks_to_exog)
                if shock is not None:
                    curlyV[k] += shock
            
            for k in curlyY:
                shock = exog[k].expectation_shock(shocks_to_exog)
                # maybe could be more efficient since we don't need to calculate pointwise?
                if shock is not None:
                    curlyY[k] += np.vdot(Dbeg, shock)

        return curlyV, curlyD, curlyY

    def backward_iteration_fakenews(self, input_shocked, output_list, T, differentiable_back_step_fun,
                            differentiable_hetinput, differentiable_hetoutput,
                            law_of_motion: ForwardShockableTransition, exog: Dict[str, ExpectationShockableTransition]):

        if differentiable_hetinput is not None and input_shocked in differentiable_hetinput.inputs:
            # if input_shocked is an input to hetinput, take numerical diff to get response
            din_dict = differentiable_hetinput.diff2({input_shocked: 1})
        else:
            # otherwise, we just have that one shock
            din_dict = {input_shocked: 1}

        # contemporaneous response to unit scalar shock
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, differentiable_back_step_fun, differentiable_hetoutput,
                                                             law_of_motion, exog, True)

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
                                                    output_list, differentiable_back_step_fun, differentiable_hetoutput,
                                                    law_of_motion, exog)
            for k in curlyY.keys():
                curlyYs[k][t] = curlyY[k]

        return curlyYs, curlyDs

    def forward_iteration_fakenews(self, o_ss, T, law_of_motion: ForwardShockableTransition):
        """Iterate transpose forward T steps to get full set of curlyEs for a given outcome."""
        curlyEs = np.empty((T,) + o_ss.shape)

        # initialize with beginning-of-period expectation of policy
        curlyEs[0, ...] = utils.misc.demean(law_of_motion[0].expectation(o_ss))
        for t in range(1, T):
            # we demean so that curlyEs converge to zero (better numerically), in theory no effect
            curlyEs[t, ...] = utils.misc.demean(law_of_motion.expectation(curlyEs[t-1, ...]))
        return curlyEs

    @staticmethod
    def build_F(curlyYs, curlyDs, curlyEs):
        T = curlyDs.shape[0]
        Tpost = curlyEs.shape[0] - T + 2
        F = np.empty((Tpost + T - 1, T))
        F[0, :] = curlyYs
        F[1:, :] = curlyEs.reshape((Tpost + T - 2, -1)) @ curlyDs.reshape((T, -1)).T
        return F

    @staticmethod
    def J_from_F(F):
        J = F.copy()
        for t in range(1, J.shape[1]):
            J[1:, t] += J[:-1, t - 1]
        return J

    '''Part 5: helpers for .jac and .ajac: preliminary processing'''

    def jac_backward_prelim(self, ss, h, exog):
        differentiable_hetinputs = None
        if self.hetinputs is not None:
            differentiable_hetinputs = self.hetinputs.differentiable(ss)

        differentiable_hetoutputs = None
        if self.hetoutputs is not None:
            differentiable_hetoutputs = self.hetoutputs.differentiable(ss)

        ss = ss.copy()
        for k in self.back_iter_vars:
            ss[k + '_p'] = exog.expectation(ss[k])
        differentiable_back_step_fun = self.back_step_fun.differentiable(ss, h=h)

        return differentiable_back_step_fun, differentiable_hetinputs, differentiable_hetoutputs

    '''Part 6: helper to extract inputs and potentially process them through hetinput'''

    def extract_ss_dict(self, ss):
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internal:
                ssnew.update(ss.internal[self.name])
            return ssnew
        else:
            return ss.copy()

    def initialize_backward(self, ss):
        if not all(k in ss for k in self.back_iter_vars):
            ss.update(self.backward_init(ss))

    # def make_inputs(self, back_step_inputs_dict):
    #     """Extract from back_step_inputs_dict exactly the inputs needed for self.back_step_fun."""
        

    #     if not all(k in input_dict for k in self.back_iter_vars):
    #         input_dict.update(self.backward_init(input_dict))

    #     for i_p in self.inputs_to_be_primed:
    #         input_dict[i_p + "_p"] = input_dict[i_p]
    #         del input_dict[i_p]

    #     try:
    #         return {k: input_dict[k] for k in self.back_step_fun.inputs if k in input_dict}
    #     except KeyError as e:
    #         print(f'Missing backward variable or Markov matrix {e} for {self.self.name}!')
    #         raise

    def make_exog_law_of_motion(self, d:dict):
        return CombinedTransition([Markov(d[k], i) for i, k in enumerate(self.exogenous)])

    def make_endog_law_of_motion(self, d: dict):
        if len(self.policy) == 1:
            return lottery_1d(d[self.policy[0]], d[self.policy[0] + '_grid'])
        else:
            return lottery_2d(d[self.policy[0]], d[self.policy[1]],
                        d[self.policy[0] + '_grid'], d[self.policy[1] + '_grid'])

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

