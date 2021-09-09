import copy
import numpy as np
from typing import Optional, Dict

from .block import Block
from .. import utilities as utils
from ..classes.steady_state_dict import SteadyStateDict
from ..classes.impulse_dict import ImpulseDict
from ..classes.jacobian_dict import JacobianDict
from ..utilities.function import ExtendedFunction, ExtendedParallelFunction
from ..utilities.ordered_set import OrderedSet
from ..utilities.bijection import Bijection
from .support.het_support import ForwardShockableTransition, ExpectationShockableTransition, lottery_1d, lottery_2d, Markov, CombinedTransition, Transition


def het(exogenous, policy, backward, backward_init=None, hetinputs=None, hetoutputs=None):
    def decorator(backward_fun):
        return HetBlock(backward_fun, exogenous, policy, backward, backward_init, hetinputs, hetoutputs)
    return decorator


class HetBlock(Block):
    def __init__(self, backward_fun, exogenous, policy, backward, backward_init=None, hetinputs=None, hetoutputs=None):
        self.backward_fun = ExtendedFunction(backward_fun)
        self.name = self.backward_fun.name
        super().__init__()

        self.exogenous = OrderedSet(utils.misc.make_tuple(exogenous))
        self.policy, self.backward = (OrderedSet(utils.misc.make_tuple(x)) for x in (policy, backward))
        self.non_backward_outputs = self.backward_fun.outputs - self.backward

        self.outputs = OrderedSet([o.upper() for o in self.non_backward_outputs])
        self.M_outputs = Bijection({o: o.upper() for o in self.non_backward_outputs})
        self.inputs = self.backward_fun.inputs - [k + '_p' for k in self.backward]
        self.inputs |= self.exogenous
        self.internal = OrderedSet(['D', 'Dbeg']) | self.exogenous | self.backward_fun.outputs

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
            if pol not in self.backward_fun.outputs:
                raise ValueError(f"Policy '{pol}' not included as output in {self.name}")
            if pol[0].isupper():
                raise ValueError(f"Policy '{pol}' is uppercase in {self.name}, which is not allowed")

        for back in self.backward:
            if back + '_p' not in self.backward_fun.inputs:
                raise ValueError(f"Backward variable '{back}_p' not included as argument in {self.name}")

            if back not in self.backward_fun.outputs:
                raise ValueError(f"Backward variable '{back}' not included as output in {self.name}")

        for out in self.non_backward_outputs:
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
                return f"<HetBlock '{self.name}' with hetinput '{self.hetinputs.name}'" \
                       f" and with hetoutput `{self.hetoutputs.name}'>"
            else:
                return f"<HetBlock '{self.name}' with hetinput '{self.hetinputs.name}'>"
        else:
            return f"<HetBlock '{self.name}'>"

    def _steady_state(self, calibration, backward_tol=1E-8, backward_maxit=5000,
                      forward_tol=1E-10, forward_maxit=100_000):
        ss = self.extract_ss_dict(calibration)
        self.update_with_hetinputs(ss)
        self.initialize_backward(ss)

        ss = self.backward_steady_state(ss, tol=backward_tol, maxit=backward_maxit)
        Dbeg, D = self.forward_steady_state(ss, forward_tol, forward_maxit)
        ss.update({'Dbeg': Dbeg, "D": D})

        self.update_with_hetoutputs(ss)

        # aggregate all outputs other than backward variables on grid, capitalize
        toreturn = self.non_backward_outputs
        if self.hetoutputs is not None:
            toreturn = toreturn | self.hetoutputs.outputs
        aggregates = {o.upper(): np.vdot(D, ss[o]) for o in toreturn}
        ss.update(aggregates)

        return SteadyStateDict({k: ss[k] for k in ss if k not in self.internal},
                               {self.name: {k: ss[k] for k in ss if k in self.internal}})

    def _impulse_nonlinear(self, ssin, inputs, outputs, monotonic=False, returnindividual=False):
        ss = self.extract_ss_dict(ssin)

        # identify individual variable paths we want from backward iteration, then run it
        toreturn = self.non_backward_outputs
        if self.hetoutputs is not None:
            toreturn = toreturn | self.hetoutputs.outputs
        
        individual_paths, exog_path = self.backward_nonlinear(ss, inputs, toreturn)

        # run forward iteration to get path of distribution (both Dbeg - what to do with this? - and D)
        Dbeg_path, D_path = self.forward_nonlinear(ss, individual_paths, exog_path)

        # obtain aggregates of all outputs, made uppercase
        aggregates = {o.upper(): utils.optimized_routines.fast_aggregate(D_path, individual_paths[o])
                      for o in individual_paths}

        # return either this, or also include distributional information
        # TODO: rethink this when dealing with internals, including Dbeg
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
        outputs = self.M_outputs.inv @ outputs

        # step 0: preliminary processing of steady state
        exog = self.make_exog_law_of_motion(ss)
        endog = self.make_endog_law_of_motion(ss)
        differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs = self.jac_backward_prelim(ss, h, exog)
        law_of_motion = CombinedTransition([exog, endog]).forward_shockable(ss['Dbeg'])
        exog_by_output = {k: exog.expectation_shockable(ss[k]) for k in outputs | self.backward}

        # step 1 of fake news algorithm
        # compute curlyY and curlyD (backward iteration) for each input i
        curlyYs, curlyDs = {}, {}
        for i in inputs:
            curlyYs[i], curlyDs[i] = self.backward_fakenews(i, outputs, T, differentiable_backward_fun,
                                                                      differentiable_hetinputs, differentiable_hetoutputs,
                                                                      law_of_motion, exog_by_output)

        # step 2 of fake news algorithm
        # compute expectation vectors curlyE for each outcome o
        curlyPs = {}
        for o in outputs:
            curlyPs[o] = self.expectation_vectors(ss[o], T-1, law_of_motion)

        # steps 3-4 of fake news algorithm
        # make fake news matrix and Jacobian for each outcome-input pair
        F, J = {}, {}
        for o in outputs:
            for i in inputs:
                if o.upper() not in F:
                    F[o.upper()] = {}
                if o.upper() not in J:
                    J[o.upper()] = {}
                F[o.upper()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
                J[o.upper()][i] = HetBlock.J_from_F(F[o.upper()][i])

        return JacobianDict(J, name=self.name, T=T)

    '''Steady-state backward and forward methods'''

    def backward_steady_state(self, ss, tol=1E-8, maxit=5000):
        """Backward iteration to get steady-state policies and other outcomes"""
        ss = ss.copy()
        exog = self.make_exog_law_of_motion(ss)

        old = {}
        for it in range(maxit):
            for k in self.backward:
                ss[k + '_p'] = exog.expectation(ss[k])
                del ss[k]

            ss.update(self.backward_fun(ss))

            if it % 10 == 1 and all(utils.optimized_routines.within_tolerance(ss[k], old[k], tol)
                                    for k in self.policy):
                break

            old.update({k: ss[k] for k in self.policy})
        else:
            raise ValueError(f'No convergence of policy functions after {maxit} backward iterations!')

        for k in self.backward:
            del ss[k + '_p']

        return ss

    def forward_steady_state(self, ss, tol=1E-10, maxit=100_000):
        """Forward iteration to get steady-state distribution"""
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

    '''Nonlinear impulse backward and forward methods'''

    def backward_nonlinear(self, ss, inputs, toreturn):
        T = inputs.T
        individual_paths = {k: np.empty((T,) + ss['D'].shape) for k in toreturn}

        backdict = ss.copy()
        exog = self.make_exog_law_of_motion(backdict)
        exog_path = []

        for t in reversed(range(T)):
            for k in self.backward:
                backdict[k + '_p'] = exog.expectation(backdict[k])
                del backdict[k]

            backdict.update({k: ss[k] + v[t, ...] for k, v in inputs.items()})
            self.update_with_hetinputs(backdict)
            backdict.update(self.backward_fun(backdict))
            self.update_with_hetoutputs(backdict)
 
            for k in individual_paths:
                individual_paths[k][t, ...] = backdict[k]

            exog = self.make_exog_law_of_motion(backdict)

            exog_path.append(exog)
        
        return individual_paths, exog_path[::-1]

    def forward_nonlinear(self, ss, individual_paths, exog_path):
        T = len(exog_path)
        Dbeg = ss['Dbeg']

        Dbeg_path = np.empty((T,) + Dbeg.shape)
        Dbeg_path[0, ...] = Dbeg
        D_path = np.empty_like(Dbeg_path)

        for t in range(T):
            endog = self.make_endog_law_of_motion({**ss, **{k: individual_paths[k][t, ...] for k in self.policy}})

            # now step forward in two, first exogenous this period then endogenous
            D_path[t, ...] = exog_path[t].forward(Dbeg)

            if t < T-1:
                Dbeg = endog.forward(D_path[t, ...])
                Dbeg_path[t+1, ...] = Dbeg # make this optional

        return Dbeg_path, D_path

    '''Jacobian calculation: four parts of fake news algorithm, plus support methods'''

    def backward_fakenews(self, input_shocked, output_list, T, differentiable_backward_fun,
                            differentiable_hetinput, differentiable_hetoutput,
                            law_of_motion: ForwardShockableTransition, exog: Dict[str, ExpectationShockableTransition]):
        """Part 1 of fake news algorithm: calculate curlyY and curlyD in response to fake news shock"""
        # contemporaneous effect of unit scalar shock to input_shocked
        din_dict = {input_shocked: 1}
        if differentiable_hetinput is not None and input_shocked in differentiable_hetinput.inputs:
            din_dict.update(differentiable_hetinput.diff2({input_shocked: 1}))

        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, differentiable_backward_fun,
                                                            differentiable_hetoutput, law_of_motion, exog, True)

        # infer dimensions from this, initialize empty arrays, and fill in contemporaneous effect
        curlyDs = np.empty((T,) + curlyD.shape)
        curlyYs = {k: np.empty(T) for k in curlyY.keys()}

        curlyDs[0, ...] = curlyD
        for k in curlyY.keys():
            curlyYs[k][0] = curlyY[k]

        # fill in anticipation effects of shock up to horizon T
        for t in range(1, T):
            curlyV, curlyDs[t, ...], curlyY = self.backward_step_fakenews({k+'_p': v for k, v in curlyV.items()},
                                                    output_list, differentiable_backward_fun,
                                                    differentiable_hetoutput, law_of_motion, exog)
            for k in curlyY.keys():
                curlyYs[k][t] = curlyY[k]

        return curlyYs, curlyDs

    def expectation_vectors(self, o_ss, T, law_of_motion: Transition):
        """Part 2 of fake news algorithm: calculate expectation vectors curlyE"""
        curlyEs = np.empty((T,) + o_ss.shape)

        # initialize with beginning-of-period expectation of steady-state policy
        curlyEs[0, ...] = utils.misc.demean(law_of_motion[0].expectation(o_ss))
        for t in range(1, T):
            # demean so that curlyEs converge to zero, in theory no effect but better numerically
            curlyEs[t, ...] = utils.misc.demean(law_of_motion.expectation(curlyEs[t-1, ...]))
        return curlyEs

    @staticmethod
    def build_F(curlyYs, curlyDs, curlyEs):
        """Part 3 of fake news algorithm: build fake news matrix from curlyY, curlyD, curlyE"""
        T = curlyDs.shape[0]
        Tpost = curlyEs.shape[0] - T + 2
        F = np.empty((Tpost + T - 1, T))
        F[0, :] = curlyYs
        F[1:, :] = curlyEs.reshape((Tpost + T - 2, -1)) @ curlyDs.reshape((T, -1)).T
        return F

    @staticmethod
    def J_from_F(F):
        """Part 4 of fake news algorithm: recursively build Jacobian from fake news matrix"""
        J = F.copy()
        for t in range(1, J.shape[1]):
            J[1:, t] += J[:-1, t - 1]
        return J

    def backward_step_fakenews(self, din_dict, output_list, differentiable_backward_fun,
                               differentiable_hetoutput, law_of_motion: ForwardShockableTransition,
                               exog: Dict[str, ExpectationShockableTransition], maybe_exog_shock=False):
        """Support for part 1 of fake news algorithm: single backward step in response to shock"""
        Dbeg, D = law_of_motion[0].Dss, law_of_motion[1].Dss
                               
        # shock perturbs outputs
        shocked_outputs = differentiable_backward_fun.diff(din_dict)
        curlyV = {k: law_of_motion[0].expectation(shocked_outputs[k]) for k in self.backward}

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
        if differentiable_hetoutput is not None and (output_list & differentiable_hetoutput.outputs):
            shocked_outputs.update(differentiable_hetoutput.diff({**shocked_outputs, **din_dict}))
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

    def jac_backward_prelim(self, ss, h, exog):
        """Support for part 1 of fake news algorithm: preload differentiable functions"""
        differentiable_hetinputs = None
        if self.hetinputs is not None:
            differentiable_hetinputs = self.hetinputs.differentiable(ss)

        differentiable_hetoutputs = None
        if self.hetoutputs is not None:
            differentiable_hetoutputs = self.hetoutputs.differentiable(ss)

        ss = ss.copy()
        for k in self.backward:
            ss[k + '_p'] = exog.expectation(ss[k])
        differentiable_backward_fun = self.backward_fun.differentiable(ss, h=h)

        return differentiable_backward_fun, differentiable_hetinputs, differentiable_hetoutputs

    '''HetInput and HetOutput options and processing'''

    def process_hetinputs_hetoutputs(self, hetinputs: Optional[ExtendedParallelFunction], hetoutputs: Optional[ExtendedParallelFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        inputs = self.original_inputs.copy()
        outputs = self.original_outputs.copy()
        internal = self.original_internal.copy()

        if hetoutputs is not None:
            inputs |= (hetoutputs.inputs - self.backward_fun.outputs - ['D'])
            outputs |= [o.upper() for o in hetoutputs.outputs]
            self.M_outputs = Bijection({o: o.upper() for o in hetoutputs.outputs}) @ self.original_M_outputs
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

    '''Additional helper functions'''

    def extract_ss_dict(self, ss):
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internal:
                ssnew.update(ss.internal[self.name])
            return ssnew
        else:
            return ss.copy()

    def initialize_backward(self, ss):
        if not all(k in ss for k in self.backward):
            ss.update(self.backward_init(ss))

    def make_exog_law_of_motion(self, d:dict):
        return CombinedTransition([Markov(d[k], i) for i, k in enumerate(self.exogenous)])

    def make_endog_law_of_motion(self, d: dict):
        if len(self.policy) == 1:
            return lottery_1d(d[self.policy[0]], d[self.policy[0] + '_grid'])
        else:
            return lottery_2d(d[self.policy[0]], d[self.policy[1]],
                        d[self.policy[0] + '_grid'], d[self.policy[1] + '_grid'])