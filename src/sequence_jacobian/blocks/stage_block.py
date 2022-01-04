from typing import List, Optional
import numpy as np
import copy

from .block import Block
from .het_block import HetBlock
from ..classes import SteadyStateDict, JacobianDict, ImpulseDict
from ..utilities.ordered_set import OrderedSet
from ..utilities.function import ExtendedFunction, CombinedExtendedFunction
from ..utilities.bijection import Bijection
from ..utilities.optimized_routines import within_tolerance
from .. import utilities as utils
from .support.law_of_motion import LawOfMotion
from .support.stages import Stage


class StageBlock(Block):
    def __init__(self, stages: List[Stage], backward_init=None, hetinputs=None, name=None):
        super().__init__()
        inputs = OrderedSet([])
        outputs = OrderedSet([])
        stages = make_all_into_stages(stages)

        for i, stage in enumerate(stages):
            # external inputs are whatever you don't take from next stage
            inputs |= (stage.inputs - stages[(i+1) % len(stages)].backward_outputs)
            outputs |= stage.report
        
        # TODO: should have internals

        self.constructor_checks(stages, inputs, outputs)
        self.stages = stages
        self.inputs = inputs
        self.outputs = OrderedSet([o.upper() for o in outputs])
        self.M_outputs = Bijection({o: o.upper() for o in outputs})
        self.save_original()

        if name is None:
            name = stages[0].name + "_to_" + stages[-1].name
        self.name = name

        if hetinputs is not None:
            hetinputs = CombinedExtendedFunction(hetinputs)
        self.process_hetinputs(hetinputs, tocopy=False)

        if backward_init is not None:
            backward_init = ExtendedFunction(backward_init)
        self.backward_init = backward_init


    @staticmethod
    def constructor_checks(stages, inputs, outputs):
        # inputs, outputs, and combined backward should not overlap at all
        if not inputs.isdisjoint(outputs):
            raise ValueError(f'inputs and outputs have overlap {inputs & outputs}')
        backward_all = set().union(*(stage.backward_outputs for stage in stages))
        if not inputs.isdisjoint(backward_all):
            raise ValueError(f'Some stage taking another non-immediate-successor stage backward {inputs & backward_all} as input')
        if not outputs.isdisjoint(backward_all):
            raise ValueError(f'Outputs and backward have overlap {outputs & backward_all}')
       
        # 'D', 'law_of_motion' are protected names; outputs should not be upper case
        for stage in stages:
            if stage.name in ['D', 'law_of_motion']:
                raise ValueError(f"Stage '{stage.name}' has invalid name")
            for o in stage.report:
                if o in ['d', 'law_of_motion']:
                    raise ValueError(f"Stages are not allowed to return outputs called 'd' or 'law_of_motion' but stage '{stage.name}' does")
                if o.isupper(): 
                    raise ValueError(f"Stages are not allowed to report upper-case outputs. Stage '{stage.name}' has an output '{o}'")

    def __repr__(self):
        return f"<StageBlock '{self.name}' with stages {[k.name for k in self.stages]}>"

    def _steady_state(self, calibration, backward_tol=1E-9, backward_maxit=5000,
                      forward_tol=1E-10, forward_maxit=100_000):
        ss = self.extract_ss_dict(calibration)
        hetinputs = self.return_hetinputs(ss)
        ss.update(hetinputs)
        self.initialize_backward(ss)

        backward, report, lom = self.backward_steady_state(ss, backward_tol, backward_maxit)

        # get initialized distribution
        try:
            Dinit = ss[self.stages[0].name]['D']
        except KeyError:
            # assume that beginning-of-first-stage distribution is uniform, with
            # same dimensions as ANY backward input to final stage / backward output from first stage
            backward_last = backward[-1]
            backward_example = backward_last[list(backward_last)[0]]
            Dinit = np.full(backward_example.shape, 1/backward_example.size)
        
        D = self.forward_steady_state(Dinit, lom, forward_tol, forward_maxit)
        
        aggregates = {}
        # initialize internals with hetinputs, then add stage-level internals
        internals = hetinputs
        for i, stage in enumerate(self.stages):
            # aggregate everything to report
            for k in stage.report:
                aggregates[k.upper()] = np.vdot(D[i], report[i][k])
            
            # put individual-level report, end-of-stage backward, and beginning-of-stage dist in internals
            internals[stage.name] = {**backward[i], **report[i],
                                     'law_of_motion': lom[i], 'D': D[i]}

        # put all inputs to the block into aggregates
        for k in self.M.inv @ self.inputs:
            aggregates[k] = ss[k]

        return SteadyStateDict(aggregates, {self.name: internals})

    def _impulse_nonlinear(self, ssin, inputs, outputs, ss_initial):
        ss = self.extract_ss_dict(ssin)
        if ss_initial is not None:
            ss[self.stages[0].name]['D'] = ss_initial[self.name][self.stages[0].name]['D']

        # report_path is dict(stage: {output: TxN-dim array})
        # lom_path is list[t][stage] in chronological order
        report_path, lom_path = self.backward_nonlinear(ss, inputs)
        
        # D_path is dict(stage: TxN-dim array)
        D_path = self.forward_nonlinear(ss, lom_path)

        aggregates = {}
        for stage in self.stages:
            for o in stage.report:
                if self.M_outputs @ o in outputs:
                    aggregates[self.M_outputs @ o] = utils.optimized_routines.fast_aggregate(D_path[stage.name], report_path[stage.name][o])

        return ImpulseDict(aggregates, T=inputs.T) - ssin

    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self._jacobian(ss, list(inputs.keys()), outputs, inputs.T).apply(inputs))

    def _jacobian(self, ss, inputs, outputs, T):
        ss = self.extract_ss_dict(ss)
        outputs = self.M_outputs.inv @ outputs
        differentiable_hetinput = self.preliminary_hetinput(ss, h=1E-4)
        backward_data, forward_data, expectations_data = self.preliminary_all_stages(ss)

        # step 1
        curlyYs, curlyDs = {}, {}
        for i in inputs:
            curlyYs[i], curlyDs[i] = self.backward_fakenews(i, outputs, T, backward_data, forward_data, differentiable_hetinput)
        
        # step 2
        curlyEs = {}
        for o in outputs:
            curlyEs[o] = self.expectation_vectors(o, T-1, expectations_data)

        # steps 3-4
        F, J = {}, {}
        for o in outputs:
            for i in inputs:
                if o.upper() not in F:
                    F[o.upper()] = {}
                if o.upper() not in J:
                    J[o.upper()] = {}
                F[o.upper()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyEs[o])
                J[o.upper()][i] = HetBlock.J_from_F(F[o.upper()][i])
        
        return JacobianDict(J, name=self.name, T=T)

    '''Steady-state backward and forward methods'''
    
    def backward_steady_state(self, ss, tol=1E-9, maxit=5000):
        # 'backward' will be dict with backward output of first stage
        # (i.e. input to last stage) from the most recent time iteration
        # initializer for first iteration should be in 'ss'
        
        backward = {k: ss[k] for k in self.stages[0].backward_outputs}

        # iterate until end-of-final-stage backward inputs converge
        for it in range(maxit):
            backward_new = self.backward_step_steady_state(backward, ss)
            if it % 10 == 0 and all(within_tolerance(backward_new[k], backward[k], tol) for k in backward):
                break
            backward = backward_new
        else:
            raise ValueError(f'No convergence after {maxit} backward iterations!')

        # one more iteration to get backward INPUTS, reported outputs, and law of motion for all stages
        return self.backward_step_nonlinear(backward, ss)[:3]

    def backward_step_steady_state(self, backward, inputs):
        """Iterate backward through all stages for a single period, ignoring reported outputs"""
        for stage in reversed(self.stages):
            backward, _ = stage.backward_step_separate({**inputs, **backward})
        return backward

    def backward_step_nonlinear(self, backward, inputs):
        # append backward INPUT to final stage
        backward_all = [backward]
        report_all = []
        lom_all = []
        for stage in reversed(self.stages):
            (backward, report), lom = stage.backward_step_separate({**inputs, **backward}, lawofmotion=True, hetoutputs=True)
            # append backward OUTPUT, reported outputs, and law of motion for each stage, in reverse chronological order
            backward_all.append(backward)
            report_all.append(report)
            lom_all.append(lom)

        # return backward INPUT, report, and lom for each stage, with stages now in chronological order
        # (to get backward inputs, skip first chronological entry of backward_all, which is backward output of first stage,
        # return that entry separately as the fourth output of this function)
        return backward_all[::-1][1:], report_all[::-1], lom_all[::-1], backward_all[-1]
            
    def forward_steady_state(self, D, lom: List[LawOfMotion], tol=1E-10, maxit=100_000):
        """Find steady-state beginning-of-stage distributions for all stages"""
        # iterate until beginning-of-stage distribution for first stage converges
        for it in range(maxit):
            D_new = self.forward_step_steady_state(D, lom)
            if it % 10 == 0 and within_tolerance(D, D_new, tol):
                break
            D = D_new
        else:
            raise ValueError(f'No convergence after {maxit} forward iterations!')

        # one more iteration to get beginning-of-stage in *all* stages
        return self.forward_step_nonlinear(D, lom)[0]

    def forward_step_steady_state(self, D, loms: List[LawOfMotion]):
        """Given beginning-of-first-stage distribution, apply laws of motion in 'loms'
        for each stage to get end-of-final-stage distribution, which is returned"""
        for lom in loms:
            D = lom @ D
        return D

    def forward_step_nonlinear(self, D, loms: List[LawOfMotion]):
        Ds = [D]
        for i, lom in enumerate(loms):
            Ds.append(lom @ Ds[i])
        # return all beginning-of-stage Ds this period, then beginning-of-period next period
        return Ds[:-1], Ds[-1]

    '''Nonlinear backward and forward methods'''

    def backward_nonlinear(self, ss, inputs):
        indict = ss.copy()
        T = inputs.T
        # populate backward with steady-state backward inputs to final stage (stored under final stage in ss dict)
        backward = {k: ss[self.stages[-1].name][k] for k in self.stages[0].backward_outputs}

        # report_path is dict(stage: {output: TxN-dim array})
        report_path = {stage.name: {o: np.empty((T,) + ss[stage.name][o].shape) for o in stage.report} for stage in self.stages}
        lom_path = []

        for t in reversed(range(T)):
            indict.update({k: ss[k] + v[t, ...] for k, v in inputs.items()})
            hetinputs = self.return_hetinputs(indict)
            indict.update(hetinputs)
            
            # get reports and lom from each stage, backward output of first stage (to feed into next iteration)
            _, report, lom, backward = self.backward_step_nonlinear(backward, indict)

            for j, stage in enumerate(self.stages):
                for o in stage.report:  
                    report_path[stage.name][o][t, ...] = report[j][o]

            lom_path.append(lom)

        return report_path, lom_path[::-1]

    def forward_nonlinear(self, ss, lom_path):
        T = len(lom_path)
        Dbeg = ss[self.stages[0].name]['D']
        D_path = {stage.name: np.empty((T,) + ss[stage.name]['D'].shape) for stage in self.stages}

        for t in range(T):
            # iterate forward from beginning-of-first-stage distribution in Dbeg to get
            # (1) beginning-of-stage distributions for all stages (in D)
            # (2) end-of-final-stage distribution, used for next period's beginning-of-first-stage dist (in Dbeg)
            D, Dbeg = self.forward_step_nonlinear(Dbeg, lom_path[t])
            
            for j, stage in enumerate(self.stages):
                D_path[stage.name][t, ...] = D[j]

        return D_path

    '''Jacobian calculation: four parts of fake news algorithm, plus support methods'''

    def backward_fakenews(self, input_shocked, output_list, T, backward_data, forward_data, differentiable_hetinput):
        din_dict = {input_shocked: 1}
        if differentiable_hetinput is not None and input_shocked in differentiable_hetinput.inputs:
            din_dict.update(differentiable_hetinput.diff(din_dict))
        curlyV, curlyD, curlyY = self.backward_step_fakenews(din_dict, output_list, backward_data, forward_data)

        # infer dimensions from this, initialize empty arrays, and fill in contemporaneous effect
        curlyDs = np.empty((T,) + curlyD.shape)
        curlyYs = {k: np.empty(T) for k in curlyY.keys()}

        curlyDs[0, ...] = curlyD
        for k in curlyY.keys():
            curlyYs[k][0] = curlyY[k]

        # fill in anticipation effects of shock up to horizon T
        for t in range(1, T):
            curlyV, curlyDs[t, ...], curlyY = self.backward_step_fakenews(curlyV, output_list, backward_data, forward_data)
            for k in curlyY.keys():
                curlyYs[k][t] = curlyY[k]

        return curlyYs, curlyDs

    def backward_step_fakenews(self, din_dict, output_list, backward_data, forward_data):
        """Given shocks to this period's inputs in 'din_dict', calculate perturbation to
        first-stage backward outputs (curlyV), to final-stage end-of-stage distribution (curlyD),
        and to any aggregate outputs that are in 'output_list' (curlyY)"""

        dback = {}  # perturbations to backward outputs from most recent stage
        dloms = []  # list of perturbations to law of motion from all stages (initially in reverse order)
        curlyY = {} # perturbations to aggregate outputs

        # go backward through stages, pick up shocks to law of motion
        # and also the part of curlyY not coming through the distribution
        for stage, ss, D, lom, precomp, hetoutputs in backward_data:
            din_all = {**din_dict, **dback}
            dout, dlom = stage.backward_step_shock(ss, din_all, precomp)
            dloms.append(dlom)

            dback = {k: dout[k] for k in stage.backward_outputs}
            
            if hetoutputs is not None and output_list & hetoutputs.outputs:
                din_all.update(dout)
                dout.update(hetoutputs.diff(din_all, outputs=output_list & hetoutputs.outputs))

            # if policy is perturbed for k in output_list, add this to curlyY
            # (effect of perturbed distribution is added separately below)
            for k in stage.report:
                if k in output_list:
                    curlyY[k] = np.vdot(D, dout[k])

        curlyV = dback

        # forward through stages, accumulate to find perturbation to D
        dD = None
        for (stage, ss, D, lom), dlom in zip(forward_data, dloms[::-1]):
            # if dD is not None, add consequences for curlyY
            if dD is not None:
                for k in stage.report:
                    if k in output_list:
                        if k in curlyY:
                            curlyY[k] += np.vdot(dD, ss[k])
                        else:
                            curlyY[k] = np.vdot(dD, ss[k])

            # advance the dD to next stage
            if dD is not None:
                dD = lom @ dD
                if dlom is not None:
                    dD += dlom @ D
            elif dlom is not None:
                dD = dlom @ D

        curlyD = dD

        return curlyV, curlyD, curlyY

    def expectation_vectors(self, o, T, expectations_data):
        """Expectation vector giving expected value of output o, from any stage,
        T periods from now, at the beginning of the first stage
        (demeaned for numerical reasons, which doesn't affect product with curlyD)."""
        curlyE0 = self.expectations_beginning_of_period(o, expectations_data)
        curlyEs = np.empty((T,) + curlyE0.shape)
        curlyEs[0] = utils.misc.demean(curlyE0)

        for t in range(1, T):
            curlyEs[t] = utils.misc.demean(
                self.expectation_step_fakenews(curlyEs[t-1], expectations_data))
        return curlyEs

    def expectations_beginning_of_period(self, o, expectations_data):
        """Find expected value of all outputs o, this period, at beginning of first stage"""
        cur_exp = None
        for ss_report, lom_T in expectations_data:
            # if we've already passed variable, take expectations
            if cur_exp is not None:
                cur_exp = lom_T @ cur_exp

            # see if variable this period
            if o in ss_report:
                cur_exp = ss_report[o]
            
        return cur_exp

    def expectation_step_fakenews(self, cur_exp, expectations_data):
        for _, lom_T in expectations_data:
            cur_exp = lom_T @ cur_exp
        return cur_exp

    '''Preliminary processing'''

    def preliminary_all_stages(self, ss):
        """Create lists of tuples with steady-state information for backward, forward, and
        expectations iterations, each list going in the same time direction as the relevant iteration"""
        # TODO: to make code more intelligible, this should be made object-oriented
        backward_data = []
        forward_data = []
        expectations_data = []
        for stage in reversed(self.stages):
            potential_inputs = {**ss[stage.name], **ss}
            input = {k: potential_inputs[k] for k in stage.inputs}
            report = {k: ss[stage.name][k] for k in stage.report}
            D = ss[stage.name]['D']
            lom = ss[stage.name]['law_of_motion']
            precomputed = stage.precompute(input, lom)

            hetoutputs = None
            if stage.hetoutputs is not None:
                hetoutputs_inputs = {k: potential_inputs[k] for k in stage.hetoutputs.inputs}
                hetoutputs = stage.hetoutputs.differentiable(hetoutputs_inputs)

            backward_data.append((stage, input, D, lom, precomputed, hetoutputs))
            forward_data.append((stage, report, D, lom))
            expectations_data.append((report, lom.T))
        return backward_data, forward_data[::-1], expectations_data

    def preliminary_hetinput(self, ss, h):
        differentiable_hetinputs = None
        if self.hetinputs is not None:
            # always use two-sided differentiation for hetinputs
            differentiable_hetinputs = self.hetinputs.differentiable(ss, h, True)
        return differentiable_hetinputs

    '''HetInput and HetOutput options and processing'''

    def extract_ss_dict(self, ss):
        """Flatten ss dict and internals for this block (if present) into one dict,
        but keeping each stage within internals as a subdict"""
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internals:
                ssnew.update(ss.internals[self.name])
            return ssnew
        else:
            return ss.copy()

    def initialize_backward(self, ss):
        """if not all backward outputs of first stage (i.e. backward inputs
        of final stage) are already in dict, call backward_init to generate them"""
        # could generalize to allow backward_init to start us at different stage?
        if not all(k in ss for k in self.stages[0].backward_outputs):
            ss.update(self.backward_init(ss))

    def next_stage(self, i):
        return self.stages[(i+1) % len(self.stages)]

    def process_hetinputs(self, hetinputs: Optional[CombinedExtendedFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        inputs = self.original_inputs.copy()
        #internals = self.original_internals.copy()

        if hetinputs is not None:
            inputs |= hetinputs.inputs
            inputs -= hetinputs.outputs
            #internals |= hetinputs.outputs

        self.inputs = inputs
        #self.internals = internals

        self.hetinputs = hetinputs
        # TODO: fix consequences with remap, as in het_block.py

        return self

    def add_hetinputs(self, functions):
        if self.hetinputs is None:
            return self.process_hetinputs(CombinedExtendedFunction(functions))
        else:
            return self.process_hetinputs(self.hetinputs.add(functions))

    def remove_hetinputs(self, names):
        return self.process_hetinputs(self.hetinputs.remove(names))

    def return_hetinputs(self, d):
        if self.hetinputs is not None:
            return self.hetinputs(d)
        else:
            return {}

    def save_original(self):
        """store "original" copies of these for use whenever we process new hetinputs/hetoutputs"""
        self.original_inputs = self.inputs
        self.original_outputs = self.outputs
        # self.original_internals = self.internals
        self.original_M_outputs = self.M_outputs

    '''Flexible expectation vectors'''

    # TODO: this is wrong; can we make something like this work?
    # def preliminary_expectations(self, ss, loms=None):
    #     """allow for arbitrary loms, not the ones from ss; useful for counterfactuals"""
    #     # loms is Dict[stage.name: lom] in forward order
    #     expectations_data = []
    #     for stage in reversed(self.stages):
    #         report = {k: ss[stage.name][k] for k in stage.report}
    #         if loms is None:
    #             lom = ss[stage.name]['law_of_motion']
    #         else:
    #             lom = loms[stage.name]
    #         expectations_data.append((report, lom.T))
    #     return expectations_data
    

    def expectation_vectors_level(self, o, T, expectations_data):
        curlyE0 = self.expectations_beginning_of_period(o, expectations_data)
        curlyEs = np.empty((T,) + curlyE0.shape)
        curlyEs[0] = curlyE0

        for t in range(1, T):
            curlyEs[t] = self.expectation_step_fakenews(curlyEs[t-1], expectations_data)
        return curlyEs

    def preliminary_expectations(self, ssin):
        ss = self.extract_ss_dict(ssin)
        expectations_data = []
        for stage in reversed(self.stages):
            report = {k: ss[stage.name][k] for k in stage.report}
            lom = ss[stage.name]['law_of_motion']
            expectations_data.append((report, lom.T))
        return expectations_data


def make_all_into_stages(stages: List[Stage]):
    """Given list of 'stages' that can include either actual stages or
    objects with a .make_stage(next_stage_backward) method, turn all into stages.
    
    Since .make_stage() requires the backward outputs from the next stage,
    we need to find an actual stage to start with, which makes this a little harder."""

    # copy since we'll overwrite
    stages = list(stages)

    # find first that is a stage (for now, an endogenous transition) already
    for i, stage in enumerate(stages):
        if isinstance(stage, Stage):
            ifirst = i
            break
    else:
        raise ValueError('No full-fledged stages supplied to constructor.')

    # iterate backward from there, so that everything before ifirst is a stage
    for i in range(ifirst-1, -1, -1):
        if not isinstance(stages[i], Stage):
            stages[i] = stages[i].make_stage(stages[i+1].backward_outputs)
    
    # now iterate backward from the end
    for i in range(len(stages)-1, ifirst, -1):
        if not isinstance(stages[i], Stage):
            stages[i] = stages[i].make_stage(stages[(i+1)%len(stages)].backward_outputs)
    
    return stages

    