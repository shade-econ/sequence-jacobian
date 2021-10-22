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
            # assume that beginning-of-first-stage distribution is uniform with
            # same dimensions as end-of-period final stage
            backward_last = backward[-1]
            backward_example = backward_last[list(backward_last)[0]]
            Dinit = np.full(backward_example.shape, 1/backward_example.size)
        
        D = self.forward_steady_state(Dinit, lom, forward_tol, forward_maxit)
        
        aggregates = {}
        # initialize internals with hetinputs, then add stage-level
        internals = hetinputs
        for i, stage in enumerate(self.stages):
            # aggregate everything to report
            for k in stage.report:
                aggregates[k.upper()] = np.vdot(D[i], report[i][k])
            
            # put individual-level report, backward, and dist in internals
            internals[stage.name] = {**backward[i], **report[i],
                                     'law_of_motion': lom[i], 'D': D[i]}

        # put all inputs to the block into aggregates
        for k in self.inputs:
            aggregates[k] = ss[k]

        return SteadyStateDict(aggregates, {self.name: internals})

    def _impulse_nonlinear(self, ssin, inputs, outputs, ss_initial):
        ss = self.extract_ss_dict(ssin)
        if ss_initial is not None:
            ss[self.stages[0].name]['D'] = ss_initial[self.stages[0].name]['D']

        # report_path is dict(stage: {output: TxN-dim array})
        # lom_path is list[t][stage] in forward order
        report_path, lom_path = self.backward_nonlinear(ss, inputs)
        
        # D_path is dict(stage: TxN-dim array)
        D_path = self.forward_nonlinear(ss, lom_path)

        aggregates = {}
        for stage in self.stages:
            for o in stage.report:
                if self.M_outputs @ o in outputs:
                    aggregates[self.M_outputs @ o] = utils.optimized_routines.fast_aggregate(D_path[stage.name], report_path[stage.name][o])

        return ImpulseDict(aggregates, T=inputs.T) - ssin

    def _impulse_linear(self, ss, inputs, outputs, Js, h=1E-4, twosided=False):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js, h=h, twosided=twosided).apply(inputs))

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
        backward = {k: ss[k] for k in self.stages[-1].backward_outputs}
        for it in range(maxit):
            backward_new = self.backward_step_steady_state(backward, ss)
            if it % 10 == 0 and all(within_tolerance(backward_new[k], backward[k], tol) for k in backward):
                break
            backward = backward_new
        else:
            raise ValueError(f'No convergence after {maxit} backward iterations!')

        # one more iteration to get backward in all stages, report, and law of motion
        return self.backward_step_nonlinear(backward, ss)[:3]

    def backward_step_steady_state(self, backward, inputs):
        for stage in reversed(self.stages):
            backward, _ = stage.backward_step_separate(backward, inputs)
        return backward

    def backward_step_nonlinear(self, backward, inputs):
        backward_all = [backward]
        report_all = []
        lom_all = []
        for stage in reversed(self.stages):
            (backward, report), lom = stage.backward_step_separate(backward, inputs, lawofmotion=True)
            backward_all.append(backward)
            report_all.append(report)
            lom_all.append(lom)
        # return end-of-stage backward, report, and lom for each stage
        # and also the final beginning-of-stage backward (i.e. end-of-stage previous period)
        return backward_all[::-1][1:], report_all[::-1], lom_all[::-1], backward_all[-1]
            
    def forward_steady_state(self, D, lom: List[LawOfMotion], tol=1E-10, maxit=100_000):
        for it in range(maxit):
            D_new = self.forward_step_steady_state(D, lom)
            if it % 10 == 0 and within_tolerance(D, D_new, tol):
                break
            D = D_new
        else:
            raise ValueError(f'No convergence after {maxit} forward iterations!')

        # one more iteration to get beginning-of-stage in all stages
        return self.forward_step_nonlinear(D, lom)[0]

    def forward_step_steady_state(self, D, loms: List[LawOfMotion]):
        for lom in loms:
            D = lom @ D
        return D

    def forward_step_nonlinear(self, D, loms: List[LawOfMotion]):
        Ds = [D]
        for i, lom in enumerate(loms):
            Ds.append(lom @ Ds[i-1])
        # return all beginning-of-stage Ds this period, then beginning-of-period next period
        return Ds[:-1], Ds[-1]

    '''Nonlinear backward and forward methods'''

    def backward_nonlinear(self, ss, inputs):
        indict = ss.copy()
        T = inputs.T
        backward = {k: ss[self.stages[-1].name][k] for k in self.stages[-1].backward_outputs}

        # report_path is dict(stage: {output: TxN-dim array})
        report_path = {stage.name: {o: np.empty((T,) + ss[stage.name][o].shape) for o in stage.report} for stage in self.stages}
        lom_path = []

        for t in reversed(range(T)):
            indict.update({k: ss[k] + v[t, ...] for k, v in inputs.items()})
            hetinputs = self.return_hetinputs(indict)
            indict.update(hetinputs)
            
            _, report, lom, backward = self.backward_step_nonlinear(backward, indict)

            for j, stage in enumerate(self.stages):
                for o in stage.report:  
                    report_path[stage.name][o][t, ...] = report[j][o]

            lom_path.append(lom)

        return report_path, lom_path[::-1]

# TODO: this does not line up
    def forward_nonlinear(self, ss, lom_path):
        T = len(lom_path)
        Dbeg = ss[self.stages[0].name]['D']
        D_path = {stage.name: np.empty((T,) + ss[stage.name]['D'].shape) for stage in self.stages}

        for t in range(T):
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
        dback = {}
        dloms = []
        curlyY = {}

        # backward through stages, pick up shocks to law of motion
        # and also the part of curlyY not coming through the distribution
        for stage, ss, D, lom, precomp in backward_data:
            dout, dlom = stage.backward_step_shock(ss, {**din_dict, **dback}, precomp)
            dloms.append(dlom)

            dback = {k: dout[k] for k in stage.backward_outputs}
            
            for k in stage.report:
                if k in output_list:
                    curlyY[k] = np.vdot(D, dout[k])

        curlyV = dback

        # forward through stages, find shock to D
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
        curlyE0 = self.expectations_beginning_of_period(o, expectations_data)
        curlyEs = np.empty((T,) + curlyE0.shape)
        curlyEs[0] = utils.misc.demean(curlyE0)

        for t in range(1, T):
            curlyEs[t] = utils.misc.demean(
                self.expectation_step_fakenews(curlyEs[t-1], expectations_data))
        return curlyEs

    def expectations_beginning_of_period(self, o, expectations_data):
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

    def preliminary_all_stages(self, ss):
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
            backward_data.append((stage, input, D, lom, precomputed))
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
        # copied from het_block.py
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internals:
                ssnew.update(ss.internals[self.name])
            return ssnew
        else:
            return ss.copy()

    def initialize_backward(self, ss):
        # always initialize the backward outputs of first stage,
        # backward inputs of final stage (maybe generalize later?!)
        if not all(k in ss for k in self.stages[0].backward_outputs):
            ss.update(self.backward_init(ss))

    def next_stage(self, i):
        return self.stages[(i+1) % len(self.stages)]

    def process_hetinputs(self, hetinputs: Optional[CombinedExtendedFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        inputs = self.original_inputs.copy()
        outputs = self.original_outputs.copy()
        #internals = self.original_internals.copy()

        if hetinputs is not None:
            inputs |= hetinputs.inputs
            inputs -= hetinputs.outputs
            #internals |= hetinputs.outputs

        self.inputs = inputs
        self.outputs = outputs
        #self.internals = internals

        self.hetinputs = hetinputs

        return self

    def add_hetinputs(self, functions):
        if self.hetinputs is None:
            return self.process_hetinputs(CombinedExtendedFunction(functions))
        else:
            return self.process_hetinputs(self.hetinputs.add(functions))

    def remove_hetinputs(self, names):
        return self.process_hetinputs_hetoutputs(self.hetinputs.remove(names))

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

def make_all_into_stages(stages: List[Stage]):
    """Given list of 'stages' that can include either actual stages or
    objects with a .make_stage(next_period_backward) method, turn all into stages."""
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
            stages[i] = stages[i].make_stages(stages[(i+1)%len(stages)].backward_outputs)
    
    return stages

    