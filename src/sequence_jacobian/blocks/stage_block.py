from typing import List
import numpy as np

from ..classes import SteadyStateDict
from ..utilities.ordered_set import OrderedSet
from ..utilities.optimized_routines import within_tolerance
from .support.law_of_motion import LawOfMotion
from .support.stages import Stage

# TODO: make sure there aren't name clashes between variables, 'D', 'law_of_motion', and stage names!

class StageBlock:
    def __init__(self, stages: List[Stage], name=None):
        inputs = OrderedSet([])
        outputs = OrderedSet([])
        stages = make_all_into_stages(stages)

        for i, stage in enumerate(stages):
            # external inputs are whatever you don't take from next stage
            inputs |= (stage.inputs - stages[(i+1) % len(stages)].backward)
            outputs |= stage.report
        
        self.constructor_checks(stages, inputs, outputs)
        self.stages = stages
        self.inputs = inputs
        self.outputs = outputs

        if name is None:
            name = stages[0].name + "_to_" + stages[-1].name
        self.name = name

    @staticmethod
    def constructor_checks(stages, inputs, outputs):
        """some checks: inputs, outputs, and combined backward should not overlap at all"""
        if not inputs.isdisjoint(outputs):
            raise ValueError(f'inputs and outputs have overlap {inputs & outputs}')
        backward_all = set().union(*(stage.backward for stage in stages))
        if not inputs.isdisjoint(backward_all):
            raise ValueError(f'Some stage taking another non-immediate-successor stage backward {inputs & backward_all} as input')
        if not outputs.isdisjoint(backward_all):
            raise ValueError(f'Outputs and backward have overlap {outputs & backward_all}')

    def _steady_state(self, calibration, backward_tol=1E-9, backward_maxit=5000,
                      forward_tol=1E-10, forward_maxit=100_000):
        ss = self.extract_ss_dict(calibration)

        (backward, report), lom = self.backward_steady_state(ss, backward_tol, backward_maxit)

        # get initialized distribution
        try:
            Dinit = ss[self.stages[0].name]['D']
        except KeyError:
            # assume that beginning-of-first-stage distribution is uniform with
            # same dimensions as an arbitrary backward variable from first stage
            backward0 = backward[0]
            backward_example = backward0[list(backward0)[0]]
            Dinit = np.full(backward_example.shape, 1/backward_example.size)
        
        D = self.forward_steady_state(Dinit, lom, forward_tol, forward_maxit)
        
        aggregates = {}
        internals = {}
        for i, stage in enumerate(self.stages):
            # aggregate everything to report
            for k in stage.report:
                aggregates[k.upper()] = np.vdot(D[i], report[i][k])
            
            # put individual-level report, backward, and dist in internals
            internals[stage.name] = {**backward[i], **report[i], 'law_of_motion': lom[i], 'D': D[i]}

        return SteadyStateDict(aggregates, {self.name: internals})

    def backward_steady_state(self, ss, tol=1E-9, maxit=5000):
        # TODO: allow for initializer function!
        backward = {k: ss[k] for k in self.stages[-1].backward}
        for it in range(maxit):
            backward_new = self.backward_step_steady_state(backward, ss)
            if it % 10 == 0 and all(within_tolerance(backward_new[k], backward[k], tol) for k in backward):
                break
            backward = backward_new
        else:
            raise ValueError(f'No convergence after {maxit} backward iterations!')

        # one more iteration to get backward in all stages, report, and law of motion
        return self.backward_step_nonlinear(backward, ss)
            
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

    def _jacobian(self, ss, inputs, outputs, T):
        ss = self.extract_ss_dict(ss)
        precomputed = self.precompute_all_stages(ss)

    def backward_step_steady_state(self, backward, inputs):
        for stage in reversed(self.stages):
            backward, _ = stage.backward_step_separate(backward, inputs)
        return backward

    def backward_step_nonlinear(self, backward, inputs):
        backward_all = []
        report_all = []
        lom_all = []
        for stage in reversed(self.stages):
            (backward, report), lom = stage.backward_step_separate(backward, inputs, lawofmotion=True)
            backward_all.append(backward)
            report_all.append(report)
            lom_all.append(lom)
        return (backward_all[::-1], report_all[::-1]), lom_all[::-1]

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

    def precompute_all_stages(self, ss):
        precomputed = []
        for stage in self.stages:
            ss_inputs = {**ss, **ss[stage.name]}
            precomputed.append(stage.precompute(ss_inputs, ss[stage.name]['law_of_motion']))
        return precomputed

    def extract_ss_dict(self, ss):
        # copied from het_block.py
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internals:
                ssnew.update(ss.internals[self.name])
            return ssnew
        else:
            return ss.copy()

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
            stages[i] = stages[i].make_stage(stages[i+1].backward)
    
    # now iterate backward from the end
    for i in range(len(stages)-1, ifirst, -1):
        if not isinstance(stages[i], Stage):
            stages[i] = stages[i].make_stages(stages[(i+1)%len(stages)].backward)
    
    return stages

    