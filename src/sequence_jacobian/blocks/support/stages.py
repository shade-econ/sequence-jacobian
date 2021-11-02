import numpy as np
from sequence_jacobian.blocks.support.het_support import DiscreteChoice
from ...utilities.function import ExtendedFunction
from ...utilities.ordered_set import OrderedSet
from ...utilities.misc import make_tuple, logit_choice
from .law_of_motion import (lottery_1d, ShockedPolicyLottery1D,
                            lottery_2d, ShockedPolicyLottery2D,
                            Markov)

class Stage:
    def backward_step(self, inputs, lawofmotion=False):
        pass

    def backward_step_shock(self, ss, shocks, precomputed=None):
        pass

    def precompute(self, ss, ss_lawofmotion=None):
        pass

    def backward_step_separate(self, backward_inputs, other_inputs, lawofmotion=False):
        """Wrapper around backward_step that takes in backward and other inputs
        separately and also returns backward and report separately"""
        outputs = self.backward_step({**other_inputs, **backward_inputs}, lawofmotion)
        if lawofmotion:
            outputs, lom = outputs
        
        backward_outputs = {k: outputs[k] for k in self.backward_outputs}
        report = {k: outputs[k] for k in self.report}

        if lawofmotion:
            return (backward_outputs, report), lom
        else:
            return backward_outputs, report

    def __init__(self):
        self.name = ""
        self.backward_outputs = OrderedSet([])
        self.report = OrderedSet([])
        self.inputs = OrderedSet([])


class LogitChoice(Stage):
    """Stage that does endogenous discrete choice with type 1 extreme value taste shocks"""
    def __init__(self, value, backward, index, taste_shock_scale, f=None, name=None):
        # flow utility function, if present, should return a single output
        if f is not None:
            f = ExtendedFunction(f)
            if not len(f.outputs) == 1:
                raise ValueError(f'Flow utility function {f.name} returning multiple outputs {f.outputs}')
            self.f = f
        else:
            self.f = None

        # other subclass-specific attributes
        self.index = index
        self.value = value
        self.backward = backward
        self.taste_shock_scale = taste_shock_scale

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward)) | [value]
        self.report = OrderedSet([])
        self.inputs = OrderedSet(make_tuple(backward)) | [value, taste_shock_scale]
        if f is not None:
            self.inputs |= f.inputs

    def __repr__(self):
        return f"<Stage-Discrete '{self.name}'>"

    def backward_step(self, inputs, lawofmotion=False):
        # start with value we're given
        V_next = inputs[self.value]

        # add dimension at beginning to allow for choice, then swap (today's choice determines next stages's state)
        V = V_next[np.newaxis, ...]
        V = np.swapaxes(V, 0, self.index+1)

        # call f if we have it to get flow utility
        if self.f is not None:
            flow_u = self.f(inputs)
        else:
            nchoice = V.shape[0]
            flow_u = np.zeros((nchoice,) + V_next.shape)

        V = flow_u + V
        
        # calculate choice probabilities and expected value
        P, EV = logit_choice(V, inputs[self.taste_shock_scale])
        
        # make law of motion, use it to take expectations of everything else
        lom = DiscreteChoice(P, self.index)

        # take expectations
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV

        if not lawofmotion:
            return outputs
        else:
            return outputs, lom

    def backward_step_shock(self, ss, shocks, precomputed):
        """See 'discrete choice math' note for background. Note that scale is inverse of 'c' in that note."""
        f, lom = precomputed

        # this part parallel to backward_step, just with derivatives...
        dV_next = shocks[self.value]
        dV = dV_next[np.newaxis, ...]
        dV = np.swapaxes(dV, 0, self.index+1)

        if f is not None:
            dflow_u = f.diff(shocks)
        else:
            dflow_u = np.zeros_like(lom.P)
        
        dV = dflow_u + dV

        # simply take expectations to get shock to expected value function (envelope result)
        dEV = lom.T @ dV

        # calculate shocks to choice probabilities (note nifty broadcasting of dEV)
        scale = ss[self.taste_shock_scale]
        dP = lom.P * (dV - dEV) / scale
        dlom = DiscreteChoice(dP, self.index)

        # find shocks to outputs, aggregate everything of interest
        doutputs = {self.value: dEV}
        for k in self.backward:
            doutputs[k] = dlom.T @ ss[k]
            if k in shocks:
                doutputs[k] += lom @ shocks[k]
        
        return doutputs, dlom

    def precompute(self, ss, ss_lawofmotion):
        f = self.f.differentiable(ss) if self.f is not None else None
        return f, ss_lawofmotion


class Continuous1D(Stage):
    """Stage that does one-dimensional endogenous continuous choice"""
    def __init__(self, backward, policy, f, name=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = policy

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward))
        self.report = self.f.outputs - self.backward_outputs
        self.inputs = self.f.inputs

    def __repr__(self):
        return f"<Stage-Continuous1D '{self.name}' with policy '{self.policy}'>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_1d(outputs[self.policy], inputs[self.policy + '_grid'], monotonic=False)
    
    def backward_step_shock(self, ss, shocks, precomputed):
        space, i, grid, f = precomputed
        outputs = f.diff(shocks)
        dpi = -outputs[self.policy] / space
        return outputs, ShockedPolicyLottery1D(i, dpi, grid)

    def precompute(self, ss, ss_lawofmotion):
        i = ss_lawofmotion.i.reshape(ss_lawofmotion.shape)
        grid = ss_lawofmotion.grid
        return grid[i + 1] - grid[i], i, grid, self.f.differentiable(ss)


class Continuous2D(Stage):
    """Stage that does two-dimensional endogenous continuous choice"""
    def __init__(self, backward, policy, f, name=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = OrderedSet(policy)

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward))
        self.report = self.f.outputs - self.backward_outputs
        self.inputs = self.f.inputs

    def __repr__(self):
        return f"<Stage-Continuous2D '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_2d(outputs[self.policy[0]], outputs[self.policy[1]],
                                       inputs[self.policy[0] + '_grid'], inputs[self.policy[1] + '_grid'])
    
    def backward_step_shock(self, ss, shocks, precomputed):
        space1, space2, i1, i2, grid1, grid2, f = precomputed
        outputs = f.diff(shocks)
        dpi1 = -outputs[self.policy[0]] / space1
        dpi2 = -outputs[self.policy[1]] / space2
        return outputs, ShockedPolicyLottery2D(i1, dpi1, i2, dpi2, grid1, grid2)

    def precompute(self, ss, ss_lawofmotion):
        i1 = ss_lawofmotion.i1.reshape(ss_lawofmotion.shape)
        i2 = ss_lawofmotion.i2.reshape(ss_lawofmotion.shape)
        grid1 = ss_lawofmotion.grid1
        grid2 = ss_lawofmotion.grid2

        return (grid1[i1 + 1] - grid1[i1], grid2[i2 + 1] - grid2[i2],
                i1, i2, grid1, grid2, self.f.differentiable(ss))


class ExogenousMaker:
    """Call make_stage with backward returned by next stage to get Exogenous stage"""
    def __init__(self, markov_name, index, name=None):
        self.markov_name = markov_name
        self.index = index
        if name is None:
            name = f"exog_{markov_name}"
        self.name = name

    def make_stage(self, backward):
        return Exogenous(self.markov_name, self.index, self.name, backward)


class Exogenous(Stage):
    """Stage that applies exogenous Markov process along one dimension"""
    def __init__(self, markov_name, index, name, backward):
        # subclass-specific attributes
        self.markov_name = markov_name
        self.index = index

        # attributes needed for any stage
        self.name = name
        self.backward_outputs = backward
        self.report = OrderedSet([])
        self.inputs = backward | [markov_name]

    def __repr__(self):
        return f"<Stage-Exogenous '{self.name}' with Markov matrix '{self.markov_name}'>"
    
    def backward_step(self, inputs, lawofmotion=False):
        Pi = Markov(inputs[self.markov_name], self.index)
        outputs = {k: Pi @ inputs[k] for k in self.backward_outputs}

        if not lawofmotion:
            return outputs
        else:
            return outputs, Pi.T
    
    def backward_step_shock(self, ss, shocks, precomputed=None):
        Pi = Markov(ss[self.markov_name], self.index)
        outputs = {k: Pi @ shocks[k] for k in self.backward_outputs if k in shocks}

        if self.markov_name in shocks:
            dPi = Markov(shocks[self.markov_name], self.index)
            for k in self.backward_outputs:
                if k in outputs:
                    outputs[k] += dPi @ ss[k]
                else:
                    outputs[k] = dPi @ ss[k]
            return outputs, dPi.T
        else:
            return outputs, None

