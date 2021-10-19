from ...utilities.function import ExtendedFunction
from ...utilities.ordered_set import OrderedSet
from ...utilities.misc import make_tuple
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

