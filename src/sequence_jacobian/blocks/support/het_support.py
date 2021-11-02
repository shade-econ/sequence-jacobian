import numpy as np
from . import het_compiled
from ...utilities.discretize import stationary as general_stationary
from ...utilities.interpolate import interpolate_coord_robust, interpolate_coord
from ...utilities.multidim import batch_multiply_ith_dimension, multiply_ith_dimension
from ...utilities.misc import logsum
from typing import Optional, Sequence, Any, List, Tuple, Union

class Transition:
    """Abstract class for PolicyLottery or ManyMarkov, i.e. some part of state-space transition"""
    def forward(self, D):
        pass

    def expectation(self, X):
        pass

    def forward_shockable(self, Dss):
        pass

    def expectation_shockable(self, Xss):
        raise NotImplementedError(f'Shockable expectation not implemented for {type(self)}')


class ForwardShockableTransition(Transition):
    """Abstract class extending Transition, allowing us to find effect of shock to transition rule
    on one-period-ahead distribution. This functionality isn't included in the regular Transition
    because it requires knowledge of the incoming ("steady-state") distribution and also sometimes
    some precomputation.

    One crucial thing here is the order of shock arguments in shocks. Also, is None is the default
    argument for a shock, we allow that shock to be None. We always allow shocks in lists to be None."""

    def forward_shock(self, shocks):
        pass


class ExpectationShockableTransition(Transition):
    def expectation_shock(self, shocks):
        pass



def lottery_1d(a, a_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery1D(*interpolate_coord_robust(a_grid, a), a_grid)
    else:
        return PolicyLottery1D(*interpolate_coord(a_grid, a), a_grid)
    

class PolicyLottery1D(Transition):
    # TODO: always operates on final dimension, highly non-generic in that sense
    def __init__(self, i, pi, grid):
        # flatten non-policy dimensions into one because that's what methods accept
        self.i = i.reshape((-1,) + grid.shape)
        self.flatshape = self.i.shape

        self.pi = pi.reshape(self.flatshape)

        # but store original shape so we can convert all outputs to it
        self.shape = i.shape
        self.grid = grid

        # also store shape of the endogenous grid itself
        self.endog_shape = self.shape[-1:]
        
    def forward(self, D):
        return het_compiled.forward_policy_1d(D.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)
    
    def expectation(self, X):
        return het_compiled.expectation_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)

    def forward_shockable(self, Dss):
        return ForwardShockablePolicyLottery1D(self.i.reshape(self.shape), self.pi.reshape(self.shape),
                                      self.grid, Dss)


class ForwardShockablePolicyLottery1D(PolicyLottery1D, ForwardShockableTransition):
    def __init__(self, i, pi, grid, Dss):
        super().__init__(i, pi, grid)
        self.Dss = Dss.reshape(self.flatshape)
        self.space = grid[self.i+1] - grid[self.i]

    def forward_shock(self, da):
        pi_shock = - da.reshape(self.flatshape) / self.space
        return het_compiled.forward_policy_shock_1d(self.Dss, self.i, pi_shock).reshape(self.shape)


def lottery_2d(a, b, a_grid, b_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery2D(*interpolate_coord_robust(a_grid, a),
                           *interpolate_coord_robust(b_grid, b), a_grid, b_grid)
    if monotonic:
        # right now we have no monotonic 2D examples, so this shouldn't be called
        return PolicyLottery2D(*interpolate_coord(a_grid, a),
                           *interpolate_coord(b_grid, b), a_grid, b_grid)


class PolicyLottery2D(Transition):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2):
        # flatten non-policy dimensions into one because that's what methods accept
        self.i1 = i1.reshape((-1,) + grid1.shape + grid2.shape)
        self.flatshape = self.i1.shape

        self.i2 = i2.reshape(self.flatshape)
        self.pi1 = pi1.reshape(self.flatshape)
        self.pi2 = pi2.reshape(self.flatshape)

        # but store original shape so we can convert all outputs to it
        self.shape = i1.shape
        self.grid1 = grid1
        self.grid2 = grid2

        # also store shape of the endogenous grid itself
        self.endog_shape = self.shape[-2:]

    def forward(self, D):
        return het_compiled.forward_policy_2d(D.reshape(self.flatshape), self.i1, self.i2,
                                                self.pi1, self.pi2).reshape(self.shape)
    
    def expectation(self, X):
        return het_compiled.expectation_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                    self.pi1, self.pi2).reshape(self.shape)

    def forward_shockable(self, Dss):
        return ForwardShockablePolicyLottery2D(self.i1.reshape(self.shape), self.pi1.reshape(self.shape),
                                            self.i2.reshape(self.shape), self.pi2.reshape(self.shape),
                                            self.grid1, self.grid2, Dss)


class ForwardShockablePolicyLottery2D(PolicyLottery2D, ForwardShockableTransition):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, Dss):
        super().__init__(i1, pi1, i2, pi2, grid1, grid2)
        self.Dss = Dss.reshape(self.flatshape)
        self.space1 = grid1[self.i1+1] - grid1[self.i1]
        self.space2 = grid2[self.i2+1] - grid2[self.i2]

    def forward_shock(self, da):
        da1, da2 = da
        pi_shock1 = -da1.reshape(self.flatshape) / self.space1
        pi_shock2 = -da2.reshape(self.flatshape) / self.space2

        return het_compiled.forward_policy_shock_2d(self.Dss, self.i1, self.i2, self.pi1, self.pi2,
                                                    pi_shock1, pi_shock2).reshape(self.shape)


class Markov(Transition):
    def __init__(self, Pi, i):
        self.Pi = Pi
        self.Pi_T = self.Pi.T
        if isinstance(self.Pi_T, np.ndarray):
            # optimization: copy to get right order in memory
            self.Pi_T = self.Pi_T.copy()
        self.i = i

    def forward(self, D):
        return multiply_ith_dimension(self.Pi_T, self.i, D)

    def expectation(self, X):
        return multiply_ith_dimension(self.Pi, self.i, X)

    def forward_shockable(self, Dss):
        return ForwardShockableMarkov(self.Pi, self.i, Dss)

    def expectation_shockable(self, Xss):
        return ExpectationShockableMarkov(self.Pi, self.i, Xss)

    def stationary(self, pi_seed, tol=1E-11, maxit=10_000):
        return general_stationary(self.Pi, pi_seed, tol, maxit)
    

class ForwardShockableMarkov(Markov, ForwardShockableTransition):
    def __init__(self, Pi, i, Dss):
        super().__init__(Pi, i)
        self.Dss = Dss

    def forward_shock(self, dPi):
        return multiply_ith_dimension(dPi.T, self.i, self.Dss)


class ExpectationShockableMarkov(Markov, ExpectationShockableTransition):
    def __init__(self, Pi, i, Xss):
        super().__init__(Pi, i)
        self.Xss = Xss

    def expectation_shock(self, dPi):
        return multiply_ith_dimension(dPi, self.i, self.Xss)


class CombinedTransition(Transition):
    def __init__(self, stages: Sequence[Transition]):
        self.stages = stages
    
    def forward(self, D):
        for stage in self.stages:
            D = stage.forward(D)
        return D

    def expectation(self, X):
        for stage in reversed(self.stages):
            X = stage.expectation(X)
        return X

    def forward_shockable(self, Dss):
        shockable_stages = []
        for stage in self.stages:
            shockable_stages.append(stage.forward_shockable(Dss))
            Dss = stage.forward(Dss)

        return ForwardShockableCombinedTransition(shockable_stages)

    def expectation_shockable(self, Xss):
        shockable_stages = []
        for stage in reversed(self.stages):
            shockable_stages.append(stage.expectation_shockable(Xss))
            Xss = stage.expectation(Xss)

        return ExpectationShockableCombinedTransition(list(reversed(shockable_stages)))

    def __getitem__(self, i):
        return self.stages[i]


Shock = Any
ListTupleShocks = Union[List[Shock], Tuple[Shock]]

class ForwardShockableCombinedTransition(CombinedTransition, ForwardShockableTransition):
    def __init__(self, stages: Sequence[ForwardShockableTransition]):
        self.stages = stages
        self.Dss = stages[0].Dss

    def forward_shock(self, shocks: Optional[Sequence[Optional[Union[Shock, ListTupleShocks]]]]):
        if shocks is None:
            return None

        # each entry of shocks is either a sequence (list or tuple) 
        dD = None

        for stage, shock in zip(self.stages, shocks):
            if shock is not None:
                dD_shock = stage.forward_shock(shock)
            else:
                dD_shock = None

            if dD is not None:
                dD = stage.forward(dD)

                if shock is not None:
                    dD += dD_shock
            else:
                dD = dD_shock

        return dD


class ExpectationShockableCombinedTransition(CombinedTransition, ExpectationShockableTransition):
    def __init__(self, stages: Sequence[ExpectationShockableTransition]):
        self.stages = stages
        self.Xss = stages[-1].Xss

    def expectation_shock(self, shocks: Sequence[Optional[Union[Shock, ListTupleShocks]]]):
        dX = None

        for stage, shock in zip(reversed(self.stages), reversed(shocks)):
            if shock is not None:
                dX_shock = stage.expectation_shock(shock)
            else:
                dX_shock = None

            if dX is not None:
                dX = stage.expectation(dX)

                if shock is not None:
                    dX += dX_shock
            else:
                dX = dX_shock

        return dX


class DiscreteChoice(Transition):
    def __init__(self, P, i):
        self.P = P                     # choice prob P(d|...s_i...), 0 for unavailable choices
        self.P_T = P.swapaxes(0, 1+i)  # P_T(s_i|...d...)
        self.i = i                     # dimension of state space that will be updated

    def forward(self, D):
        return batch_multiply_ith_dimension(self.P, self.i, D)

    def expectation(self, X):
        '''NOT meant for value function'''
        return batch_multiply_ith_dimension(self.P_T, self.i, X)

    def forward_shockable(self, Dss):
        return NotImplementedError