import numpy as np
from . import het_compiled
from ...utilities.discretize import stationary as general_stationary
from ...utilities.interpolate import interpolate_coord_robust
from typing import Optional, Sequence, Any, List, Tuple, Union

class Transition:
    """Abstract class for PolicyLottery or ManyMarkov, i.e. some part of state-space transition"""
    def forward(self, D):
        pass

    def expectations(self, X):
        pass

    def shockable(self, Dss):
        # return ShockableTransition
        pass


class ShockableTransition(Transition):
    """Abstract class extending Transition, allowing us to find effect of shock to transition rule
    on one-period-ahead distribution. This functionality isn't included in the regular Transition
    because it requires knowledge of the incoming ("steady-state") distribution and also sometimes
    some precomputation.

    One crucial thing here is the order of shock arguments in shocks. Also, is None is the default
    argument for a shock, we allow that shock to be None. We always allow shocks in lists to be None."""

    def forward_shock(self, shocks):
        pass


def lottery_1d(a, a_grid):
    return PolicyLottery1D(*interpolate_coord_robust(a_grid, a), a_grid)
    

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
    
    def expectations(self, X):
        return het_compiled.expectations_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)

    def shockable(self, Dss):
        return ShockablePolicyLottery1D(self.i.reshape(self.shape), self.pi.reshape(self.shape),
                                      self.grid, Dss)


class ShockablePolicyLottery1D(PolicyLottery1D, ShockableTransition):
    def __init__(self, i, pi, grid, Dss):
        super().__init__(i, pi, grid)
        self.Dss = Dss.reshape(self.flatshape)
        self.space = grid[self.i+1] - grid[self.i]

    def forward_shock(self, da, dgrid=None):
        # TODO: think about da being None too for more general applications
        pi_shock = - da.reshape(self.flatshape) / self.space

        if dgrid is not None:
            # see "linearizing_interpolation" note
            dgrid = np.broadcast_to(dgrid, self.shape)
            pi_shock += self.expectations(dgrid).reshape(self.flatshape) / self.space

        return het_compiled.forward_policy_shock_1d(self.Dss, self.i, pi_shock).reshape(self.shape)


def lottery_2d(a, b, a_grid, b_grid):
    return PolicyLottery2D(*interpolate_coord_robust(a_grid, a),
                           *interpolate_coord_robust(b_grid, b), a_grid, b_grid)


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
    
    def expectations(self, X):
        return het_compiled.expectations_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                    self.pi1, self.pi2).reshape(self.shape)

    def shockable(self, Dss):
        return ShockablePolicyLottery2D(self.i1.reshape(self.shape), self.pi1.reshape(self.shape),
                                            self.i2.reshape(self.shape), self.pi2.reshape(self.shape),
                                            self.grid1, self.grid2, Dss)


class ShockablePolicyLottery2D(PolicyLottery2D, ShockableTransition):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, Dss):
        super().__init__(i1, pi1, i2, pi2, grid1, grid2)
        self.Dss = Dss.reshape(self.flatshape)
        self.space1 = grid1[self.i1+1] - grid1[self.i1]
        self.space2 = grid2[self.i2+1] - grid2[self.i2]

    def forward_shock(self, da1, da2, dgrid1=None, dgrid2=None):
        pi_shock1 = -da1.reshape(self.flatshape) / self.space1
        pi_shock2 = -da2.reshape(self.flatshape) / self.space2

        if dgrid1 is not None:
            dgrid1 = np.broadcast_to(dgrid1[:, np.newaxis], self.shape)
            pi_shock1 += self.expectations(dgrid1).reshape(self.flatshape) / self.space1
        
        if dgrid2 is not None:
            dgrid2 = np.broadcast_to(dgrid2, self.shape)
            pi_shock2 += self.expectations(dgrid2).reshape(self.flatshape) / self.space2

        return het_compiled.forward_policy_shock_2d(self.Dss, self.i1, self.i2, self.pi1, self.pi2,
                                                    pi_shock1, pi_shock2).reshape(self.shape)


def multiply_ith_dimension(Pi, i, X):
    """If Pi is a square matrix, multiply Pi times the ith dimension of X and return"""
    X = X.swapaxes(0, i)
    shape = X.shape
    X = X.reshape((X.shape[0], -1))

    # iterate forward using Pi
    X = Pi @ X

    # reverse steps
    X = X.reshape(shape)
    return X.swapaxes(0, i)


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

    def expectations(self, X):
        return multiply_ith_dimension(self.Pi, self.i, X)

    def shockable(self, Dss):
        return ShockableMarkov(self.Pi, self.i, Dss)

    def stationary(self, tol=1E-11, maxit=10_000):
        pi_seed = getattr(self.Pi, 'pi_seed', None)
        return general_stationary(self.Pi, pi_seed, tol, maxit)
    

class ShockableMarkov(Markov, ShockableTransition):
    def __init__(self, Pi, i, Dss):
        super().__init__(Pi, i)
        self.Dss = Dss

    def forward_shock(self, dPi):
        return multiply_ith_dimension(dPi.T, self.i, self.Dss)


class CombinedTransition(Transition):
    def __init__(self, stages: Sequence[Transition]):
        self.stages = stages
    
    def forward(self, D):
        for stage in self.stages:
            D = stage.forward(D)
        return D

    def expectations(self, X):
        for stage in reversed(self.stages):
            X = stage.expectations(X)
        return X

    def shockable(self, Dss):
        shockable_stages = []
        for stage in self.stages:
            shockable_stages.append(stage.shockable(Dss))
            Dss = stage.forward(Dss)

        return ShockableCombinedTransition(shockable_stages)

Shock = Any
ListTupleShocks = Union[List[Shock], Tuple[Shock]]

class ShockableCombinedTransition(CombinedTransition, ShockableTransition):
    def __init__(self, stages: Sequence[ShockableTransition]):
        self.stages = stages

    def forward_shock(self, shocks: Sequence[Optional[Union[Shock, ListTupleShocks]]]):
        # each entry of shocks is either a sequence (list or tuple) 
        dD = None

        for stage, shock in zip(self.stages, shocks):
            if shock is not None:
                if isinstance(shock, tuple) or isinstance(shock, list):
                    dD_shock = stage.forward_shock(*shock)
                else:
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
