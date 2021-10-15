import numpy as np
from . import het_compiled
from ...utilities.discretize import stationary as general_stationary
from ...utilities.interpolate import interpolate_coord_robust, interpolate_coord
from ...utilities.multidim import batch_multiply_ith_dimension, multiply_ith_dimension
from ...utilities.misc import logsum
from typing import Optional, Sequence, Any, List, Tuple, Union
import copy

class LawOfMotion:
    """Abstract class representing a matrix that operates on state space.
    Rather than giant Ns*Ns matrix (even if sparse), some other representation
    almost always desirable; such representations are subclasses of this."""
    
    def __matmul__(self, X):
        pass
    
    @property
    def T(self):
        pass


def lottery_1d(a, a_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery1D(*interpolate_coord_robust(a_grid, a), a_grid)
    else:
        return PolicyLottery1D(*interpolate_coord(a_grid, a), a_grid)


class PolicyLottery1D(LawOfMotion):
    # TODO: always operates on final dimension, make more general!
    def __init__(self, i, pi, grid, forward=True):
        # flatten non-policy dimensions into one because that's what methods accept
        self.i = i.reshape((-1,) + grid.shape)
        self.flatshape = self.i.shape

        self.pi = pi.reshape(self.flatshape)

        # but store original shape so we can convert all outputs to it
        self.shape = i.shape
        self.grid = grid

        # also store shape of the endogenous grid itself
        self.endog_shape = self.shape[-1:]
        
        self.forward = forward

    @property
    def T(self):
        newself = copy.copy(self)
        newself.forward = not self.forward
        return newself

    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)
        else:
            return het_compiled.expectation_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)


class ShockedPolicyLottery1D(PolicyLottery1D):
    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_shock_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)
        else:
            raise NotImplementedError


def lottery_2d(a, b, a_grid, b_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery2D(*interpolate_coord_robust(a_grid, a),
                           *interpolate_coord_robust(b_grid, b), a_grid, b_grid)
    if monotonic:
        # right now we have no monotonic 2D examples, so this shouldn't be called
        return PolicyLottery2D(*interpolate_coord(a_grid, a),
                           *interpolate_coord(b_grid, b), a_grid, b_grid)


class PolicyLottery2D(LawOfMotion):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, forward=True):
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

        self.forward = forward

    @property
    def T(self):
        newself = copy.copy(self)
        newself.forward = not self.forward
        return newself

    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                self.pi1, self.pi2).reshape(self.shape)
        else:
            return het_compiled.expectation_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                    self.pi1, self.pi2).reshape(self.shape)


class ShockedPolicyLottery2D(PolicyLottery2D):
    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_shock_2d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)
        else:
            raise NotImplementedError


class Markov(LawOfMotion):
    def __init__(self, Pi, i):
        self.Pi = Pi
        self.i = i

    @property
    def T(self):
        newself = copy.copy(self)
        newself.Pi = newself.Pi.T
        if isinstance(newself.Pi, np.ndarray):
            # optimizing: copy to get right order in memory
            newself.Pi = newself.Pi.copy()
        return newself

    def __matmul__(self, X):
        return multiply_ith_dimension(self.Pi, self.i, X)


class DiscreteChoice(LawOfMotion):
    def __init__(self, P, i, scale):
        self.P = P                     # choice prob P(d|...s_i...), 0 for unavailable choices
        self.i = i                     # dimension of state space that will be updated
        self.scale = scale             # scale of taste shocks (on grid?)

    @property
    def T(self):
        newself = copy.copy(self)
        newself.P = self.P.swapaxes(0, 1+self.i).copy()
        return newself

    def __matmul__(self, X):
        return batch_multiply_ith_dimension(self.P, self.i, X)

