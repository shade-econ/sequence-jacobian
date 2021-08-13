"""Class definition of a simple block"""

import warnings
import numpy as np
from copy import deepcopy

from .support.simple_displacement import ignore, Displace, AccumulatedDerivative
from .support.impulse import ImpulseDict
from .support.bijection import Bijection
from ..primitives import Block
from ..steady_state.classes import SteadyStateDict
from ..jacobian.classes import JacobianDict, SimpleSparse, ZeroMatrix, verify_saved_jacobian
from ..utilities import misc

'''Part 1: SimpleBlock class and @simple decorator to generate it'''


def simple(f):
    return SimpleBlock(f)


class SimpleBlock(Block):
    """Generated from simple block written in Dynare-ish style and decorated with @simple, e.g.

    @simple
    def production(Z, K, L, alpha):
        Y = Z * K(-1) ** alpha * L ** (1 - alpha)
        return Y

    which is a SimpleBlock that takes in Z, K, L, and alpha, all of which can be either constants
    or series, and implements a Cobb-Douglas production function, noting that for production today
    we use the capital K(-1) determined yesterday.

    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f
        self.name = f.__name__
        self.output_list = misc.output_list(f)
        self.inputs = set(misc.input_list(f))
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<SimpleBlock '{self.name}'>"

    def _steady_state(self, calibration):
        input_args = {k: ignore(v) for k, v in calibration.items() if k in misc.input_list(self.f)}
        output_vars = [misc.numeric_primitive(o) for o in self.f(**input_args)] if len(self.output_list) > 1 else [
            misc.numeric_primitive(self.f(**input_args))]
        return SteadyStateDict({**calibration, **dict(zip(self.output_list, output_vars))})

    def _impulse_nonlinear(self, ss, exogenous):
        input_args = {}
        for k, v in exogenous.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            input_args[k] = Displace(v + ss[k], ss=ss[k], name=k)

        for k in self.input_list:
            if k not in input_args:
                input_args[k] = ignore(ss[k])

        return ImpulseDict(make_impulse_uniform_length(self.f(**input_args), self.output_list)) - ss

    def _impulse_linear(self, ss, exogenous, T=None, Js=None):
        return ImpulseDict(self.jacobian(ss, exogenous=list(exogenous.keys()), T=T, Js=Js).apply(exogenous))

    def _jacobian(self, ss, inputs, outputs, T):
        invertedJ = {i: {} for i in inputs}

        # Loop over all inputs/shocks which we want to differentiate with respect to
        for i in inputs:
            invertedJ[i] = self.compute_single_shock_J(ss, i)

        # Because we computed the Jacobian of all outputs with respect to each shock (invertedJ[i][o]),
        # we need to loop back through to have J[o][i] to map for a given output `o`, shock `i`,
        # the Jacobian curlyJ^{o,i}.
        J = {o: {} for o in self.outputs}
        for o in self.outputs:
            for i in inputs:
                # Keep zeros, so we can inspect supplied Jacobians for completeness
                if not invertedJ[i][o] or invertedJ[i][o].iszero:
                    J[o][i] = ZeroMatrix()
                else:
                    J[o][i] = invertedJ[i][o]

        print(J)

        return JacobianDict(J, name=self.name)[outputs, :]

    def compute_single_shock_J(self, ss, i):
        input_args = {i: ignore(ss[i]) for i in self.inputs}
        input_args[i] = AccumulatedDerivative(f_value=ss[i])

        J = {o: {} for o in self.output_list}
        for o, o_name in zip(misc.make_tuple(self.f(**input_args)), self.output_list):
            if isinstance(o, AccumulatedDerivative):
                J[o_name] = SimpleSparse(o.elements)

        return J


def make_impulse_uniform_length(out, output_list):
    # If the function has multiple outputs
    if isinstance(out, tuple):
        # Because we know at least one of the outputs in `out` must be of length T
        T = np.max([np.size(o) for o in out])
        out_unif_dim = [np.full(T, misc.numeric_primitive(o)) if np.isscalar(o) else
                        misc.numeric_primitive(o) for o in out]
        return dict(zip(output_list, misc.make_tuple(out_unif_dim)))
    else:
        return dict(zip(output_list, misc.make_tuple(misc.numeric_primitive(out))))
