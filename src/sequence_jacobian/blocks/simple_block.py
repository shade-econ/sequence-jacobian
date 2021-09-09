"""Class definition of a simple block"""

import numpy as np
from copy import deepcopy

from .support.simple_displacement import ignore, Displace, AccumulatedDerivative
from .block import Block
from ..classes import SteadyStateDict, ImpulseDict, JacobianDict, SimpleSparse, ZeroMatrix
from ..utilities import misc
from ..utilities.function import ExtendedFunction

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
        self.f = ExtendedFunction(f)
        self.name = self.f.name
        self.inputs = self.f.inputs
        self.outputs = self.f.outputs

    def __repr__(self):
        return f"<SimpleBlock '{self.name}'>"

    def _steady_state(self, ss):
        outputs = self.f.wrapped_call(ss, preprocess=ignore, postprocess=misc.numeric_primitive)
        return SteadyStateDict({**ss, **outputs})

    def _impulse_nonlinear(self, ss, inputs, outputs):
        input_args = {}
        for k, v in inputs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            input_args[k] = Displace(v + ss[k], ss=ss[k], name=k)

        for k in self.inputs:
            if k not in input_args:
                input_args[k] = ignore(ss[k])

        return ImpulseDict(make_impulse_uniform_length(self.f(input_args)))[outputs] - ss

    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))

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

        return JacobianDict(J, name=self.name, T=T)[outputs, :]

    def compute_single_shock_J(self, ss, i):
        input_args = {i: ignore(ss[i]) for i in self.inputs}
        input_args[i] = AccumulatedDerivative(f_value=ss[i])

        J = {o: {} for o in self.outputs}
        for o_name, o in self.f(input_args).items():
            if isinstance(o, AccumulatedDerivative):
                J[o_name] = SimpleSparse(o.elements)

        return J


# TODO: move this to impulse.py?
def make_impulse_uniform_length(out):
    T = np.max([np.size(v) for v in out.values()])
    return {k: (np.full(T, misc.numeric_primitive(v)) if np.isscalar(v) else misc.numeric_primitive(v))
                                                        for k, v in out.items()}
