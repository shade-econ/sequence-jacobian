"""Class definition of a simple block"""

import warnings
import numpy as np
from copy import deepcopy

from .support.simple_displacement import ignore, Displace, AccumulatedDerivative
from .support.impulse import ImpulseDict
from .support.bijection import Bijection
from ..primitives import Block
from ..steady_state.classes import SteadyStateDict
from ..jacobian.classes import JacobianDict, SimpleSparse, ZeroMatrix
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
        self.f = f
        self.name = f.__name__
        self.input_list = misc.input_list(f)
        self.output_list = misc.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)
        self.M = Bijection({})

    def __repr__(self):
        return f"<SimpleBlock '{self.name}'>"

    def _steady_state(self, calibration):
        input_args = {k: ignore(v) for k, v in calibration.items() if k in misc.input_list(self.f)}
        output_vars = [misc.numeric_primitive(o) for o in self.f(**input_args)] if len(self.output_list) > 1 else [
            misc.numeric_primitive(self.f(**input_args))]
        return SteadyStateDict({**calibration, **dict(zip(self.output_list, output_vars))})

    def impulse_nonlinear(self, ss, exogenous):
        input_args = {}
        for k, v in exogenous.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            input_args[k] = Displace(v + ss[k], ss=ss[k], name=k)

        for k in self.input_list:
            if k not in input_args:
                input_args[k] = ignore(ss[k])

        return ImpulseDict(make_impulse_uniform_length(self.f(**input_args), self.output_list)) - ss

    def impulse_linear(self, ss, exogenous, T=None, Js=None):
        return ImpulseDict(self.jacobian(ss, exogenous=list(exogenous.keys()), T=T, Js=Js).apply(exogenous))

    def jacobian(self, ss, exogenous=None, T=None, Js=None):
        """Assemble nested dict of Jacobians

        Parameters
        ----------
        ss : dict,
            steady state values
        exogenous : list of str, optional
            names of input variables to differentiate wrt; if omitted, assume all inputs
        T : int, optional
            number of time periods for explicit T*T Jacobian
            if omitted, more efficient SimpleSparse objects returned
        Js : dict of {str: JacobianDict}, optional
            pre-computed Jacobians

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives Jacobian of o with respect to i
            This Jacobian is a SimpleSparse object or, if T specific, a T*T matrix, omitted by convention if zero
        """

        if exogenous is None:
            exogenous = list(self.inputs)

        relevant_shocks = [i for i in self.inputs if i in exogenous]

        # if we supply Jacobians, use them if possible, warn if they cannot be used
        if Js is not None:
            if misc.verify_saved_jacobian(self.name, Js, self.outputs, relevant_shocks, T):
                return Js[self.name]

        # If none of the shocks passed in shock_list are relevant to this block (i.e. none of the shocks
        # are an input into the block), then return an empty dict
        if not relevant_shocks:
            return JacobianDict({})
        else:
            invertedJ = {shock_name: {} for shock_name in relevant_shocks}

            # Loop over all inputs/shocks which we want to differentiate with respect to
            for shock in relevant_shocks:
                invertedJ[shock] = compute_single_shock_curlyJ(self.f, ss, shock)

            # Because we computed the Jacobian of all outputs with respect to each shock (invertedJ[i][o]),
            # we need to loop back through to have J[o][i] to map for a given output `o`, shock `i`,
            # the Jacobian curlyJ^{o,i}.
            J = {o: {} for o in self.output_list}
            for o in self.output_list:
                for i in relevant_shocks:
                    # Keep zeros, so we can inspect supplied Jacobians for completeness
                    if not invertedJ[i][o] or invertedJ[i][o].iszero:
                        J[o][i] = ZeroMatrix()
                    else:
                        if T is not None:
                            J[o][i] = invertedJ[i][o].matrix(T)
                        else:
                            J[o][i] = invertedJ[i][o]

            return JacobianDict(J, name=self.name)


def compute_single_shock_curlyJ(f, steady_state_dict, shock_name):
    """Find the Jacobian of the function `f` with respect to a single shocked argument, `shock_name`"""
    input_args = {i: ignore(steady_state_dict[i]) for i in misc.input_list(f)}
    input_args[shock_name] = AccumulatedDerivative(f_value=steady_state_dict[shock_name])

    J = {o: {} for o in misc.output_list(f)}
    for o, o_name in zip(misc.make_tuple(f(**input_args)), misc.output_list(f)):
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
