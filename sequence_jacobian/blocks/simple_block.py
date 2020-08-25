import numpy as np

from .. import utilities as utils
from .. import jacobian
from .support.simple_displacement import ignore, numeric_primitive, Displace, AccumulatedDerivative

'''Part 1: SimpleBlock class and @simple decorator to generate it'''


def simple(f):
    return SimpleBlock(f)


class SimpleBlock:
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
        self.input_list = utils.misc.input_list(f)
        self.output_list = utils.misc.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<SimpleBlock '{self.f.__name__}'>"

    def _output_in_ss_format(self, *args, **kwargs):
        """Returns output of the method ss as either a tuple of numeric primitives (scalars/vectors) or a single
        numeric primitive, as opposed to Ignore/IgnoreVector objects"""
        if len(self.output_list) > 1:
            return tuple([numeric_primitive(o) for o in self.f(*args, **kwargs)])
        else:
            return numeric_primitive(self.f(*args, **kwargs))

    def ss(self, *args, **kwargs):
        # Wrap args and kwargs in Ignore/IgnoreVector classes to be passed into the function "f"
        args = [ignore(x) for x in args]
        kwargs = {k: ignore(v) for k, v in kwargs.items()}

        return self._output_in_ss_format(*args, **kwargs)

    def _output_in_td_format(self, **kwargs_new):
        """Returns output of the method td as a dict mapping output names to numeric primitives (scalars/vectors)
        or a single numeric primitive of output values, as opposed to Ignore/IgnoreVector/Displace objects.

        Also accounts for the fact that for outputs of block.td that were *not* affected by a Displace object, i.e.
        variables that remained at their ss value in spite of other variables within that same block being
        affected by the Displace object (e.g. I in the mkt_clearing block of the two_asset model
        is unchanged by a shock to rstar, being only a function of K's ss value and delta),
        we still want to return them as paths (i.e. vectors, if they were
        previously scalars) to impose uniformity on the dimensionality of the td returned values.
        """
        out = self.f(**kwargs_new)
        if len(self.output_list) > 1:
            # Because we know at least one of the outputs in `out` must be of length T
            T = np.max([np.size(o) for o in out])
            out_unif_dim = [np.full(T, numeric_primitive(o)) if np.isscalar(o) else numeric_primitive(o) for o in out]
            return dict(zip(self.output_list, utils.misc.make_tuple(out_unif_dim)))
        else:
            return dict(zip(self.output_list, utils.misc.make_tuple(numeric_primitive(out))))

    def td(self, ss, **kwargs):
        kwargs_new = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            kwargs_new[k] = Displace(v, ss=ss.get(k, None), name=k)

        for k in self.input_list:
            if k not in kwargs_new:
                kwargs_new[k] = ignore(ss[k])

        return self._output_in_td_format(**kwargs_new)

    def jac(self, ss, T=None, shock_list=[]):
        """Assemble nested dict of Jacobians

        Parameters
        ----------
        ss : dict,
            steady state values
        T : int, optional
            number of time periods for explicit T*T Jacobian
            if omitted, more efficient SimpleSparse objects returned
        shock_list : list of str, optional
            names of input variables to differentiate wrt; if omitted, assume all inputs
        h : float, optional
            radius for symmetric numerical differentiation

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives Jacobian of o with respect to i
            This Jacobian is a SimpleSparse object or, if T specific, a T*T matrix, omitted by convention
            if zero
        """

        relevant_shocks = [i for i in self.inputs if i in shock_list]

        # If none of the shocks passed in shock_list are relevant to this block (i.e. none of the shocks
        # are an input into the block), then return an empty dict
        if not relevant_shocks:
            return {}
        else:
            invertedJ = {shock_name: {} for shock_name in relevant_shocks}

            # Loop over all inputs/shocks which we want to differentiate with respect to
            for shock in relevant_shocks:
                invertedJ[shock] = compute_single_shock_curlyJ(self.f, ss, shock, T=T)

            # Because we computed the Jacobian of all outputs with respect to each shock (invertedJ[i][o]),
            # we need to loop back through to have J[o][i] to map for a given output `o`, shock `i`,
            # the Jacobian curlyJ^{o,i}.
            J = {o: {} for o in self.output_list}
            for o in self.output_list:
                for i in relevant_shocks:
                    # Do not write an entry into J if shock `i` did not affect output `o`
                    if not invertedJ[i][o] or invertedJ[i][o].iszero:
                        continue
                    else:
                        if T is not None:
                            J[o][i] = invertedJ[i][o].nonzero().matrix(T)
                        else:
                            J[o][i] = invertedJ[i][o].nonzero()

                # If output `o` is entirely unaffected by all of the shocks passed in, then
                # remove the empty Jacobian corresponding to `o` from J
                if not J[o]:
                    del J[o]

            return J


def compute_single_shock_curlyJ(f, steady_state_dict, shock_name, T=None):
    """Find the Jacobian of the function `f` with respect to a single shocked argument, `shock_name`"""
    input_args = {i: ignore(steady_state_dict[i]) for i in utils.misc.input_list(f)}
    input_args[shock_name] = AccumulatedDerivative(f_value=steady_state_dict[shock_name])

    J = {o: {} for o in utils.misc.output_list(f)}
    for o, o_name in zip(utils.misc.make_tuple(f(**input_args)), utils.misc.output_list(f)):
        if isinstance(o, AccumulatedDerivative):
            J[o_name] = jacobian.SimpleSparse(o.elements)

    return J
