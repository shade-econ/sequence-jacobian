"""HelperBlock class and @helper decorator to generate it"""

import warnings

from ..utilities import misc


def helper(f):
    return HelperBlock(f)


class HelperBlock:
    """ A block for providing pre-computed solutions in lieu of solving for variables within the DAG.
    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        self.f = f
        self.name = f.__name__
        self.input_list = misc.input_list(f)
        self.output_list = misc.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<HelperBlock '{self.f.__name__}'>"

    # TODO: Deprecated methods, to be removed!
    def ss(self, *args, **kwargs):
        warnings.warn("This method has been deprecated. Please invoke by calling .steady_state", DeprecationWarning)
        return self.steady_state(*args, **kwargs)

    def _output_in_ss_format(self, *args, **kwargs):
        """Returns output of the method ss as either a tuple of numeric primitives (scalars/vectors) or a single
        numeric primitive, as opposed to Ignore/IgnoreVector objects"""
        if len(self.output_list) > 1:
            return dict(zip(self.output_list, [misc.numeric_primitive(o) for o in self.f(*args, **kwargs)]))
        else:
            return dict(zip(self.output_list, [misc.numeric_primitive(self.f(*args, **kwargs))]))

    # Currently does not use any of the machinery in SimpleBlock to deal with time displacements and hence
    # can handle non-scalar inputs.
    def steady_state(self, *args, **kwargs):
        args = [x for x in args]
        kwargs = {k: v for k, v in kwargs.items()}
        return self._output_in_ss_format(*args, **kwargs)
