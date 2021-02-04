"""HelperBlock class and @helper decorator to generate it"""

from .. import utilities as utils


def helper(f):
    return HelperBlock(f)


class HelperBlock:
    """ A block for providing pre-computed solutions in lieu of solving for variables within the DAG.
    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        self.f = f
        self.name = f.__name__
        self.input_list = utils.misc.input_list(f)
        self.output_list = utils.misc.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<HelperBlock '{self.f.__name__}'>"

    # Currently does not use any of the machinery in SimpleBlock to deal with time displacements and hence
    # can handle non-scalar inputs.
    def ss(self, *args, **kwargs):
        args = [x for x in args]
        kwargs = {k: v for k, v in kwargs.items()}
        return self.f(*args, **kwargs)
