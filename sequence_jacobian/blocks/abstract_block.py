"""AbstractBlock class for defining general input/output access methods for all block types"""

from .. import utilities as utils

# TODO: Look into abc (Abstract Base Class) package for a more formal implementation of AbstractBlock as an ABC

# TODO: Note to self, AbstractBlock will be the supertype of Simple, Het, Solved, and Helper blocks
#   but notably *not* CombinedBlocks, since the commonality between the above blocks is they are all built on top
#   of a *single* user-provided function, where CombinedBlocks are fundamentally different objects, combining
#   multiple single-function-based blocks with other potential functionality like cycle-checking/block optimization.
class AbstractBlock:

    # Define an "abstract __init__ method" to be replaced with @abstractmethod from abc
    def __init__(self, f):
        self.f = f
        self.input_list = utils.misc.input_list(f)
        self.output_list = utils.misc.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)
