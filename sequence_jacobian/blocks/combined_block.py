"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from .. import utilities as utils
from ..steady_state import eval_block_ss


def combine(*args):
    # TODO: Implement a check that all args are child types of AbstractBlock, when that is properly implemented
    return CombinedBlock(*args)


class CombinedBlock:

    def __init__(self, *args, name=""):
        self.blocks = args
        if not name:
            self.name = f"{self.blocks[0].name}_to_{self.blocks[-1].name}_combined"
        else:
            self.name = name

        # Find all outputs (including those used as intermediary inputs)
        self.outputs = set().union(*[block.outputs for block in self.blocks])

        # Find all inputs that are *not* intermediary outputs
        all_inputs = set().union(*[block.inputs for block in self.blocks])
        self.inputs = all_inputs.difference(self.outputs)

        self.blocks_sorted_indices = utils.graph.block_sort(self.blocks)

    def ss(self, **kwargs):
        ss_values = deepcopy(kwargs)
        for i in self.blocks_sorted_indices:
            ss_values.update(eval_block_ss(self.blocks[i], ss_values))
        return ss_values

    # TODO: Define td method for CombinedBlock
    def td(self, ss, **kwargs):
        pass

