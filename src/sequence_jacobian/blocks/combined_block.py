"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from ..primitives import Block
from ..utilities.misc import dict_diff
from ..utilities.graph import block_sort, find_intermediate_inputs
from ..utilities.graph import topological_sort
from ..utilities.ordered_set import OrderedSet
from ..blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from ..blocks.parent import Parent
from ..jacobian.classes import JacobianDict


def combine(blocks, name="", model_alias=False):
    return CombinedBlock(blocks, name=name, model_alias=model_alias)


# Useful functional alias
def create_model(blocks, **kwargs):
    return combine(blocks, model_alias=True, **kwargs)


class CombinedBlock(Block, Parent):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, blocks, name="", model_alias=False, sorted_indices=None, intermediate_inputs=None):
        super().__init__()

        self._blocks_unsorted = [b if isinstance(b, Block) else JacobianDictBlock(b) for b in blocks]
        self._sorted_indices = block_sort(blocks) if sorted_indices is None else sorted_indices
        self._required = find_intermediate_inputs(blocks) if intermediate_inputs is None else intermediate_inputs
        self.blocks = [self._blocks_unsorted[i] for i in self._sorted_indices]

        if not name:
            self.name = f"{self.blocks[0].name}_to_{self.blocks[-1].name}_combined"
        else:
            self.name = name

        # now that it has a name, do Parent initialization
        Parent.__init__(self, blocks)

        # Find all outputs (including those used as intermediary inputs)
        self.outputs = set().union(*[block.outputs for block in self.blocks])

        # Find all inputs that are *not* intermediary outputs
        all_inputs = set().union(*[block.inputs for block in self.blocks])
        self.inputs = all_inputs.difference(self.outputs)

        # If the create_model() is used instead of combine(), we will have __repr__ show this object as a 'Model'
        self._model_alias = model_alias

    def __repr__(self):
        if self._model_alias:
            return f"<Model '{self.name}'>"
        else:
            return f"<CombinedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""

        ss = deepcopy(calibration)
        for block in self.blocks:
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **kwargs)
            ss.update(outputs)

        return ss

    def _impulse_nonlinear(self, ss, inputs, outputs, Js):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()

        irf_nonlin_partial_eq = deepcopy(inputs)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_nonlin_partial_eq.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                irf_nonlin_partial_eq.update(block.impulse_nonlinear(ss, input_args, outputs & block.outputs, Js))

        return irf_nonlin_partial_eq[original_outputs]

    def _impulse_linear(self, ss, inputs, outputs, Js):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()
        
        irf_lin_partial_eq = deepcopy(inputs)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs} 

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update(block.impulse_linear(ss, input_args, outputs & block.outputs, Js))

        return irf_lin_partial_eq[original_outputs]

    def _partial_jacobians(self, ss, inputs, outputs, T, Js):
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued

        curlyJs = {}
        for block in self.blocks:
            descendants = block.descendants if isinstance(block, Parent) else {block.name: None}
            Js_block = {k: v for k, v in Js.items() if k in descendants}

            curlyJ = block.partial_jacobians(ss, inputs & block.inputs, outputs & block.outputs, T, Js_block)
            curlyJs.update(curlyJ)
            
        return curlyJs

    def _jacobian(self, ss, inputs, outputs, T, Js={}):
        Js = self._partial_jacobians(ss, inputs, outputs, T=T, Js=Js)

        original_outputs = outputs
        total_Js = JacobianDict.identity(inputs)

        # TODO: horrible, redoing work from partial_jacobians, also need more efficient sifting of intermediates!
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued
        for block in self.blocks:
            descendants = block.descendants if isinstance(block, Parent) else {block.name: None}
            Js_block = {k: v for k, v in Js.items() if k in descendants}
            J = block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T, Js_block)
            total_Js.update(J @ total_Js)

        return total_Js[original_outputs, :]


# Useful type aliases
Model = CombinedBlock
