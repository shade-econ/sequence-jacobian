"""CombinedBlock class and the combine function to generate it"""

from copy import deepcopy

from .block import Block
from .auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from .support.parent import Parent
from ..classes import ImpulseDict, JacobianDict
from ..utilities.graph import block_sort, find_intermediate_inputs


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

    def _steady_state(self, calibration, dissolve, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""

        ss = calibration.copy()
        for block in self.blocks:
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **kwargs)
            ss.update(outputs)

        return ss

    def _impulse_nonlinear(self, ss, inputs, outputs, internals, Js, options):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()

        impulses = inputs.copy()
        for block in self.blocks:
            input_args = {k: v for k, v in impulses.items() if k in block.inputs}

            if input_args:  # If this block is actually perturbed
                impulses.update(block.impulse_nonlinear(ss, input_args, outputs & block.outputs, internals, Js, options))

        return ImpulseDict({k: impulses.toplevel[k] for k in original_outputs}, impulses.internals, impulses.T)

    def _impulse_linear(self, ss, inputs, outputs, Js, options):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()
        
        #irf_lin_partial_eq = inputs.copy()
        irf_lin_partial_eq = deepcopy(inputs)
        for block in self.blocks:
            input_args = {k: v for k, v in irf_lin_partial_eq.items() if k in block.inputs} 

            if input_args:  # If this block is actually perturbed
                irf_lin_partial_eq.update(block.impulse_linear(ss, input_args, outputs & block.outputs, Js, options))

        return irf_lin_partial_eq[original_outputs]

    def _partial_jacobians(self, ss, inputs, outputs, T, Js, options):
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued

        curlyJs = {}
        for block in self.blocks:
            curlyJ = block.partial_jacobians(ss, inputs & block.inputs, outputs & block.outputs, T, Js, options)
            curlyJs.update(curlyJ)
            
        return curlyJs

    def _jacobian(self, ss, inputs, outputs, T, Js, options):
        Js = self._partial_jacobians(ss, inputs, outputs, T, Js, options)

        original_outputs = outputs
        total_Js = JacobianDict.identity(inputs)

        # TODO: horrible, redoing work from partial_jacobians, also need more efficient sifting of intermediates!
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued
        for block in self.blocks:
            J = block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T, Js, options)
            total_Js.update(J @ total_Js)

        return total_Js[original_outputs, :]


# Useful type aliases
Model = CombinedBlock
