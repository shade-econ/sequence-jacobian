"""A CombinedBlock sub-class specifically for steady state calibration with helper blocks"""

from ..combined_block import CombinedBlock
from ...utilities.ordered_set import OrderedSet
from ...utilities.graph import block_sort_w_helpers, find_intermediate_inputs_w_helpers


class CalibrationBlock(CombinedBlock):
    """A CalibrationBlock is a Block object, which includes a set of 'helper' blocks to be used for altering
    the behavior of .steady_state and .solve_steady_state methods. In practice, the common use-case for an
    CalibrationBlock is to help .solve_steady_state solve for a subset of the unknowns/targets analytically."""
    def __init__(self, blocks, helper_blocks, calibration, name=""):
        sorted_indices, inputs, outputs = block_sort_w_helpers(blocks, helper_blocks, calibration, return_io=True)
        intermediate_inputs = find_intermediate_inputs_w_helpers(blocks, helper_blocks=helper_blocks)

        super().__init__(blocks, name=name, sorted_indices=sorted_indices, intermediate_inputs=intermediate_inputs)

        self.helper_blocks = helper_blocks
        self.inputs, self.outputs = OrderedSet(inputs), OrderedSet(outputs)

        self.outputs_orig = set().union(*[block.outputs for block in self.blocks if block not in helper_blocks])
        self.inputs_orig = set().union(*[block.inputs for block in self.blocks if block not in helper_blocks]) - self.outputs_orig

    def __repr__(self):
        return f"<CalibrationBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], helper_targets={}, evaluate_helpers=True, **block_kwargs):
        """Evaluate a partial equilibrium steady state of the RedirectedBlock given a `calibration`"""
        ss = calibration.copy()
        helper_outputs = {}
        for block in self.blocks:
            if not evaluate_helpers and block in self.helper_blocks:
                continue
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **block_kwargs)
            if evaluate_helpers and block in self.helper_blocks:
                helper_outputs.update({k: v for k, v in outputs.toplevel.items() if k in block.outputs | set(helper_targets.keys())})
                ss.update(outputs)
            else:
                # Don't overwrite entries in ss_values corresponding to what has already
                # been solved for in helper_blocks so we can check for consistency after-the-fact
                ss.update(outputs) if evaluate_helpers else ss.update(outputs.difference(helper_outputs))
        return ss
