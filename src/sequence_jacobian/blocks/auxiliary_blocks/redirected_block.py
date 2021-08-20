"""An auxiliary Block class for altering the direction of some of the equations in a given Block object"""

import numpy as np

from ..combined_block import CombinedBlock
from ..parent import Parent
from ...primitives import Block
from ...utilities.ordered_set import OrderedSet


class RedirectedBlock(CombinedBlock):
    """A RedirectedBlock is a Block where a subset of the input-output mappings are altered, or 'redirected'.
       This is useful when computing steady states in particular, where often we will want to set target values
       explicitly and back out what are the implied values of the unknowns that justify those targets."""
    def __init__(self, block, redirect_block):
        Block.__init__(self)

        # TODO: Figure out what are the criteria we want to require of the helper block
        # if not redirect_block.inputs & block.outputs:
        #     raise ValueError("User-provided redirect_block must ")
        # assert redirect_block.outputs <= (block.inputs | block.outputs)
        self.directed = block
        # TODO: Implement case with multiple redirecting (helper) blocks later.
        #   If `block` is a non-nested Block then multiple helper blocks may seem a bit redundant, since
        #   helper blocks should typically be sorted before the `block` they are associated to (CHECK THIS),
        #   but multiple helper blocks may be necessary when `block` is a nested Block, i.e. if helpers need to
        #   be inserted at different stages of the DAG. Then we also need to do some non-trivial sorting when
        #   filling out the self.blocks attribute
        self.redirected = redirect_block
        self.blocks = [redirect_block, block]

        self.name = block.name + "_redirect"

        # now that it has a name, do Parent initialization
        Parent.__init__(self, [block, redirect_block])

        self.inputs = (redirect_block.inputs | block.inputs) - redirect_block.outputs
        self.outputs = (redirect_block.outputs | block.outputs) - redirect_block.inputs

        # Calculate what are the inputs and outputs of the Block objects underlying `self`, without
        # any of the redirecting blocks.
        if not isinstance(self.directed, Parent):
            self.inputs_directed = self.directed.inputs
            self.outputs_directed = self.directed.outputs
        else:
            inputs_directed, outputs_directed = OrderedSet({}), OrderedSet({})
            ps_checked = set({})
            for d in self.directed.descendants:
                # The descendant's parent's name (if it has one, o/w the descendant's name)
                p = self.directed.descendants[d]
                if p is None or p in ps_checked:
                    continue
                else:
                    ps_checked |= set(p)
                if hasattr(self.directed[p], "directed"):
                    inputs_directed |= self.directed[p].directed.inputs
                    outputs_directed |= self.directed[p].directed.outputs
                else:
                    inputs_directed |= self[d].inputs
                    outputs_directed |= self[d].outputs
            self.inputs_directed = inputs_directed - outputs_directed
            self.outputs_directed = outputs_directed

    def __repr__(self):
        return f"<RedirectedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], bypass_redirection=False, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""
        ss = calibration.copy()

        if not bypass_redirection:
            for block in self.blocks:
                # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
                inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
                outputs = block.steady_state(ss, dissolve=inner_dissolve, bypass_redirection=bypass_redirection, **kwargs)
                ss.update(outputs)
        else:
            inner_dissolve = [k for k in dissolve if self.descendants[k] == self.directed.name]
            outputs = self.directed.steady_state(ss, dissolve=inner_dissolve, bypass_redirection=bypass_redirection, **kwargs)
            ss.update(outputs)

        return ss

    # TODO: May not even need this! Just pass in bypass_redirection at the top-level block.steady_state call
    #   and check it against targets.
    def validate_steady_state(self, calibration, targets, ctol=1e-8, **kwargs):
        targets = {t: 0. for t in targets} if isinstance(targets, list) else targets
        ss_val = self.directed.steady_state(self.steady_state(calibration, **kwargs), bypass_redirection=True, **kwargs)
        if not np.all([np.abs(ss_val[k] - targets[k]) < ctol for k in targets]):
            raise RuntimeError(f"{self.directed.name}'s steady state does not hit the targets with the provided calibration")
