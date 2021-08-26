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

        # TODO: This all may not be necessary so long as there is a function that undoes the
        #   insertion of all of the redirect blocks and re-sorts the DAG (if necessary)
        # # Calculate what are the inputs and outputs of the Block objects underlying `self`, without
        # # any of the redirecting blocks.
        # # TODO: Think more carefully about evaluating a DAG with bypass_redirection, if the redirecting
        # #   blocks change the sort order of the DAG!!
        # if not isinstance(self.directed, Parent):
        #     self.inputs_directed = self.directed.inputs
        #     self.outputs_directed = self.directed.outputs
        # else:
        #     inputs_directed, outputs_directed = OrderedSet({}), OrderedSet({})
        #     ps_checked = set({})
        #     for d in self.directed.descendants:
        #         # The descendant's parent's name (if it has one, o/w the descendant's name)
        #         p = self.directed.descendants[d]
        #         if p is None or p in ps_checked:
        #             continue
        #         else:
        #             ps_checked |= set(p)
        #         if hasattr(self.directed[p], "directed"):
        #             inputs_directed |= self.directed[p].directed.inputs
        #             outputs_directed |= self.directed[p].directed.outputs
        #         else:
        #             inputs_directed |= self[d].inputs
        #             outputs_directed |= self[d].outputs
        #     self.inputs_directed = inputs_directed - outputs_directed
        #     self.outputs_directed = outputs_directed

    def __repr__(self):
        return f"<RedirectedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], bypass_redirection=False, **kwargs):
        """Evaluate a partial equilibrium steady state of the RedirectedBlock given a `calibration`"""
        if bypass_redirection:
            kwargs['dissolve'], kwargs['bypass_redirection'] = dissolve, bypass_redirection
            return self.directed.steady_state(calibration, **{k: v for k, v in kwargs.items() if k in self.directed.ss_valid_input_kwargs})
        else:
            ss = calibration.copy()
            for block in self.blocks:
                # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
                inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
                outputs = block.steady_state(ss, dissolve=inner_dissolve, **kwargs)
                # If we're not bypassing redirection, don't overwrite entries in ss populated by the redirect block
                ss.update(outputs) if block == self.redirected else ss.update(outputs.difference(ss))
            return ss
