"""Primitives to provide clarity and structure on blocks/models work"""

import numpy as np
import abc
from numbers import Real
from typing import Dict, Union

from . import utilities as utils


# TODO: Refactor .ss, .td, and .jac methods for SimpleBlock and HetBlock to be cleaner so they can be interpreted from
#   this more abstract representation of what canonical "Block"-like behavior should be
class Block(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, f):
        self.f = f
        self.inputs = set(utils.misc.input_list(f))
        self.outputs = set(utils.misc.output_list(f))

    @abc.abstractmethod
    def ss(self, *ss_args, **ss_kwargs) -> Dict[str, Union[Real, np.ndarray]]:
        """Call the block's function attribute `.f` on the pre-processed steady state (keyword) arguments,
        ensuring that any time displacements will be ignored when `.f` is called.
        See blocks.support.simple_displacement for an example of how SimpleBlocks do this pre-processing."""
        return self.f(*ss_args, **ss_kwargs)

    @abc.abstractmethod
    def td(self, ss: Dict[str, Real], shock_paths: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def jac(self, ss, shock_list, T):
        pass
