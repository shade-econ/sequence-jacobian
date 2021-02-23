"""Primitives to provide clarity and structure on blocks/models work"""

import abc
from numbers import Real
from typing import Dict, Union, Tuple, Optional, List

from .base import Array
from .jacobian.classes import JacobianDict
from .steady_state.drivers import steady_state
from .nonlinear import td_solve
from .jacobian.drivers import get_G


class Block(abc.ABC):
    """The abstract base class for all `Block` objects."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def inputs(self):
        pass

    @property
    @abc.abstractmethod
    def outputs(self):
        pass

    # Typing information is purely to inform future user-developed `Block` sub-classes to enforce a canonical
    # input and output argument structure
    @abc.abstractmethod
    def steady_state(self, *ss_args, **ss_kwargs) -> Dict[str, Union[Real, Array]]:
        pass

    @abc.abstractmethod
    def impulse_nonlinear(self, ss: Dict[str, Union[Real, Array]],
                          shocked_paths: Dict[str, Array], **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                       shocked_paths: Dict[str, Array], **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def jacobian(self, ss: Dict[str, Union[Real, Array]], exogenous=None, T=None, **kwargs) -> JacobianDict:
        pass

    @abc.abstractmethod
    def solve_steady_state(self, calibration: Dict[str, Union[Real, Array]],
                           unknowns: Dict[str, Union[Real, Tuple[Real, Real]]],
                           targets: Union[Array, Dict[str, Union[str, Real]]],
                           solver: Optional[str] = "", **kwargs) -> Dict[str, Union[Real, Array]]:
        # What is a consistent interface for passing things to steady_state?
        # Should change steady_state from expecting a block_list to a *single* Block object
        # duck-type by checking for attr ".blocks", to signify if is a CombinedBlock
        # Should try to figure out a nicer way to pass variable kwargs to eval_block_ss to clean that function up

        # Also should change td_solve and get_G to also only expect *single* Block objects, with a deprecation
        # allowing for lists to be passed, which will then automatically build those lists into CombinedBlocks
        pass

    @abc.abstractmethod
    def solve_impulse_nonlinear(self, ss: Dict[str, Union[Real, Array]],
                                exogenous: Dict[str, Array],
                                unknowns: List[str], targets: List[str],
                                in_deviations: Optional[bool] = True, **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def solve_impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                             exogenous: Dict[str, Array],
                             unknowns: List[str], targets: List[str],
                             T: Optional[int] = None, in_deviations: Optional[bool] = True,
                             **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def solve_jacobian(self, ss: Dict[str, Union[Real, Array]],
                       exogenous: List[str],
                       unknowns: List[str], targets: List[str],
                       T: Optional[int] = None, **kwargs) -> Dict[str, Array]:
        pass
