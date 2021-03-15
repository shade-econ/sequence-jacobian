"""Primitives to provide clarity and structure on blocks/models work"""

import abc
from abc import ABCMeta as NativeABCMeta
from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
import numpy as np

from .steady_state.drivers import steady_state
from .steady_state.support import provide_solver_default
from .nonlinear import td_solve
from .jacobian.drivers import get_impulse, get_G
from .jacobian.classes import JacobianDict
from .blocks.support.impulse import ImpulseDict

# Basic types
Array = Any


###############################################################################
# Because abc doesn't implement "abstract attribute"s
# https://stackoverflow.com/questions/23831510/abstract-attribute-not-property
class DummyAttribute:
    pass


def abstract_attribute(obj=None):
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstract_attribute__ = True
    return obj


class ABCMeta(NativeABCMeta):

    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(abstract_attributes)
                )
            )
        return instance
###############################################################################


class Block(abc.ABC, metaclass=ABCMeta):
    """The abstract base class for all `Block` objects."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abstract_attribute
    def inputs(self):
        pass

    @abstract_attribute
    def outputs(self):
        pass

    # Typing information is purely to inform future user-developed `Block` sub-classes to enforce a canonical
    # input and output argument structure
    @abc.abstractmethod
    def steady_state(self, *ss_args, **ss_kwargs) -> Dict[str, Union[Real, Array]]:
        pass

    @abc.abstractmethod
    def impulse_nonlinear(self, ss: Dict[str, Union[Real, Array]],
                          exogenous: Dict[str, Array], **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                       exogenous: Dict[str, Array], **kwargs) -> Dict[str, Array]:
        pass

    @abc.abstractmethod
    def jacobian(self, ss: Dict[str, Union[Real, Array]], exogenous=None, T=None, **kwargs) -> JacobianDict:
        pass

    def solve_steady_state(self, calibration: Dict[str, Union[Real, Array]],
                           unknowns: Dict[str, Union[Real, Tuple[Real, Real]]],
                           targets: Union[Array, Dict[str, Union[str, Real]]],
                           solver: Optional[str] = "", **kwargs) -> Dict[str, Union[Real, Array]]:
        """Evaluate a general equilibrium steady state of Block given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        blocks = self.blocks_w_helpers if hasattr(self, "blocks_w_helpers") else [self]
        solver = solver if solver else provide_solver_default(unknowns)
        return steady_state(blocks, calibration, unknowns, targets, solver=solver, **kwargs)

    def solve_impulse_nonlinear(self, ss: Dict[str, Union[Real, Array]],
                                exogenous: Dict[str, Array],
                                unknowns: List[str], targets: List[str],
                                **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, non-linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        irf_nonlin_gen_eq = td_solve(blocks, ss,
                                     exogenous={k: ss[k] + v for k, v in exogenous.items()},
                                     unknowns=unknowns, targets=targets, **kwargs)
        return ImpulseDict(irf_nonlin_gen_eq, ss)

    def solve_impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                             exogenous: Dict[str, Array],
                             unknowns: List[str], targets: List[str],
                             T: Optional[int] = None,
                             **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        irf_lin_gen_eq = get_impulse(blocks, exogenous, unknowns, targets, T=T, ss=ss, **kwargs)
        return ImpulseDict(irf_lin_gen_eq, ss)

    def solve_jacobian(self, ss: Dict[str, Union[Real, Array]],
                       exogenous: List[str],
                       unknowns: List[str], targets: List[str],
                       T: Optional[int] = None, **kwargs) -> JacobianDict:
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        return get_G(blocks, exogenous, unknowns, targets, T=T, ss=ss, **kwargs)
