"""Primitives to provide clarity and structure on blocks/models work"""

import abc
from abc import ABCMeta as NativeABCMeta
from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
from copy import deepcopy

from .steady_state.drivers import steady_state as ss
from .steady_state.support import provide_solver_default
from .nonlinear import td_solve
from .jacobian.drivers import get_impulse, get_G
from .steady_state.classes import SteadyStateDict, UserProvidedSS, make_steadystatedict
from .jacobian.classes import JacobianDict
from .blocks.support.impulse import ImpulseDict
from .blocks.support.bijection import Bijection
from .blocks.parent import Parent
from .utilities import misc

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
                "Cannot instantiate abstract class `{}` with"
                " abstract attributes: `{}`.\n"
                "Define concrete implementations of these attributes in the child class prior to preceding.".format(
                    cls.__name__,
                    '`, `'.join(abstract_attributes)
                )
            )
        return instance
###############################################################################


class Block(abc.ABC, metaclass=ABCMeta):
    """The abstract base class for all `Block` objects."""

    #@abc.abstractmethod
    def __init__(self):
        self.M = Bijection({})
        self.ss_valid_input_kwargs = misc.input_kwarg_list(self._steady_state)

    @abstract_attribute
    def inputs(self):
        pass

    @abstract_attribute
    def outputs(self):
        pass

    def steady_state(self, calibration: Union[SteadyStateDict, UserProvidedSS], 
                     dissolve: Optional[List[str]] = [], **kwargs) -> SteadyStateDict:
        """Evaluate a partial equilibrium steady state of Block given a `calibration`."""
        # special handling: add all unknowns of dissolved blocks to inputs
        inputs = self.inputs.copy()
        if isinstance(self, Parent):
            for k in dissolve:
                inputs |= self.get_attribute(k, 'unknowns').keys()

        calibration = make_steadystatedict(calibration)[inputs]
        kwargs['dissolve'] = dissolve

        return self.M @ self._steady_state(self.M.inv @ calibration, **{k: v for k, v in kwargs.items() if k in self.ss_valid_input_kwargs})

    def impulse_nonlinear(self, ss: SteadyStateDict,
                          exogenous: Dict[str, Array], **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, non-linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`."""
        return self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ exogenous, **kwargs)

    def impulse_linear(self, ss: SteadyStateDict,
                       exogenous: Dict[str, Array], **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`."""
        return self.M @ self._impulse_linear(self.M.inv @ ss, self.M.inv @ exogenous, **kwargs)

    def partial_jacobians(self, ss, inputs=None, T=None, Js={}):
        # TODO: annotate signature
        if inputs is None:
            inputs = self.inputs
        
        # if you have a J for this block that already has everything you need, use it
        if (self.name in Js) and (inputs <= Js[self.name].inputs) and (self.outputs == Js[self.name].outputs):
            return Js[self.name][:, inputs]

        # if it's a leaf, just call Jacobian method, include if nonzero
        if not isinstance(self, Parent):
            jac = self.jacobian(ss, inputs, T)
            return {self.name: jac} if jac else {}

        # otherwise call child method with remapping (but not for Js, which are not remapped to top level)
        return self.M @ self._partial_jacobians(self.M.inv @ ss, self.M.inv @ inputs, T, Js)

    def jacobian(self, ss: SteadyStateDict,
                 exogenous: List[str],
                 T: Optional[int] = None, **kwargs) -> JacobianDict:
        """Calculate a partial equilibrium Jacobian to a set of `exogenous` shocks at a steady state `ss`."""
        return self.M @ self._jacobian(self.M.inv @ ss, self.M.inv @ exogenous, T=T, **kwargs)

    def solve_steady_state(self, calibration: Dict[str, Union[Real, Array]],
                           unknowns: Dict[str, Union[Real, Tuple[Real, Real]]],
                           targets: Union[Array, Dict[str, Union[str, Real]]],
                           solver: Optional[str] = "", **kwargs) -> SteadyStateDict:
        """Evaluate a general equilibrium steady state of Block given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        solver = solver if solver else provide_solver_default(unknowns)
        return ss(blocks, calibration, unknowns, targets, solver=solver, **kwargs)

    def solve_impulse_nonlinear(self, ss: Dict[str, Union[Real, Array]],
                                exogenous: Dict[str, Array],
                                unknowns: List[str], targets: List[str],
                                Js: Optional[Dict[str, JacobianDict]] = None,
                                **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, non-linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        irf_nonlin_gen_eq = td_solve(blocks, ss,
                                     exogenous={k: v for k, v in exogenous.items()},
                                     unknowns=unknowns, targets=targets, Js=Js, **kwargs)
        return ImpulseDict(irf_nonlin_gen_eq)

    def solve_impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                             exogenous: Dict[str, Array],
                             unknowns: List[str], targets: List[str],
                             T: Optional[int] = None,
                             Js: Optional[Dict[str, JacobianDict]] = None,
                             **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        irf_lin_gen_eq = get_impulse(blocks, exogenous, unknowns, targets, T=T, ss=ss, Js=Js, **kwargs)
        return ImpulseDict(irf_lin_gen_eq)

    def solve_jacobian(self, ss: Dict[str, Union[Real, Array]],
                       exogenous: List[str],
                       unknowns: List[str], targets: List[str],
                       T: Optional[int] = None,
                       Js: Optional[Dict[str, JacobianDict]] = None,
                       **kwargs) -> JacobianDict:
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        return get_G(blocks, exogenous, unknowns, targets, T=T, ss=ss, Js=Js, **kwargs)

    def remap(self, map):
        other = deepcopy(self)
        other.M = self.M @ Bijection(map)
        # TODO: maybe we want to have an ._inputs and ._outputs that never changes, so that it can be used internally?
        other.inputs = other.M @ self.inputs
        other.outputs = other.M @ self.outputs
        if hasattr(self, 'input_list'):
            other.input_list = other.M @ self.input_list
        if hasattr(self, 'output_list'):
            other.output_list = other.M @ self.output_list
        if hasattr(self, 'non_back_iter_outputs'):
            other.non_back_iter_outputs = other.M @ self.non_back_iter_outputs
        return other

    def rename(self, name):
        renamed = deepcopy(self)
        renamed.name = name
        return renamed
