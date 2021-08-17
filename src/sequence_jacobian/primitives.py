"""Primitives to provide clarity and structure on blocks/models work"""

import abc
import numpy as np
from abc import ABCMeta as NativeABCMeta
from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
from copy import deepcopy

from sequence_jacobian.utilities.ordered_set import OrderedSet

from .steady_state.drivers import steady_state as ss
from .steady_state.support import provide_solver_default
from .nonlinear import td_solve
from .jacobian.drivers import get_impulse, get_G
from .steady_state.classes import SteadyStateDict, UserProvidedSS
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

        calibration = SteadyStateDict(calibration)[inputs]
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

    def partial_jacobians(self, ss, inputs=None, outputs=None, T=None, Js={}):
        # TODO: annotate signature
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs
        inputs, outputs = set(inputs), set(outputs)
        
        # if you have a J for this block that already has everything you need, use it
        # TODO: add check for T,  maybe look at verify_saved_jacobian for ideas?
        if (self.name in Js) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return {self.name: Js[self.name][outputs, inputs]}

        # if it's a leaf, just call Jacobian method, include if nonzero
        if not isinstance(self, Parent):
            jac = self.jacobian(ss, inputs, outputs, T)
            return {self.name: jac} if jac else {}

        # otherwise call child method with remapping (and remap your own but none of the child Js)
        partial = self._partial_jacobians(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T, Js)
        if self.name in partial:
            partial[self.name] = self.M @ partial[self.name]
        return partial

    def jacobian(self, ss: SteadyStateDict, inputs: List[str], outputs: Optional[List[str]] = None,
                 T: Optional[int] = None, Js={}) -> JacobianDict:
        """Calculate a partial equilibrium Jacobian to a set of `input` shocks at a steady state `ss`."""
        inputs, outputs = self.default_inputs_outputs(ss, inputs, outputs)
        inputs, outputs = OrderedSet(inputs), OrderedSet(outputs)

        # if you have a J for this block that has everything you need, use it
        if (self.name in Js) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return Js[self.name][outputs, inputs]
        
        # if it's a leaf, call Jacobian method, don't supply Js
        if not isinstance(self, Parent):
            # TODO should this be remapped?
            return self._jacobian(ss, inputs, outputs, T)
        
        # otherwise remap own J (currently needed for SolvedBlock only)
        Js = Js.copy()
        if self.name in Js:
            Js[self.name] = self.M.inv @ Js[self.name]
        return self.M @ self._jacobian(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T=T, Js=Js)

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
                                Js: Optional[Dict[str, JacobianDict]] = {},
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
                             Js: Optional[Dict[str, JacobianDict]] = {},
                             **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, linear impulse response to a set of `exogenous` shocks
        from a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        blocks = self.blocks if hasattr(self, "blocks") else [self]
        irf_lin_gen_eq = get_impulse(blocks, exogenous, unknowns, targets, T=T, ss=ss, Js=Js, **kwargs)
        return ImpulseDict(irf_lin_gen_eq)

    def solve_jacobian(self, ss: Dict[str, Union[Real, Array]], unknowns: List[str], targets: List[str],
                       inputs: List[str], outputs: Optional[List[str]] = None, T: Optional[int] = None,
                       Js: Optional[Dict[str, JacobianDict]] = {},
                       **kwargs) -> JacobianDict:
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        # TODO: do we really want this? is T just optional because we want it to come after outputs in docstring?
        if T is None:
            T = 300

        inputs, outputs = self.default_inputs_outputs(ss, inputs, outputs)
        inputs, unknowns, targets = list(inputs), list(unknowns), list(targets)

        Js = self.partial_jacobians(ss, set(inputs) | set(unknowns), (set(outputs) | set(targets)) - set(unknowns), T, Js)
        
        H_U = self.jacobian(ss, unknowns, targets, T, Js).pack(T)
        H_Z = self.jacobian(ss, inputs, targets, T, Js).pack(T)
        U_Z = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, inputs, T)

        from . import combine
        self_with_unknowns = combine([U_Z, self])
        return self_with_unknowns.jacobian(ss, inputs, set(unknowns) | set(outputs), T, Js)

    def solved(self, unknowns, targets, name=None, solver=None, solver_kwargs=None):
        if name is None:
            name = self.name + "_solved"
        from .blocks.solved_block import SolvedBlock
        return SolvedBlock(self, name, unknowns, targets, solver, solver_kwargs)

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

    def default_inputs_outputs(self, ss: SteadyStateDict, inputs, outputs):
        # TODO: there should be checks to make sure you don't ask for multidimensional stuff for Jacobians?
        # should you be allowed to ask for it (even if not default) for impulses?
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs - ss._vector_valued()
        return inputs, outputs
