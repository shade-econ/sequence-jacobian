"""Primitives to provide clarity and structure on blocks/models work"""

import numpy as np
from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
from copy import deepcopy

from .support.steady_state import provide_solver_default, solve_for_unknowns, compute_target_values
from .support.parent import Parent
from ..utilities import misc
from ..utilities.function import input_list, input_kwarg_list
from ..utilities.bijection import Bijection
from ..utilities.ordered_set import OrderedSet
from ..classes import SteadyStateDict, UserProvidedSS, ImpulseDict, JacobianDict, FactoredJacobianDict

Array = Any

class Block:
    """The abstract base class for all `Block` objects."""

    def __init__(self):
        self.M = Bijection({})

        self.steady_state_options = input_list(self._steady_state) - ['calibration', 'dissolve', 'evaluate_helpers', 'options']

        standard = ['ss', 'inputs', 'outputs', 'T', 'Js', 'options']
        self.impulse_nonlinear_options = input_list(self._impulse_nonlinear) - standard
        self.impulse_linear_options = input_list(self._impulse_linear) - standard
        self.jacobian_options = input_list(self._jacobian) - standard

        if isinstance(self, Parent):
            self.partial_jacobians_options = input_list(self._partial_jacobians) - standard
    
        self.ss_valid_input_kwargs = input_kwarg_list(self._steady_state)

    def inputs(self):
        pass

    def outputs(self):
        pass

    def steady_state(self, calibration: Union[SteadyStateDict, UserProvidedSS], 
                     dissolve: List[str] = [], evaluate_helpers: bool = False,
                     options: Dict[str, dict] = {}, **kwargs) -> SteadyStateDict:
        """Evaluate a partial equilibrium steady state of Block given a `calibration`."""
        # Special handling: 1) Find inputs/outputs of the Block w/o helpers blocks
        #                   2) Add all unknowns of dissolved blocks to inputs
        if not evaluate_helpers:
            inputs = self.inputs_orig.copy() if hasattr(self, "inputs_orig") else self.inputs.copy()
        else:
            inputs = self.inputs.copy()
        if isinstance(self, Parent):
            for k in dissolve:
                inputs |= self.get_attribute(k, 'unknowns').keys()

        calibration = SteadyStateDict(calibration)[inputs]
        own_options = self.get_options(options, kwargs, 'steady_state')
        if isinstance(self, Parent):
            return self.M @ self._steady_state(self.M.inv @ calibration, dissolve=dissolve, evaluate_helpers=evaluate_helpers, options=options, **own_options)
        else:
            return self.M @ self._steady_state(self.M.inv @ calibration, **own_options)

    def impulse_nonlinear(self, ss: SteadyStateDict, inputs: Union[Dict[str, Array], ImpulseDict],
                          outputs: Optional[List[str]] = None,
                          Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, non-linear impulse response of `outputs` to a set of shocks in `inputs`
        around a steady state `ss`."""
        own_options = self.get_options(options, kwargs, 'impulse_nonlinear')
        inputs = ImpulseDict(inputs)
        _, outputs = self.default_inputs_outputs(ss, inputs.keys(), outputs)
        
        # SolvedBlocks may use Js and may be nested in a CombinedBlock 
        if isinstance(self, Parent):
            return self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, Js, options, **own_options)
        else:
            return self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, **own_options)

    def impulse_linear(self, ss: SteadyStateDict, inputs: Union[Dict[str, Array], ImpulseDict], outputs: Optional[List[str]] = None, 
                       Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, linear impulse response of `outputs` to a set of shocks in `inputs`
        around a steady state `ss`."""
        own_options = self.get_options(options, kwargs, 'impulse_linear')
        inputs = ImpulseDict(inputs)
        _, outputs = self.default_inputs_outputs(ss, inputs.keys(), outputs)

        if isinstance(self, Parent):
            return self.M @ self._impulse_linear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, Js, options, **own_options)
        else:
            return self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, **own_options)

    def partial_jacobians(self, ss: SteadyStateDict, inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None,
                          T: Optional[int] = None, Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs):
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs
        
        # if you have a J for this block that already has everything you need, use it
        # TODO: add check for T, maybe look at verify_saved_jacobian for ideas?
        if (self.name in Js) and isinstance(Js[self.name], JacobianDict) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return {self.name: Js[self.name][outputs, inputs]}

        # if it's a leaf, just call Jacobian method, include if nonzero
        if not isinstance(self, Parent):
            own_options = self.get_options(options, kwargs, 'jacobian')
            jac = self.jacobian(ss, inputs, outputs, T, **own_options)
            return {self.name: jac} if jac else {}

        # otherwise call child method with remapping (and remap your own but none of the child Js)
        own_options = self.get_options(options, kwargs, 'partial_jacobians')
        partial = self._partial_jacobians(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T, Js, options, **own_options)
        if self.name in partial:
            partial[self.name] = self.M @ partial[self.name]
        return partial

    def jacobian(self, ss: SteadyStateDict, inputs: List[str], outputs: Optional[List[str]] = None,
                 T: Optional[int] = None, Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs) -> JacobianDict:
        """Calculate a partial equilibrium Jacobian to a set of `input` shocks at a steady state `ss`."""
        inputs, outputs = self.default_inputs_outputs(ss, inputs, outputs)
        own_options = self.get_options(options, kwargs, 'jacobian')

        # if you have a J for this block that has everything you need, use it
        if (self.name in Js) and isinstance(Js[self.name], JacobianDict) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return Js[self.name][outputs, inputs]
        
        # if it's a leaf, call Jacobian method, don't supply Js
        if not isinstance(self, Parent):
            # TODO should this be remapped?
            return self._jacobian(ss, inputs, outputs, T, **own_options)
        
        # otherwise remap own J (currently needed for SolvedBlock only)
        Js = Js.copy()
        if self.name in Js:
            Js[self.name] = self.M.inv @ Js[self.name]
        return self.M @ self._jacobian(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T=T, Js=Js, options=options, **own_options)

    def solve_steady_state(self, calibration: Dict[str, Union[Real, Array]],
                           unknowns: Dict[str, Union[Real, Tuple[Real, Real]]],
                           targets: Union[Array, Dict[str, Union[str, Real]]], dissolve: List = [],
                           helper_blocks: List = [], helper_targets: Dict = {},
                           solver: str = "", solver_kwargs: Dict = {},
                           ttol: float = 1e-12, ctol: float = 1e-9,
                           verbose: bool = False, check_consistency: bool = True,
                           constrained_method: str = "linear_continuation",
                           constrained_kwargs: Dict = {}, options: Dict[str, dict] = {}, **kwargs):
        """Evaluate a general equilibrium steady state of Block given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        if helper_blocks is not None:
            if helper_targets is None:
                raise ValueError("Must provide the dict of targets and their values that the `helper_blocks` solve"
                                 " in the `helper_targets` keyword argument.")
            else:
                from .support.steady_state import augment_dag_w_helper_blocks
                dag, ss, unknowns_to_solve, targets_to_solve = augment_dag_w_helper_blocks(self, calibration, unknowns,
                                                                                           targets, helper_blocks,
                                                                                           helper_targets)
        else:
            dag, ss, unknowns_to_solve, targets_to_solve = self, SteadyStateDict(calibration), unknowns, targets

        solver = solver if solver else provide_solver_default(unknowns)

        def residual(unknown_values, unknowns_keys=unknowns_to_solve.keys(), targets=targets_to_solve,
                     evaluate_helpers=True):
            ss.update(misc.smart_zip(unknowns_keys, unknown_values))
            kwargs['evaluate_helpers'] = evaluate_helpers
            ss.update(dag.steady_state(ss, dissolve=dissolve, options=options, **kwargs))
            return compute_target_values(targets, ss)

        unknowns_solved = solve_for_unknowns(residual, unknowns_to_solve, solver, solver_kwargs, tol=ttol, verbose=verbose,
                                             constrained_method=constrained_method, constrained_kwargs=constrained_kwargs)

        if helper_blocks and helper_targets and check_consistency:
            # Add in the unknowns solved analytically by helper blocks and re-evaluate the DAG without helpers
            unknowns_solved.update({k: ss[k] for k in unknowns if k not in unknowns_solved})
            cresid = np.max(abs(residual(unknowns_solved.values(), unknowns_keys=unknowns_solved.keys(),
                                         targets=targets, evaluate_helpers=False)))
            if cresid > ctol:
                raise RuntimeError(f"Target value residual {cresid} exceeds ctol specified for checking"
                                   f" the consistency of the DAG without redirection.")
        return ss

    def solve_impulse_nonlinear(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                                inputs: Union[Dict[str, Array], ImpulseDict], outputs: Optional[List[str]] = None,
                                Js: Dict[str, JacobianDict] = {},
                                tol: float = 1E-8, maxit: int = 30,
                                verbose: bool = True, options: Dict[str, dict] = {}, **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, non-linear impulse response to a set of shocks in `inputs` 
           around a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
           variables to be solved for and the `targets` that must hold in general equilibrium"""
        inputs = ImpulseDict(inputs)
        input_names, outputs = self.default_inputs_outputs(ss, inputs.keys(), outputs)
        unknowns, targets = OrderedSet(unknowns), OrderedSet(targets)
        T = inputs.T

        Js = self.partial_jacobians(ss, input_names | unknowns, (outputs | targets) - unknowns, T, Js, options, **kwargs)
        H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs)
        H_U_factored = FactoredJacobianDict(H_U, T)

        # Newton's method
        U = ImpulseDict({k: np.zeros(T) for k in unknowns})
        for it in range(maxit):
            results = self.impulse_nonlinear(ss, inputs | U, outputs | targets, Js, options, **kwargs)
            errors = {k: np.max(np.abs(results[k])) for k in targets}
            if verbose:
                print(f'On iteration {it}')
                for k in errors:
                    print(f'   max error for {k} is {errors[k]:.2E}')
            if all(v < tol for v in errors.values()):
                break
            else:
                U += H_U_factored.apply(results)
        else:
            raise ValueError(f'No convergence after {maxit} backward iterations!')

        return results | U

    def solve_impulse_linear(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                             inputs: Union[Dict[str, Array], ImpulseDict], outputs: Optional[List[str]] = None,
                             Js: Optional[Dict[str, JacobianDict]] = {}, options: Dict[str, dict] = {}, **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, linear impulse response to a set of shocks in `inputs`
           around a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
           variables to be solved for and the target conditions that must hold in general equilibrium"""
        inputs = ImpulseDict(inputs)
        input_names, outputs = self.default_inputs_outputs(ss, inputs.keys(), outputs)
        unknowns, targets = OrderedSet(unknowns), OrderedSet(targets)
        T = inputs.T

        Js = self.partial_jacobians(ss, input_names | unknowns, (outputs | targets) - unknowns, T, Js, options, **kwargs)

        H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs).pack(T)
        dH = self.impulse_linear(ss, inputs, targets, Js, options, **kwargs).pack()
        dU = ImpulseDict.unpack(-np.linalg.solve(H_U, dH), unknowns, T)

        return self.impulse_linear(ss, dU | inputs, outputs, Js, options, **kwargs)

    def solve_jacobian(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                       inputs: List[str], outputs: Optional[List[str]] = None, T: Optional[int] = None,
                       Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs) -> JacobianDict:
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        # TODO: do we really want this? is T just optional because we want it to come after outputs in docstring?
        if T is None:
            T = 300

        inputs, outputs = self.default_inputs_outputs(ss, inputs, outputs)
        unknowns, targets = OrderedSet(unknowns), OrderedSet(targets)

        Js = self.partial_jacobians(ss, inputs | unknowns, (outputs | targets) - unknowns, T, Js, options, **kwargs)
        
        H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs).pack(T)
        H_Z = self.jacobian(ss, inputs, targets, T, Js, options, **kwargs).pack(T)
        U_Z = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, inputs, T)

        from sequence_jacobian import combine
        self_with_unknowns = combine([U_Z, self])
        return self_with_unknowns.jacobian(ss, inputs, unknowns | outputs, T, Js, options, **kwargs)

    def solved(self, unknowns, targets, name=None, solver=None, solver_kwargs=None):
        if name is None:
            name = self.name + "_solved"
        from .solved_block import SolvedBlock
        return SolvedBlock(self, name, unknowns, targets, solver, solver_kwargs)

    def remap(self, map):
        other = deepcopy(self)
        other.M = self.M @ Bijection(map)
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
        return OrderedSet(inputs), OrderedSet(outputs)

    def get_options(self, options: dict, kwargs, method):
        if self.name in options:
            options = {**options[self.name], **kwargs}
        else:
            options = kwargs
    
        return {k: v for k, v in options.items() if k in self.getattr(method + "_options")}

    solve_jacobian_options = OrderedSet([])
    solve_impulse_linear_options = OrderedSet([])
    solve_impulse_nonlinear_options = OrderedSet(['tol', 'maxit', 'verbose'])
    solve_steady_state_options = (input_list(solve_steady_state) - 
                    ['self', 'calibration', 'unknowns', 'targets', 'dissolve', 'helper_blocks', 'helper_targets'])
    