import warnings

from ..primitives import Block
from ..blocks.simple_block import simple
from ..blocks.parent import Parent
from ..utilities import graph

from ..jacobian.classes import JacobianDict, FactoredJacobianDict


def solved(unknowns, targets, solver=None, solver_kwargs={}, name=""):
    """Convenience @solved(unknowns=..., targets=...) decorator on a single SimpleBlock"""
    # call as decorator, return function of function
    def singleton_solved_block(f):
        return SolvedBlock(simple(f).rename(f.__name__ + '_inner'), f.__name__, unknowns, targets, solver=solver, solver_kwargs=solver_kwargs)
    return singleton_solved_block


class SolvedBlock(Block, Parent):
    """SolvedBlocks are mini SHADE models embedded as blocks inside larger SHADE models.

    When creating them, we need to provide the basic ingredients of a SHADE model: the list of
    blocks comprising the model, the list on unknowns, and the list of targets.

    When we use .jac to ask for the Jacobian of a SolvedBlock, we are really solving for the 'G'
    matrices of the mini SHADE models, which then become the 'curlyJ' Jacobians of the block.

    Similarly, when we use .td to evaluate a SolvedBlock on a path, we are really solving for the
    nonlinear transition path such that all internal targets of the mini SHADE model are zero.
    """

    def __init__(self, block: Block, name, unknowns, targets, solver=None, solver_kwargs={}):
        super().__init__()

        self.block = block
        self.name = name
        self.unknowns = unknowns
        self.targets = targets
        self.solver = solver
        self.solver_kwargs = solver_kwargs

        Parent.__init__(self, [self.block])

        # validate unknowns and targets
        if not len(unknowns) == len(targets):
            raise ValueError(f'Unknowns {set(unknowns)} and targets {set(targets)} different sizes in SolvedBlock {name}')
        if not set(unknowns) <= block.inputs:
            raise ValueError(f'Unknowns has element {set(unknowns) - block.inputs} not in inputs in SolvedBlock {name}')
        if not set(targets) <= block.outputs:
            raise ValueError(f'Targets has element {set(targets) - block.outputs} not in outputs in SolvedBlock {name}')

        # what are overall outputs and inputs?
        self.outputs = block.outputs | set(unknowns)
        self.inputs = block.inputs - set(unknowns)

    def __repr__(self):
        return f"<SolvedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve=[], unknowns=None, solver=None, ttol=1e-9, ctol=1e-9, verbose=False):
        if self.name in dissolve:
            solver = "solved"
            unknowns = {k: v for k, v in calibration.items() if k in self.unknowns}

        # Allow override of unknowns/solver, if one wants to evaluate the SolvedBlock at a particular set of
        # unknown values akin to the steady_state method of Block
        if unknowns is None:
            unknowns = self.unknowns
        if solver is None:
            solver = self.solver

        return self.block.solve_steady_state(calibration, unknowns, self.targets, solver=solver,
                                          ttol=ttol, ctol=ctol, verbose=verbose)

    def _impulse_nonlinear(self, ss, exogenous=None, monotonic=False, Js=None, returnindividual=False, verbose=False):
        return self.block.solve_impulse_nonlinear(ss, exogenous=exogenous,
                                               unknowns=list(self.unknowns.keys()), Js=Js,
                                               targets=self.targets if isinstance(self.targets, list) else list(self.targets.keys()),
                                               monotonic=monotonic, returnindividual=returnindividual, verbose=verbose)

    def _impulse_linear(self, ss, exogenous, T=None, Js=None):
        return self.block.solve_impulse_linear(ss, exogenous=exogenous, unknowns=list(self.unknowns.keys()),
                                            targets=self.targets if isinstance(self.targets, list) else list(self.targets.keys()),
                                            T=T, Js=Js)

    def _jacobian(self, ss, inputs, outputs, T, Js):
        return self.block.solve_jacobian(ss, set(self.unknowns), set(self.targets), inputs, outputs, T, Js)[outputs]

    def _partial_jacobians(self, ss, inputs, outputs, T, Js={}):
        # call it on the child first
        inner_Js = self.block.partial_jacobians(ss, inputs=(set(self.unknowns) | inputs),
                                                outputs=(set(self.targets) | outputs), T=T, Js=Js)

        # with these inner Js, also compute H_U and factorize
        H_U = self.block.jacobian(ss, inputs=self.unknowns.keys(), outputs=self.targets.keys(), T=T, Js=inner_Js)
        H_U_factored = FactoredJacobianDict(H_U, T)

        return {**inner_Js, self.name: H_U_factored}
