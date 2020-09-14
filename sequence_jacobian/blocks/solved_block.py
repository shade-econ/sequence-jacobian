from .. import nonlinear
from .. import jacobian as jac
from ..steady_state import steady_state
from ..blocks.simple_block import simple


def solved(unknowns, targets, block_list=[], solver=None, solver_kwargs={}):
    """Creates SolvedBlocks. Can be applied in two ways, both of which return a SolvedBlock:
        - as @solved(unknowns=..., targets=...) decorator on a single SimpleBlock
        - as function solved(blocklist=..., unknowns=..., targets=...) where blocklist
            can be any list of blocks
    """

    if block_list:
        # ordinary call, not as decorator
        return SolvedBlock(block_list, unknowns, targets, solver=solver, solver_kwargs=solver_kwargs)
    else:
        # call as decorator, return function of function
        def singleton_solved_block(f):
            return SolvedBlock([simple(f)], unknowns, targets, solver=solver, solver_kwargs=solver_kwargs)
        return singleton_solved_block


class SolvedBlock:
    """SolvedBlocks are mini SHADE models embedded as blocks inside larger SHADE models.

    When creating them, we need to provide the basic ingredients of a SHADE model: the list of
    blocks comprising the model, the list on unknowns, and the list of targets.

    When we use .jac to ask for the Jacobian of a SolvedBlock, we are really solving for the 'G'
    matrices of the mini SHADE models, which then become the 'curlyJ' Jacobians of the block.

    Similarly, when we use .td to evaluate a SolvedBlock on a path, we are really solving for the
    nonlinear transition path such that all internal targets of the mini SHADE model are zero.
    """

    def __init__(self, block_list, unknowns, targets, solver=None, solver_kwargs={}):
        self.block_list = block_list
        self.unknowns = unknowns
        self.targets = targets
        self.solver = solver
        self.solver_kwargs = solver_kwargs

        # need to have inputs and outputs!!!
        self.outputs = (set.union(*(b.outputs for b in block_list)) | set(list(self.unknowns.keys()))) - set(self.targets)
        self.inputs = set.union(*(b.inputs for b in block_list)) - self.outputs

    def ss(self, consistency_check=True, ttol=1e-9, ctol=1e-9, verbose=False, **calibration):
        if self.solver is None:
            raise RuntimeError("Cannot call the ss method on this SolvedBlock without specifying a solver.")
        else:
            return steady_state(self.block_list, calibration, self.unknowns, self.targets,
                                consistency_check=consistency_check, ttol=ttol, ctol=ctol, verbose=verbose,
                                solver=self.solver, **self.solver_kwargs)

    def td(self, ss, monotonic=False, returnindividual=False, verbose=False, **kwargs):
        # TODO: add H_U_factored caching of some kind
        # also, inefficient since we are repeatedly starting from the steady state, need option
        # to provide a guess (not a big deal with just SimpleBlocks, of course)
        return nonlinear.td_solve(ss, self.block_list, list(self.unknowns.keys()), self.targets, monotonic=monotonic,
                                  returnindividual=returnindividual, verbose=verbose, **kwargs)
    
    def jac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False):
        relevant_shocks = [i for i in self.inputs if i in shock_list]

        # H_U_factored caching could be helpful here too
        return jac.get_G(self.block_list, relevant_shocks, list(self.unknowns.keys()), self.targets,
                         T, ss, output_list, save=save, use_saved=use_saved)

    def ajac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False, Tpost=None):
        relevant_shocks = [i for i in self.inputs if i in shock_list]

        if Tpost is None:
            Tpost = 2*T
        return jac.get_G_asymptotic(self.block_list, relevant_shocks, list(self.unknowns.keys()),
                                    self.targets, T, ss, output_list, save=save, use_saved=use_saved, Tpost=Tpost)
