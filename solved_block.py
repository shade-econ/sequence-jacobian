import numpy as np
import nonlinear
import jacobian as jac
from simple_block import simple


def solved(unknowns, targets, block_list=[]):
    """Creates SolvedBlocks. Can be applied in two ways, both of which return a SolvedBlock:
        - as @solved(unknowns=..., targets=...) decorator on a single SimpleBlock
        - as function solved(blocklist=..., unknowns=..., targets=...) where blocklist
            can be any list of blocks
    """

    if block_list:
        # ordinary call, not as decorator
        return SolvedBlock(block_list, unknowns, targets)
    else:
        # call as decorator, return function of function
        def solver(f):
            return SolvedBlock([simple(f)], unknowns, targets)
        return solver


class SolvedBlock:
    """SolvedBlocks are mini SHADE models embedded as blocks inside larger SHADE models.

    When creating them, we need to provide the basic ingredients of a SHADE model: the list of
    blocks comprising the model, the list on unknowns, and the list of targets.

    When we use .jac to ask for the Jacobian of a SolvedBlock, we are really solving for the 'G'
    matrices of the mini SHADE models, which then become the 'curlyJ' Jacobians of the block.

    Similarly, when we use .td to evaluate a SolvedBlock on a path, we are really solving for the
    nonlinear transition path such that all internal targets of the mini SHADE model are zero.
    """

    def __init__(self, block_list, unknowns, targets):
        self.block_list = block_list
        self.unknowns = unknowns
        self.targets = targets

        # need to have inputs and outputs!!!
        self.outputs = (set.union(*(b.outputs for b in block_list)) | set(self.unknowns)) - set(self.targets)
        self.inputs = set.union(*(b.inputs for b in block_list)) - self.outputs

    def ss(self, *args, **kwargs):
        # implementing steady-state for general solved block would be tantamount to providing 
        # a general method to solve for the steady state of any SHADE model
        # this is not our focus for now, since our methodological contributions are on dynamic side
        raise NotImplementedError('Cannot evaluate steady state for a SolvedBlock!')

    def td(self, ss, monotonic=False, returnindividual=False, noisy=False, **kwargs):
        # TODO: add H_U_factored caching of some kind
        # also, inefficient since we are repeatedly starting from the steady state, need option
        # to provide a guess (not a big deal with just SimpleBlocks, of course)
        return nonlinear.td_solve(ss, self.block_list, self.unknowns, self.targets, monotonic=monotonic, 
                                  returnindividual=returnindividual, noisy=noisy, **kwargs)
    
    def jac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False):
        # H_U_factored caching could be helpful here too
        return jac.get_G(self.block_list, shock_list, self.unknowns, self.targets,
                         T, ss, output_list, save=save, use_saved=use_saved)

    def ajac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False, Tpost=None):
        if Tpost is None:
            Tpost = 2*T
        return jac.get_G_asymptotic(self.block_list, shock_list, self.unknowns,
                self.targets, T, ss, output_list, save=save, use_saved=use_saved, Tpost=Tpost)
