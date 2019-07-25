import numpy as np
import nonlinear
import jacobian as jac
from simple_block import simple


def solved(exogenous, unknowns, targets, block_list=[]):
    if block_list:
        # ordinary call, not as decorator
        return SolvedBlock(block_list, exogenous, unknowns, targets)
    else:
        # call as decorator, return function of function
        def solver(f):
            return SolvedBlock([simple(f)], exogenous, unknowns, targets)
        return solver


class SolvedBlock:
    def __init__(self, block_list, exogenous, unknowns, targets):
        self.block_list = block_list
        self.exogenous = exogenous
        self.unknowns = unknowns
        self.targets = targets

        # need to have inputs and outputs!!!
        self.outputs = (set.union(*(b.outputs for b in block_list)) | set(self.unknowns)) - set(self.targets)
        self.inputs = set.union(*(b.inputs for b in block_list)) - self.outputs

    def ss(self, *args, **kwargs):
        raise NotImplementedError('Cannot evaluate steady state for a SolvedBlock!')

    def td(self, ss, monotonic=False, returnindividual=False, noisy=False, **kwargs):
        # TODO: add H_U_factored caching of some kind
        return nonlinear.td_solve(ss, self.block_list, self.unknowns, self.targets, monotonic=monotonic, 
                                  returnindividual=returnindividual, noisy=noisy, **kwargs)
    
    def jac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False):
        # TODO: consider whether we want the other arguments of HetBlock.jac here!
        # also H_U_factored caching could be helpful here too
        # shock_list better be subset of exogenous!
        return jac.get_G(self.block_list, shock_list, self.unknowns, self.targets,
                         T, ss, output_list, save=save, use_saved=use_saved)

    def ajac(self, ss, T, shock_list, output_list=None, save=False, use_saved=False, Tpost=None):
        # test to see if this works?
        if Tpost is None:
            Tpost = 2*T
        return jac.get_G_asymptotic(self.block_list, shock_list, self.unknowns,
                self.targets, T, ss, output_list, save=save, use_saved=use_saved, Tpost=Tpost)
