"""Low-level tools/classes for analyzing sequence-jacobian model DAGs to support debugging"""

import numpy as np
import xarray as xr
from collections.abc import Iterable

from ..utilities import graph


class BlockIONetwork:
    """
    A 3-d axis-labeled DataArray (blocks x inputs x outputs), which allows for the analysis of the input-output
    structure of a DAG.
    """
    def __init__(self, blocks):
        topsorted, inset, outset = graph.block_sort(blocks, return_io=True)
        self.blocks = {b.name: b for b in blocks}
        self.blocks_names = list(self.blocks.keys())
        self.blocks_as_list = list(self.blocks.values())
        self.var_names = list(inset.union(outset))
        self.darray = xr.DataArray(np.zeros((len(blocks), len(self.var_names), len(self.var_names))),
                                   coords=[self.blocks_names, self.var_names, self.var_names],
                                   dims=["blocks", "inputs", "outputs"])

    def __repr__(self):
        return self.darray.__repr__()

    # User-facing "print" methods
    def print_block_links(self, block_name):
        print(f" {block_name}")
        print(" " + "-" * (len(block_name)))

        link_inds = np.nonzero(self._subset_by_block(block_name)).data
        for i in range(np.shape(link_inds)[0]):
            i_ind, o_ind = link_inds[:, i]
            i_var = str(self._subset_by_block(block_name).coords["inputs"][i_ind].data)
            o_var = str(self._subset_by_block(block_name).coords["outputs"][o_ind].data)
            print(f" {i_var} -> {o_var}")


    # User-facing "analysis" methods
    def record_input_variable_paths(self, inputs_to_be_recorded, block_input_args):
        """
        Updates the VariableIONetwork with the paths that a set of inputs influence, as they propagate through the DAG

        Parameters
        ----------
        inputs_to_be_recorded: `list(str)`
            A list of input variable names, whose paths will be traced and recorded in the VariableIONetwork
        block_input_args: `dict`
            A dict of variable/parameter names and values (typically from the steady state of the model) on which,
            a block can perform a valid evaluation
        """
        block_inds_sorted = graph.block_sort(self.blocks_as_list)
        for input_var in inputs_to_be_recorded:
            all_input_vars = set(input_var)
            for ib in block_inds_sorted:
                ib_input_args = {k: v for k, v in block_input_args.items() if k in self.blocks_as_list[ib].inputs}
                # This extra step is needed because some arguments required for calling .jac on
                # HetBlock and SolvedBlock are not a part of .inputs
                ib_input_args.update(**self.blocks_as_list[ib].ss(**ib_input_args))
                io_links = find_io_links(self.blocks_as_list[ib], list(all_input_vars), ib_input_args)
                if io_links:
                    self._record_io_links(self.blocks_names[ib], io_links)
                    # Need to also track the paths of outputs which could be intermediate inputs further down the DAG
                    all_input_vars = all_input_vars.union(set(io_links.keys()))

    # Analysis support methods
    def _subset_by_block(self, block_name):
        return self.darray.loc[block_name, list(self.blocks[block_name].inputs), list(self.blocks[block_name].outputs)]

    def _subset_by_vars(self, vars_names):
        if isinstance(vars_names, Iterable):
            return self.darray.loc[[b.name for b in self.blocks.values() if np.any(v in b.inputs for v in vars_names)],
                                    vars_names, :]
        else:
            return self.darray.loc[[b.name for b in self.blocks.values() if vars_names in b.inputs], vars_names, :]

    def _record_io_links(self, block_name, io_links):
        for o, i in io_links.items():
            self.darray.loc[block_name, i, o] = 1.


def find_io_links(block, input_args, block_input_args):
    """
    For a given `block`, see which output arguments the input argument `input_args` affects

    Parameters
    ----------
    block: `Block` object
        One of the various kinds of `Block` objects (`SimpleBlock`, `HetBlock`, etc.)
    input_args: `str` or `list(str)`
        The input arguments, whose paths through the block to the output variables we want to see
    block_input_args: `dict{str: num}`
        The rest of the input arguments required to evaluate the block's Jacobian

    Returns
    -------
    links: `dict{str: list(str)}`
        A dict with *output arguments* as keys, and the input arguments that affect it as values
    """
    J = block.jac(ss=block_input_args, T=2, shock_list=input_args)
    links = {}
    for o in J.outputs:
        links[o] = list(J.nesteddict[o].keys())
    return links

