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
        print(f" Links in {block_name}")
        print(" " + "-" * (len(f" Links in {block_name}")))

        link_inds = np.nonzero(self._subset_by_block(block_name)).data
        for i in range(np.shape(link_inds)[0]):
            i_ind, o_ind = link_inds[:, i]
            i_var = str(self._subset_by_block(block_name).coords["inputs"][i_ind].data)
            o_var = str(self._subset_by_block(block_name).coords["outputs"][o_ind].data)
            print(f" {i_var} -> {o_var}")
        print("")  # To break lines

    def print_var_links(self, var_name, calibration=None, ignore_helpers=True):
        print(f" Links from {var_name}")
        print(" " + "-" * (len(f" Links for {var_name}")))

        links = self.find_var_links(var_name, calibration=calibration, ignore_helpers=ignore_helpers)

        for link_c in links:
            print(" " + " -> ".join(link_c))
        print("")  # To break lines

    def print_unknowns_targets_links(self, unknowns, targets, calibration=None, ignore_helpers=True):
        print(f"Links between {unknowns} and {targets}")
        print(" " + "-" * (len(f"Links between {unknowns} and {targets}")))
        unknown_target_net = xr.DataArray(np.zeros((len(unknowns), len(targets))),
                                          coords=[unknowns, targets],
                                          dims=["inputs", "outputs"])
        for u in unknowns:
            links = self.find_var_links(u, calibration=calibration, ignore_helpers=ignore_helpers)
            for link in links:
                if link[0] == u and link[-1] in targets:
                    unknown_target_net.loc[u, link[-1]] = 1.
        print(unknown_target_net)
        print("")  # To break lines

    # User-facing "analysis" methods
    def record_input_variables_paths(self, inputs_to_be_recorded, block_input_args):
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

    def find_var_links(self, var_name, calibration=None, ignore_helpers=True):
        # Find the indices of *direct* links between `var_name` and the affected `outputs`/`blocks` containing those
        # `outputs` and instantiate the initial list of links
        link_inds = np.nonzero(self._subset_by_vars(var_name).data)
        links = [[var_name] for i in range(len(link_inds[0]))]

        block_inds_sorted = graph.block_sort(self.blocks_as_list, calibration=calibration,
                                             ignore_helpers=ignore_helpers)
        required = graph.find_outputs_that_are_intermediate_inputs(self.blocks_as_list, ignore_helpers=ignore_helpers)
        intermediates = set()
        for ib in block_inds_sorted:
            # Note: This block is ordered before the bottom block of code since the intermediate outputs from a block
            #   `ib` do not need to be checked as inputs to that same block `ib`, only the subsequent blocks
            if intermediates:
                intm_link_inds = np.nonzero(self._subset_by_vars(list(intermediates)).data)
                # If there are `intermediate` inputs that have been recorded, we need to find the *indirect* links
                for iil, iilb in enumerate(intm_link_inds[0]):
                    # Check if those inputs are inputs to block `ib`
                    if ib == iilb:
                        # If so, repeat the logic from below, where you find the input-output var link
                        o_var = str(self._subset_by_vars(list(intermediates)).coords["outputs"][intm_link_inds[2][iil]].data)
                        intm_i_var = str(self._subset_by_vars(list(intermediates)).coords["inputs"][intm_link_inds[1][iil]].data)

                        # And add it to the set of all links, recording this new links' output if it hasn't appeared
                        # before and if it is an intermediate input
                        links.append([intm_i_var, o_var])
                        if o_var in required:
                            intermediates = intermediates.union(o_var)

            # Check if `var_name` enters into that block as an input, and if so add the link between it and the output
            # it directly affects, recording that output as an intermediate input if needed
            # Note: link_inds' 0-th row indicates the blocks and 1st row indicates the outputs
            for il, ilb in enumerate(link_inds[0]):
                if ib == ilb:
                    o_var = str(self._subset_by_vars(var_name).coords["outputs"][link_inds[1][il]].data)
                    links[il].append(o_var)
                    if o_var in required:
                        intermediates = intermediates.union(o_var)
        return _compose_dyad_links(links)

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


def _compose_dyad_links(links):
    links_composed = []
    inds_to_ignore = set()
    outputs = set()

    for il, link in enumerate(links):
        if il in inds_to_ignore:
            continue
        if links_composed:
            if link[0] in outputs:
                # Since `link` has as its input one of the outputs recorded from prior links in `links_composed`
                # search through the links in `links_composed` to see which links need to be extended with `link`
                # and the other links with the same input as `link`
                link_extensions = []
                # Potential link extensions will only be located past the stage il that we are at
                for il_e in range(il, len(links)):
                    if links[il_e][0] == link[0]:
                        link_extensions.append(links[il_e])
                        outputs = outputs.union([links[il_e][-1]])
                        inds_to_ignore = inds_to_ignore.union([il_e])

                links_to_add = []
                inds_to_omit = []
                for il_c, link_c in enumerate(links_composed):
                    if link_c[-1] == link[0]:
                        inds_to_omit.append(il_c)
                        links_to_add.extend([link_c + [ext[-1]] for ext in link_extensions])

                links_composed = [link_c for i, link_c in enumerate(links_composed) if i not in inds_to_omit] + links_to_add
            else:
                links_composed.append(link)
                outputs = outputs.union([link[-1]])
        else:
            links_composed.append(link)
            outputs = outputs.union([link[-1]])
    return links_composed


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
