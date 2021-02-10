"""Top-level tools to help users debug their sequence-jacobian code"""

import warnings
import numpy as np

from . import analysis
from ..utilities import graph


def ensure_computability(blocks, calibration=None, unknowns_ss=None,
                         exogenous=None, unknowns=None, targets=None, ss=None,
                         verbose=False, fragile=True, ignore_helpers=True):
    # Check if `calibration` and `unknowns` jointly have all of the required variables needed to be able
    # to calculate a steady state. If ss is provided, assume the user doesn't need to check this
    if calibration and unknowns_ss and not ss:
        ensure_all_inputs_accounted_for(blocks, calibration, unknowns_ss, verbose=verbose, fragile=fragile)

    # Check if unknowns and exogenous are not outputs of any blocks, and that targets are not inputs to any blocks
    if exogenous and unknowns and targets:
        ensure_unknowns_exogenous_and_targets_valid_candidates(blocks, exogenous + unknowns, targets,
                                                               verbose=verbose, fragile=fragile,
                                                               ignore_helpers=ignore_helpers,
                                                               calibration=calibration)

    # Check if there are any "broken" links between unknowns and targets, i.e. if there are any unknowns that don't
    # affect any targets, or if there are any targets that aren't affected by any unknowns
    if unknowns and targets and ss:
        ensure_unknowns_and_targets_are_valid(blocks, unknowns, targets, ss, verbose=verbose, fragile=fragile,
                                              ignore_helpers=ignore_helpers, calibration=calibration)


# To ensure that no input argument that is required for one of the blocks to evaluate is missing
def ensure_all_inputs_accounted_for(blocks, calibration, unknowns, verbose=False, fragile=True):
    variables_accounted_for = set(unknowns.keys()).union(set(calibration.keys()))
    all_inputs = set().union(*[b.inputs for b in blocks])
    required = graph.find_outputs_that_are_intermediate_inputs(blocks)
    non_computed_inputs = all_inputs.difference(required)

    variables_unaccounted_for = non_computed_inputs.difference(variables_accounted_for)
    if variables_unaccounted_for:
        if fragile:
            raise RuntimeError(f"The following variables were not listed as unknowns or provided as fixed variables/"
                               f"parameters: {variables_unaccounted_for}")
        else:
            warnings.warn(f"\nThe following variables were not listed as unknowns or provided as fixed variables/"
                          f"parameters: {variables_unaccounted_for}")
    if verbose:
        print("This DAG accounts for all inputs variables.")


def ensure_unknowns_exogenous_and_targets_valid_candidates(blocks, exogenous_unknowns, targets,
                                                           verbose=False, fragile=True,
                                                           ignore_helpers=True, calibration=None):
    cand_xu, cand_targets = find_candidate_unknowns_and_targets(blocks, ignore_helpers=ignore_helpers,
                                                                calibration=calibration)
    invalid_xu = []
    invalid_targ = []
    for xu in exogenous_unknowns:
        if xu not in cand_xu:
            invalid_xu.append(xu)
    for targ in targets:
        if targ not in cand_targets:
            invalid_targ.append(targ)
    if invalid_xu or invalid_targ:
        if fragile:
            raise RuntimeError(f"The following exogenous/unknowns are invalid candidates: {invalid_xu}\n"
                               f"The following targets are invalid candidates: {invalid_targ}")
        else:
            warnings.warn(f"\nThe following exogenous/unknowns are invalid candidates: {invalid_xu}\n"
                          f"The following targets are invalid candidates: {invalid_targ}")
    if verbose:
        print("The provided exogenous/unknowns and targets are all valid candidates for this DAG.")


def find_candidate_unknowns_and_targets(block_list, verbose=False, ignore_helpers=True, calibration=None):
    dep, inputs, outputs = graph.block_sort(block_list, return_io=True, ignore_helpers=ignore_helpers,
                                            calibration=calibration)
    required = graph.find_outputs_that_are_intermediate_inputs(block_list, ignore_helpers=ignore_helpers)

    # Candidate exogenous and unknowns (also includes parameters): inputs that are not outputs of any block
    # Candidate targets: outputs that are not inputs to any block
    cand_xu = inputs.difference(required)
    cand_targets = outputs.difference(required)

    if verbose:
        print(f"Candidate exogenous/unknowns: {cand_xu}\n"
              f"Candidate targets: {cand_targets}")

    return cand_xu, cand_targets


def ensure_unknowns_and_targets_are_valid(blocks, unknowns, targets, ss, verbose=False, fragile=True,
                                          calibration=None, ignore_helpers=True):
    io_net = analysis.BlockIONetwork(blocks)
    io_net.record_input_variables_paths(unknowns, ss, calibration=calibration, ignore_helpers=ignore_helpers)
    ut_net = io_net.find_unknowns_targets_links(unknowns, targets, calibration=calibration,
                                                ignore_helpers=ignore_helpers)
    broken_unknowns = []
    broken_targets = []
    for u in unknowns:
        if not np.any(ut_net.loc[u, :]):
            broken_unknowns.append(u)
    for t in targets:
        if not np.any(ut_net.loc[:, t]):
            broken_targets.append(t)
    if broken_unknowns or broken_targets:
        if fragile:
            raise RuntimeError(f"The following unknowns don't affect any targets: {broken_unknowns}\n"
                               f"The following targets aren't affected by any unknowns: {broken_targets}")
        else:
            warnings.warn(f"\nThe following unknowns don't affect any targets: {broken_unknowns}\n"
                          f"The following targets aren't affected by any unknowns: {broken_targets}")
    if verbose:
        print("This DAG does not contain any broken links between unknowns and targets.")
