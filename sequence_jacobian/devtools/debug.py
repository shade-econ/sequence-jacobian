"""Top-level tools to help users debug their sequence-jacobian code"""

from ..utilities import graph


def ensure_computability(blocks, calibration, unknowns, verbose=False):
    ensure_all_inputs_accounted_for(blocks, calibration, unknowns, verbose=verbose)
    # TODO: Ensure that all of the unknowns map in some way to at least one target each


# To ensure that no input argument that is required for one of the blocks to evaluate is missing
def ensure_all_inputs_accounted_for(blocks, calibration, unknowns, verbose=False):
    variables_accounted_for = set(unknowns.keys()).union(set(calibration.keys()))
    all_inputs = set().union(*[b.inputs for b in blocks])
    required = graph.find_outputs_that_are_intermediate_inputs(blocks)
    non_computed_inputs = all_inputs.difference(required)

    variables_unaccounted_for = non_computed_inputs.difference(variables_accounted_for)
    if variables_unaccounted_for:
        raise RuntimeError(f"The following variables were not listed as unknowns or provided as fixed variables/"
                           f"parameters: {variables_unaccounted_for}")
    if verbose:
        print("This DAG accounts for all inputs variables.")


def ensure_unknowns_and_targets_valid(blocks, unknowns, targets, verbose=False):
    pass


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
